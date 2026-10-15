"""Canonicalize and compare exactly the fixed eight scenarios using existing metrics.

Run without --candidate_llm to inspect Luca's checked-in baseline before HPC access.
The numerical tolerance for 'close' is explicit/configurable, not a statistical proof.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "scripts"))
from canonicalize_outputs import HEADER, process_file, repair
from metrics_comp.task_computemetrics import TaskMetrics
from run_state import atomic_write, state_path


def response_bodies(path):
    raw = Path(path).read_text(encoding="utf-8")
    headers = list(HEADER.finditer(raw))
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
        block = raw[header.end() : end]
        separator = block.rfind("\n-----------------------")
        if separator == -1:
            raise ValueError(f"Incomplete reply block in {path}")
        yield block[:separator].strip()


def select_files(config, llm, model, date=None):
    files = []
    for task, scenario in config["scenarios"]:
        folder = (
            ROOT
            / "inference"
            / f'{config["dataset"]}_expt_dataset/task_{task}/edge/output/{llm}out'
        )
        matches = [
            p
            for p in folder.glob("*.txt")
            if f"_{model}_" in p.name
            and f"_{scenario}.log_" in p.name
            and "_v2_" in p.name
            and (not date or f"_{date}_temp-" in p.name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {llm} output for {task}/{scenario}; found {len(matches)}. Select a date to disambiguate"
            )
        files.append((task, matches[0]))
    return files


def summarize(config, llm, model, date=None, historical=False):
    files = select_files(config, llm, model, date)
    totals = {
        "replies": 0,
        "empty": 0,
        "invalid_json_nonempty": 0,
        "repaired": 0,
        "unrepaired": 0,
    }
    telemetry = []
    telemetry_complete = True
    metrics = {}
    for task in dict(config["scenarios"]):
        selected = [(t, p) for t, p in files if t == task]
        output_paths = []
        for _, path in selected:
            for body in response_bodies(path):
                totals["replies"] += 1
                status = repair(body)[1]
                totals["empty"] += int(status == "empty")
                totals["invalid_json_nonempty"] += int(
                    status in {"repaired", "unrepaired"}
                )
                if status in {"repaired", "unrepaired"}:
                    totals[status] += 1
            destination = Path(
                str(path).replace(f"/output/{llm}out/", f"/output/{llm}-canonout/")
            )
            process_file(str(path), str(destination))
            relative = "./" + str(destination.relative_to(ROOT / "inference"))
            output_paths.append(
                (relative, destination.name, str(Path(relative).parent))
            )
            # Preserve the leading './': existing scorer reconstructs input paths from it.
            output_paths[-1] = (relative, destination.name, relative.rsplit("/", 1)[0])
            checkpoint = state_path(path)
            if checkpoint.exists():
                state = json.loads(checkpoint.read_text())
                if not state["complete"]:
                    raise ValueError(f"Incomplete experiment: {path}")
                actual = state["config"]
                expected = {
                    key: config[key]
                    for key in ("dataset", "model", "num_lines", "num_runs", "temp")
                }
                if any(actual.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        f"Validation generation settings do not match: {path}"
                    )
                if llm == "local" and (
                    actual.get("local", {}).get("max_tokens") != config["max_tokens"]
                    or actual.get("local", {}).get("extra_body") != config["extra_body"]
                    or actual.get("timezone") != config["timezone"]
                ):
                    raise ValueError(
                        f"Validation token/reasoning/timezone settings do not match: {path}"
                    )
                telemetry.extend(state["results"].values())
            else:
                telemetry_complete = False
        before = Path.cwd()
        try:
            os.chdir(ROOT / "inference")
            with redirect_stdout(io.StringIO()):
                tp, fp, fn, tn, _ = TaskMetrics(
                    task, "v2", output_paths, config["dataset"]
                ).compute_correctness()
            metrics[task] = {
                name: float(sum(values.values()))
                for name, values in [("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn)]
            }
        finally:
            os.chdir(before)
    expected = config["expected_chunks"] * config["num_runs"]
    if totals["replies"] != expected:
        raise ValueError(f'Expected {expected} replies, found {totals["replies"]}')
    total = totals["replies"]
    totals["invalid_json_rate_nonempty"] = totals["invalid_json_nonempty"] / total
    totals["invalid_json_rate_including_empty"] = (
        totals["invalid_json_nonempty"] + totals["empty"]
    ) / total
    if telemetry_complete and len(telemetry) == total:
        truncated = sum(bool(event["truncated"]) for event in telemetry)
        totals["refusals"] = sum(bool(event.get("refused")) for event in telemetry)
        totals["input_tokens"] = (
            sum(event["input_tokens"] for event in telemetry)
            if all(event["input_tokens"] is not None for event in telemetry)
            else None
        )
        totals["output_tokens"] = (
            sum(event["output_tokens"] for event in telemetry)
            if all(event["output_tokens"] is not None for event in telemetry)
            else None
        )
        source = "checkpointed final responses"
    elif historical:
        log = (ROOT / "gptoss_together_baseline.log").read_text()
        truncated = log.count("Error: Output token size exceeds max token count")
        source = "historical console log (per-call token counts unavailable)"
    else:
        truncated, source = None, "unavailable: no complete per-call telemetry"
    totals["truncated"] = truncated
    totals["truncation_rate"] = truncated / total if truncated is not None else None
    return {
        "llm": llm,
        "model": model,
        "quality": totals,
        "telemetry_source": source,
        "metrics": metrics,
        "files": [str(path.relative_to(ROOT)) for _, path in files],
    }


def compare(config, baseline, candidate):
    policy = config["validation"]
    counts = {}
    for task, reference in baseline["metrics"].items():
        # TP captures detections in the existing scorer, FP counts false positives.
        counts[task] = {
            name: {
                "delta": candidate["metrics"][task][name] - reference[name],
                "pass": abs(candidate["metrics"][task][name] - reference[name])
                <= policy["max_count_delta_per_task"],
            }
            for name in ("TP", "FP")
        }
    quality = candidate["quality"]
    rates = {}
    for metric, target, tolerance in [
        (
            "truncation_rate",
            policy["target_truncation_rate"],
            policy["truncation_rate_tolerance"],
        ),
        (
            "invalid_json_rate_nonempty",
            policy["target_invalid_json_rate"],
            policy["invalid_json_rate_tolerance"],
        ),
    ]:
        value = quality[metric]
        rates[metric] = {
            "value": value,
            "target": target,
            "tolerance": tolerance,
            "pass": value is not None and abs(value - target) <= tolerance,
        }
    return {
        "policy": policy,
        "counts": counts,
        "rates": rates,
        "pass": all(v["pass"] for task in counts.values() for v in task.values())
        and all(v["pass"] for v in rates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/gptoss_local_validation.json"
    )
    parser.add_argument(
        "--candidate_llm", help="e.g. local; omit for baseline-only report"
    )
    parser.add_argument("--candidate_date")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    logging.disable(logging.CRITICAL)
    os.environ["TZ"] = config["timezone"]
    import time

    time.tzset()
    baseline = summarize(
        config,
        "llama",
        config["baseline_model"],
        config["baseline_date"],
        historical=True,
    )
    report = {"baseline": baseline}
    if args.candidate_llm:
        candidate = summarize(
            config,
            args.candidate_llm,
            config["model"].replace("/", "-"),
            args.candidate_date,
        )
        report.update(
            candidate=candidate, comparison=compare(config, baseline, candidate)
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        atomic_write(args.output, rendered)
    print(rendered)
    return int("comparison" in report and not report["comparison"]["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
