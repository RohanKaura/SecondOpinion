"""Run the fixed Luca subset on any locally served vLLM model; no model-specific code."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from helpers.helpers import create_output_filename
from run_state import legacy_chunks


def load_plan(config, base_url=None, output_root=None):
    plan = []
    chunks = 0
    for task, scenario in config["scenarios"]:
        matches = list(
            (
                ROOT
                / "inference"
                / f'{config["dataset"]}_expt_dataset'
                / f"task_{task}"
                / "edge/input"
            ).glob(f"*_{scenario}.log")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one input for {task}/{scenario}, found {len(matches)}"
            )
        path = matches[0]
        with path.open() as handle:
            lines = sum(1 for _ in handle)
        chunks += len(list(legacy_chunks(1, lines, config["num_lines"])))
        prompt = f'{task}-investigation_v2_{task}_{config["dataset"]}_edge'
        command = [
            sys.executable,
            "task-query-v2.py",
            "--dataset",
            config["dataset"],
            "--task",
            task,
            "--log_fpath",
            str(path.relative_to(ROOT / "inference")),
            "--num_lines",
            str(config["num_lines"]),
            "--prompt_key",
            prompt,
            "--num_runs",
            str(config["num_runs"]),
            "--temp",
            str(config["temp"]),
            "--llm",
            "local",
            "--model",
            config["model"],
            "--base_url",
            base_url or config["base_url"],
            "--max_tokens",
            str(config["max_tokens"]),
            "--timeout",
            str(config.get("timeout", 180)),
            "--max_retries",
            str(config.get("max_retries", 2)),
            "--extra_body",
            json.dumps(config.get("extra_body", {})),
        ]
        if output_root:
            date = datetime.now(ZoneInfo(config["timezone"])).date().isoformat()
            name = create_output_filename(
                prompt,
                str(path),
                date,
                float(config["temp"]),
                config["num_runs"],
                config["model"],
            )
            command += ["--output_file", str(Path(output_root).resolve() / task / name)]
        plan.append(command)
    if chunks != config["expected_chunks"]:
        raise ValueError(
            f'Subset changed: {chunks} chunks; expected {config["expected_chunks"]}'
        )
    return plan, chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/gptoss_local_validation.json"
    )
    parser.add_argument("--base_url")
    parser.add_argument(
        "--output_root",
        type=Path,
        help="Optional separate transcript folder for experiments/tests",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run each harness preflight without model requests or output writes",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan, chunks = load_plan(config, args.base_url, args.output_root)
    print(
        f'{len(plan)} scenarios / {chunks} chunks / {chunks * config["num_runs"]} calls',
        flush=True,
    )
    env = {**os.environ, "TZ": config["timezone"]}
    failed = False
    for command in plan:
        if args.resume:
            command.append("--resume")
        if args.dry_run:
            command.append("--dry_run")
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT / "inference", env=env)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
