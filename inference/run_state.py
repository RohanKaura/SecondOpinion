"""Durable per-response checkpoints and unchanged, scorer-compatible transcripts."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from llm_base.interact_local import ModelReply


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def legacy_chunks(start, end, size):
    """Exactly the original v2 inclusive indices, including its one-line overlap.

    Deliberately retained for Luca's parity experiment, not a chunking correction.
    """
    for line in range(start, end + (1 if start == end else 0), size):
        yield line - 1, min(line + size - 1, end - 1)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_path(output):
    output = Path(output)
    return output.parent / ".run_state" / (output.name + ".state.json")


def find_resume_output(default_output, config):
    """Find the same experiment across dates without guessing among multiple runs."""
    candidates = []
    for path in state_path(default_output).parent.glob("*.state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config") == config:
            candidates.append(
                Path(default_output).parent / path.name.removesuffix(".state.json")
            )
    if len(candidates) > 1:
        raise ValueError("Multiple matching checkpoints; select one with --output_file")
    return candidates[0] if candidates else Path(default_output)


class RunState:
    def __init__(self, output, config, resume=False):
        self.output = Path(output)
        self.config = config
        self.resume = resume
        self.path = state_path(output)
        self.calls_path = self.path.with_name(self.output.name + ".calls.jsonl")
        self.partial_path = self.path.with_name(self.output.name + ".partial.txt")

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Another process is writing {self.output}") from None
            try:
                if self.path.exists():
                    if not self.resume:
                        raise ValueError(
                            "Checkpoint already exists; use --resume or a different --output_file"
                        )
                    self.state = json.loads(self.path.read_text(encoding="utf-8"))
                    if (
                        self.state.get("schema") != 1
                        or self.state.get("config") != self.config
                    ):
                        raise ValueError(
                            "Checkpoint configuration differs (input, prompt, model, or generation settings)"
                        )
                elif self.output.exists() or self.calls_path.exists():
                    raise ValueError(
                        "Output/call log exists without a checkpoint; use a new --output_file. Legacy outputs cannot be resumed safely"
                    )
                else:
                    self.state = {
                        "schema": 1,
                        "config": self.config,
                        "results": {},
                        "complete": False,
                    }
                    self.save()
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save(self):
        atomic_write(
            self.path, json.dumps(self.state, ensure_ascii=False, indent=2) + "\n"
        )

    def log_call(self, event):
        # One event per logical call; the SDK retries transient transport/server failures.
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def render(self):
        blocks = []
        for result in sorted(
            self.state["results"].values(), key=lambda r: (r["start_lno"], r["run"])
        ):
            # Byte-compatible with helpers.write_output. Run identity lives in the checkpoint.
            blocks.append(
                f'\n\n****Starting line num: {result["start_lno"]}. '
                f'Num lines: {self.config["num_lines"]}. {result["time_range"]}****\n\n'
                f'{result["content"]}\n\n'
                f'-----------------------{self.config["llm"]}------------------------\n'
            )
        atomic_write(
            self.output if self.state["complete"] else self.partial_path,
            "".join(blocks),
        )


def run_inference(
    output,
    config,
    chunks,
    make_message,
    time_range,
    invoke,
    resume=False,
    fail_fast=False,
):
    """Persist each successful run; failures leave holes to retry on --resume.

    Only complete experiments are published in the scorer's output directory.
    Every completed response, including empty/refused/truncated answers, is retained
    and flagged for parity analysis. Only request exceptions are retried on resume;
    resampling low-quality responses would bias comparison with Luca's baseline.
    """
    failed = skipped = completed = quality_failures = 0
    with RunState(output, config, resume).locked() as store:
        for start, end in chunks:
            pending = [
                run
                for run in range(1, config["num_runs"] + 1)
                if f"{start}:{run}" not in store.state["results"]
            ]
            skipped += config["num_runs"] - len(pending)
            if not pending:
                continue
            # Compute input metadata before sending paid/expensive requests.
            message = make_message(start, end)
            timestamps = time_range(start, end)
            for run in pending:
                event = {
                    "call_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "start_lno": start + 1,
                    "end_lno": end + 1,
                    "run": run,
                    "llm": config["llm"],
                    "model": config["model"],
                    **{key: None for key in ModelReply("").telemetry()},
                }
                started = time.monotonic()
                reply = None
                try:
                    reply = invoke(message)
                    if isinstance(reply, str):
                        reply = ModelReply(reply)
                    event.update(reply.telemetry())
                except KeyboardInterrupt:
                    event.update(
                        status="interrupted",
                        error_type="KeyboardInterrupt",
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                    store.log_call(event)
                    raise
                except Exception as error:
                    event.update(
                        status="error",
                        error_type=type(error).__name__,
                        error=str(error)[:2000],
                        http_status=getattr(error, "status_code", None),
                    )
                    failed += 1
                else:
                    event["status"] = "ok"
                    try:
                        reply.require_content()
                    except ValueError as error:
                        event.update(
                            status="response_error",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                        quality_failures += 1
                event["elapsed_seconds"] = round(time.monotonic() - started, 3)
                store.log_call(event)
                if event["status"] != "error":
                    store.state["results"][f"{start}:{run}"] = {
                        **event,
                        "content": reply.content,
                        "time_range": timestamps,
                    }
                    store.save()
                    completed += 1
                print(
                    f'Lines {start + 1}-{end + 1}, run {run}: {event["status"]}',
                    flush=True,
                )
                if fail_fast and event["status"] == "error":
                    store.render()
                    print(
                        f"Stopped after failure; details: {store.calls_path}",
                        flush=True,
                    )
                    return 1
        expected = len(chunks) * config["num_runs"]
        store.state["complete"] = len(store.state["results"]) == expected
        store.save()
        store.render()
        print(
            f"Completed {completed}, skipped {skipped}, request failures {failed}, empty/refused responses {quality_failures}; expected {expected} responses.\nOutput: {store.output if store.state['complete'] else store.partial_path}\nCalls: {store.calls_path}",
            flush=True,
        )
    return int(failed > 0)
