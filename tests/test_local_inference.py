"""Offline integration tests: real HTTP + SDK + harness, never model inference."""

from contextlib import contextmanager, redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "scripts"))
from helpers import helpers
from llm_base.interact_local import LocalClient, ModelReply, json_object
from run_state import (
    RunState,
    find_resume_output,
    legacy_chunks,
    run_inference,
    state_path,
)
from run_local_subset import load_plan
from serve_local_vllm import server_command
from validate_local_baseline import response_bodies, summarize, compare
import validate_local_baseline


def completion(content="{}", finish_reason="stop", refusal=None):
    return {
        "id": "mock-response",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


@contextmanager
def server(replies):
    requests = []
    queue = list(replies)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                {
                    "path": self.path,
                    "body": json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    ),
                    "auth": self.headers.get("Authorization"),
                }
            )
            if not queue:
                code, body = 500, {"error": {"message": "Unexpected extra request"}}
            else:
                reply = queue.pop(0)
                code, body = reply if isinstance(reply, tuple) else (200, reply)
            encoded = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", requests
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


class LocalAdapterTests(unittest.TestCase):
    def test_payload_and_metadata_for_generic_models(self):
        with server([completion('{"ok": true}', "length")]) as (url, requests):
            with patch.dict(os.environ, {"LOCAL_LLM_API_KEY": "test-local-key"}):
                with LocalClient(
                    base_url=url,
                    max_tokens=77,
                    extra_body={"reasoning_effort": "medium", "top_k": 20},
                ) as client:
                    reply = client.generate(
                        "arbitrary/vendor-model", "unaltered prompt", 0.4
                    )
            request = requests[0]
            self.assertEqual(request["path"], "/v1/chat/completions")
            self.assertEqual(
                request["body"],
                {
                    "model": "arbitrary/vendor-model",
                    "messages": [{"role": "user", "content": "unaltered prompt"}],
                    "temperature": 0.4,
                    "max_tokens": 77,
                    "reasoning_effort": "medium",
                    "top_k": 20,
                },
            )
            self.assertEqual(request["auth"], "Bearer test-local-key")
            self.assertEqual(reply.input_tokens, 123)
            self.assertEqual(reply.output_tokens, 45)
            self.assertEqual(reply.reasoning_tokens, 20)
            self.assertTrue(reply.telemetry()["truncated"])

    def test_missing_usage_empty_and_refusal(self):
        fixture = completion("", refusal="Cannot comply")
        fixture.pop("usage")
        with server([fixture]) as (url, _):
            with LocalClient(base_url=url) as client:
                reply = client.generate("model", "prompt", 0)
        self.assertIsNone(reply.input_tokens)
        self.assertTrue(reply.telemetry()["empty_response"])
        self.assertEqual(reply.refusal, "Cannot comply")
        with self.assertRaisesRegex(ValueError, "refused"):
            reply.require_content()

    def test_transient_error_is_retried_but_bad_request_is_not(self):
        from openai import BadRequestError

        with server([(500, {"error": {"message": "busy"}}), completion()]) as (
            url,
            requests,
        ):
            with LocalClient(base_url=url, max_retries=1) as client:
                self.assertEqual(client.generate("model", "prompt", 0).content, "{}")
            self.assertEqual(len(requests), 2)
        with server([(400, {"error": {"message": "context too long"}})]) as (
            url,
            requests,
        ):
            with LocalClient(base_url=url, max_retries=2) as client:
                with self.assertRaises(BadRequestError):
                    client.generate("model", "prompt", 0)
            self.assertEqual(len(requests), 1)

    def test_timeout_and_connection_failure(self):
        from openai import APITimeoutError, APIConnectionError

        with LocalClient(
            base_url="http://127.0.0.1:1/v1", timeout=0.1, max_retries=0
        ) as client:
            with self.assertRaises(APIConnectionError):
                client.generate("model", "prompt", 0)
        import httpx

        with LocalClient(
            base_url="http://127.0.0.1:1/v1", timeout=0.1, max_retries=0
        ) as client:
            with patch.object(
                client.client._client, "send", side_effect=httpx.ReadTimeout("test")
            ):
                with self.assertRaises(APITimeoutError):
                    client.generate("model", "prompt", 0)

    def test_configuration_rejects_overrides_and_bad_urls(self):
        for value in ["[]", '{"model":"other"}', '{"stream":true}', '{"top_k":NaN}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                json_object(value)
        for url in [
            "ftp://localhost",
            "http://user:secret@localhost/v1",
            "http://localhost/v1?key=secret",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalClient(base_url=url)

    def test_local_routing_has_no_cloud_initialization(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "OPENAI_API_KEY",
                "TOGETHER_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
            }
        }
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import inference_helpers, sys; assert 'llm_base.interact_gemini' not in sys.modules; assert 'llm_base.interact_llama' not in sys.modules",
            ],
            cwd=ROOT / "inference",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(check.stdout, "")


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "results_v2_test.txt"
        self.config = {
            "llm": "local",
            "model": "model",
            "num_lines": 400,
            "num_runs": 2,
        }
        self.chunks = [(0, 400), (400, 499)]

    def execute(self, invoke, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run_inference(
                self.output,
                self.config,
                self.chunks,
                lambda a, b: f"{a}:{b}",
                lambda a, b: "start-to-end",
                invoke,
                **kwargs,
            )

    def test_resume_only_missing_pairs_and_output_bytes(self):
        responses = iter(["first", RuntimeError("server down"), "third", "fourth"])

        def invoke(_):
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return item

        self.assertEqual(self.execute(invoke), 1)
        self.assertFalse(self.output.exists(), "Partial experiment must not be scored")
        state = json.loads(state_path(self.output).read_text())
        self.assertEqual(len(state["results"]), 3)
        calls = []
        self.assertEqual(
            self.execute(
                lambda message: calls.append(message) or "second", resume=True
            ),
            0,
        )
        self.assertEqual(calls, ["0:400"])
        reference = self.output.parent / "expected.txt"
        for content, line in [
            ("first", 1),
            ("second", 1),
            ("third", 401),
            ("fourth", 401),
        ]:
            helpers.write_output(
                content, str(reference), line, 400, "local", "start-to-end"
            )
        self.assertEqual(self.output.read_bytes(), reference.read_bytes())
        self.assertEqual(
            self.execute(lambda _: self.fail("Completed call repeated"), resume=True), 0
        )
        events = [
            json.loads(line)
            for line in state_path(self.output)
            .with_name(self.output.name + ".calls.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(len(events), 5)
        self.assertEqual(sum(e["status"] == "error" for e in events), 1)

    def test_interrupt_preserves_previous_response(self):
        responses = iter(["saved", KeyboardInterrupt()])

        def invoke(_):
            item = next(responses)
            if isinstance(item, BaseException):
                raise item
            return item

        with self.assertRaises(KeyboardInterrupt):
            self.execute(invoke)
        self.assertEqual(
            len(json.loads(state_path(self.output).read_text())["results"]), 1
        )
        self.assertEqual(self.execute(lambda _: "resumed", resume=True), 0)

    def test_no_overwrite_or_configuration_mix(self):
        self.execute(lambda _: "{}")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.execute(lambda _: "{}")
        self.config["model"] = "different-model"
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.execute(lambda _: "{}", resume=True)

    def test_completed_empty_refusal_and_truncation_preserved_for_parity(self):
        responses = iter(
            [
                ModelReply("", "length", output_tokens=16384),
                ModelReply("", "stop", refusal="refused"),
                ModelReply('{"broken":', "length"),
                ModelReply("{}", "stop"),
            ]
        )
        self.assertEqual(self.execute(lambda _: next(responses)), 0)
        events = list(
            json.loads(state_path(self.output).read_text())["results"].values()
        )
        self.assertEqual(sum(e["empty_response"] for e in events), 2)
        self.assertEqual(sum(e["truncated"] for e in events), 2)
        self.assertEqual(sum(e["status"] == "response_error" for e in events), 2)
        self.assertTrue(self.output.exists())
        self.execute(
            lambda _: self.fail("Completed response should not be resampled"),
            resume=True,
        )

    def test_fail_fast_and_exclusive_lock(self):
        with RunState(self.output, self.config).locked():
            with self.assertRaisesRegex(ValueError, "Another process"):
                with RunState(self.output, self.config, resume=True).locked():
                    self.fail("Acquired same output twice")
        calls = []

        def fail(message):
            calls.append(message)
            raise RuntimeError("unavailable")

        self.assertEqual(self.execute(fail, resume=True, fail_fast=True), 1)
        self.assertEqual(len(calls), 1)

    def test_resume_across_dates(self):
        self.execute(lambda _: "{}")
        self.assertEqual(
            find_resume_output(self.output.parent / "tomorrow.txt", self.config),
            self.output,
        )

    def test_legacy_chunk_boundaries_are_preserved(self):
        for start, end, size in [
            (1, n, 400) for n in (1, 2, 399, 400, 401, 402, 800, 801, 2120, 6569)
        ] + [(12, 450, 25)]:
            original = [
                (i - 1, min(i + size - 1, end - 1))
                for i in range(start, end + (1 if start == end else 0), size)
            ]
            self.assertEqual(list(legacy_chunks(start, end, size)), original)


class HarnessTests(unittest.TestCase):
    def test_server_configs_are_localhost_and_generic(self):
        for filename in ("gptoss_local_validation.json", "qwen38_27b_local.json"):
            config = json.loads((ROOT / "configs" / filename).read_text())
            command = server_command(config, tensor_parallel_size=2)
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
            self.assertEqual(command[command.index("--generation-config") + 1], "vllm")
            self.assertIn(config["model"], command)
        with self.assertRaises(ValueError):
            server_command(config, tensor_parallel_size=0)
        config["base_url"] = "http://0.0.0.0:8000/v1"
        with self.assertRaises(ValueError):
            server_command(config)

    def test_full_88_response_replay_matches_lucas_prompts_outputs_and_scores(self):
        config = json.loads((ROOT / "configs/gptoss_local_validation.json").read_text())
        fixtures = []
        originals = []
        expected_messages = []
        golden = json.loads(
            (ROOT / "tests/fixtures/gptoss_prompt_hashes.json").read_text()
        )["scenarios"]
        for task, scenario in config["scenarios"]:
            task_dir = ROOT / f"inference/labgen_expt_dataset/task_{task}"
            (original,) = (task_dir / "edge/output/llamaout").glob(
                f'*_openai-gpt-oss-120b_*_{scenario}.log_{config["baseline_date"]}_*.txt'
            )
            originals.append((task, original))
            bodies = list(response_bodies(original))
            fixtures.extend(
                completion(body, "length" if not body else "stop") for body in bodies
            )
            (reference,) = [
                item
                for item in golden
                if item["task"] == task and item["scenario"] == scenario
            ]
            for chunk in reference["chunks"]:
                expected_messages.extend([chunk["message_sha256"]] * 2)
        self.assertEqual(len(fixtures), 88)
        with tempfile.TemporaryDirectory() as directory, server(fixtures) as (
            url,
            requests,
        ):
            temporary = Path(directory)
            for task in {task for task, _ in config["scenarios"]}:
                destination = temporary / f"inference/labgen_expt_dataset/task_{task}"
                (destination / "edge").mkdir(parents=True)
                source = ROOT / f"inference/labgen_expt_dataset/task_{task}"
                (destination / "edge/input").symlink_to(
                    source / "edge/input", target_is_directory=True
                )
                groundtruth = source / f"groundtruth-{task}.jsonl"
                if groundtruth.exists():
                    (destination / groundtruth.name).symlink_to(groundtruth)
            commands, _ = load_plan(config, url)
            for command in commands:
                command[1] = str(ROOT / "inference/task-query-v2.py")
                result = subprocess.run(
                    command,
                    cwd=temporary / "inference",
                    env={**os.environ, "TZ": config["timezone"]},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(requests), 88)

            # A transcript with different request settings must not pass baseline validation.
            state_file = next((temporary / "inference").rglob("*.state.json"))
            original_state = state_file.read_text()
            changed = json.loads(original_state)
            changed["config"]["local"]["max_tokens"] = 8192
            state_file.write_text(json.dumps(changed))
            with patch.object(validate_local_baseline, "ROOT", temporary):
                with self.assertRaisesRegex(ValueError, "settings do not match"):
                    summarize(config, "local", config["baseline_model"])
            state_file.write_text(original_state)
            self.assertEqual(
                [
                    hashlib.sha256(
                        r["body"]["messages"][0]["content"].encode()
                    ).hexdigest()
                    for r in requests
                ],
                expected_messages,
            )
            for task, original in originals:
                local_dir = (
                    temporary
                    / f"inference/labgen_expt_dataset/task_{task}/edge/output/localout"
                )
                scenario = original.name.split(".log_")[0].split("_")[-1]
                (local,) = local_dir.glob(f"*_{scenario}.log_*.txt")
                self.assertEqual(
                    local.read_text(),
                    original.read_text().replace(
                        "-----------------------llama------------------------",
                        "-----------------------local------------------------",
                    ),
                )
            with redirect_stdout(io.StringIO()):
                baseline = summarize(
                    config,
                    "llama",
                    config["baseline_model"],
                    config["baseline_date"],
                    historical=True,
                )
                with patch.object(validate_local_baseline, "ROOT", temporary):
                    candidate = summarize(config, "local", config["baseline_model"])
            self.assertEqual(candidate["metrics"], baseline["metrics"])
            self.assertTrue(compare(config, baseline, candidate)["pass"])
            # Completed runs make zero new HTTP calls on resume.
            for command in commands:
                result = subprocess.run(
                    command + ["--resume"],
                    cwd=temporary / "inference",
                    env={**os.environ, "TZ": config["timezone"]},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(requests), 88)

    def test_real_cli_all_tasks_and_optc_without_cloud_keys(self):
        cases = [
            ("labgen", task)
            for task in ("classification", "exfiltration", "lm", "persistence")
        ] + [("optc", "lm")]
        with server([completion()] * len(cases)) as (
            url,
            requests,
        ), tempfile.TemporaryDirectory() as folder:
            for dataset, task in cases:
                with self.subTest(dataset=dataset, task=task):
                    path = next(
                        (
                            ROOT
                            / f"inference/{dataset}_expt_dataset/task_{task}/edge/input"
                        ).glob("*.log")
                    )
                    target = (
                        Path(folder)
                        / f"{dataset}_expt_dataset/task_{task}/edge/input"
                        / path.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(path)
                    command = [
                        sys.executable,
                        str(ROOT / "inference/task-query-v2.py"),
                        "--dataset",
                        dataset,
                        "--task",
                        task,
                        "--log_fpath",
                        str(target),
                        "--num_lines",
                        "400",
                        "--start_lno",
                        "1",
                        "--end_lno",
                        "2",
                        "--prompt_key",
                        f"{task}-investigation_v2_{task}_{dataset}_edge",
                        "--num_runs",
                        "1",
                        "--temp",
                        "0",
                        "--llm",
                        "local",
                        "--model",
                        "arbitrary/model",
                        "--base_url",
                        url,
                    ]
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if k
                        not in {
                            "OPENAI_API_KEY",
                            "TOGETHER_API_KEY",
                            "GOOGLE_APPLICATION_CREDENTIALS",
                        }
                    }
                    result = subprocess.run(
                        command,
                        cwd=folder,
                        env={**env, "TZ": "America/Chicago"},
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stderr + result.stdout
                    )
                    self.assertNotIn("Sending this query to", result.stdout)
            self.assertEqual(len(requests), len(cases))

    def test_generated_commands_round_trip_extra_parameters(self):
        extra = {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20}
        result = subprocess.run(
            [
                sys.executable,
                "inference/generate_commands_v2.py",
                "--dataset",
                "labgen",
                "--task",
                "lm",
                "--rep",
                "edge",
                "--type",
                "attack",
                "--llm",
                "local",
                "--model",
                "vendor/name with spaces",
                "--num_runs",
                "2",
                "--temp",
                "0.25",
                "--extra_body",
                json.dumps(extra),
                "--resume",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            if line:
                command = shlex.split(line)
                self.assertEqual(
                    json.loads(command[command.index("--extra_body") + 1]), extra
                )
                self.assertEqual(
                    command[command.index("--model") + 1], "vendor/name with spaces"
                )
                self.assertIn("--resume", command)

    def test_subset_configuration_and_historical_quality(self):
        config = json.loads((ROOT / "configs/gptoss_local_validation.json").read_text())
        commands, chunks = load_plan(config)
        self.assertEqual((len(commands), chunks), (8, 44))
        for command in commands:
            self.assertEqual(
                json.loads(command[command.index("--extra_body") + 1]),
                {"reasoning_effort": "medium"},
            )
        with redirect_stdout(io.StringIO()):
            baseline = summarize(
                config,
                "llama",
                config["baseline_model"],
                config["baseline_date"],
                historical=True,
            )
        self.assertEqual(baseline["quality"]["replies"], 88)
        self.assertEqual(baseline["quality"]["invalid_json_nonempty"], 9)
        self.assertEqual(baseline["quality"]["truncated"], 2)
        self.assertTrue(compare(config, baseline, baseline)["pass"])
        changed = json.loads(json.dumps(baseline))
        changed["metrics"]["lm"]["FP"] += 2
        self.assertFalse(compare(config, baseline, changed)["pass"])


if __name__ == "__main__":
    unittest.main()
