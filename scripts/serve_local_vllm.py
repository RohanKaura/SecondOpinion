"""Prepare or start vLLM on an allocated GPU node, bound only to localhost.

Use the HPC's compatible vLLM server environment, separate from the client .venv.
No scheduler, partition, account, GPU type, or module names are assumed here.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def server_command(config, tensor_parallel_size=None):
    endpoint = urlsplit(config["base_url"])
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"localhost", "127.0.0.1"}
        or endpoint.path.rstrip("/") != "/v1"
        or endpoint.port is None
        or endpoint.username
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(
            "Server config must use http://127.0.0.1:PORT/v1 (same GPU node)"
        )
    settings = config["server"]
    tp = (
        tensor_parallel_size
        if tensor_parallel_size is not None
        else settings["tensor_parallel_size"]
    )
    if tp < 1:
        raise ValueError("tensor_parallel_size must be positive")
    return [
        "vllm",
        "serve",
        config["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(endpoint.port or 8000),
        "--served-model-name",
        config["model"],
        "--max-model-len",
        str(settings["max_model_len"]),
        "--tensor-parallel-size",
        str(tp),
        "--max-num-seqs",
        str(settings["max_num_seqs"]),
        "--generation-config",
        "vllm",
        *settings.get("extra_args", []),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/gptoss_local_validation.json"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, help="Number of GPUs allocated on this node"
    )
    parser.add_argument(
        "--print_command", action="store_true", help="Do not start vLLM"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    command = server_command(config, args.tensor_parallel_size)
    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0
    if not shutil.which("vllm"):
        parser.error(
            "vllm is not installed here. Activate the GPU node's vLLM server environment first"
        )
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
