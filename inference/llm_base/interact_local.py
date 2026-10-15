"""One adapter for any chat model served by a local vLLM HTTP server.

This is a client only: it needs neither CUDA nor the vllm package on the Mac.
Model-specific sampling and chat-template settings go in extra_body.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI


def json_object(value):
    """argparse-compatible JSON object, also used for generated commands."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("extra_body must be a JSON object")
    reserved = {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "n",
    }
    if reserved.intersection(value):
        raise ValueError(
            f"extra_body cannot override {sorted(reserved.intersection(value))}"
        )
    # Reject NaN/Infinity, which are not valid JSON request values.
    json.dumps(value, allow_nan=False)
    return value


@dataclass
class ModelReply:
    content: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    refusal: str | None = None
    request_id: str | None = None

    def telemetry(self):
        result = asdict(self)
        result.pop("content")
        result["empty_response"] = not bool(self.content.strip())
        result["refused"] = bool(self.refusal) or self.finish_reason == "content_filter"
        result["truncated"] = self.finish_reason == "length"
        try:
            parsed = json.loads(self.content)
            result["invalid_json"] = not isinstance(parsed, (dict, list))
        except (ValueError, TypeError):
            result["invalid_json"] = True
        return result

    def require_content(self):
        if self.refusal or self.finish_reason == "content_filter":
            raise ValueError(
                "Model refused or filtered the request; see per-call metadata"
            )
        if not self.content.strip():
            raise ValueError(
                "Model returned an empty content field; check reasoning parser and token budget"
            )
        # Keep truncated/invalid JSON responses for baseline parity; flag them in telemetry.


class LocalClient:
    def __init__(
        self,
        base_url=None,
        max_tokens=16384,
        timeout=180,
        max_retries=2,
        extra_body=None,
    ):
        load_dotenv()
        self.base_url = (
            base_url
            or os.environ.get("LOCAL_LLM_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) endpoint without credentials, query, or fragment"
            )
        if (
            max_tokens < 1
            or max_retries < 0
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "max_tokens/timeout must be positive and max_retries nonnegative"
            )
        self.max_tokens = max_tokens
        self.extra_body = json_object(extra_body or {})
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("LOCAL_LLM_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, model, message, temp):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=temp,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        if not response.choices:
            raise ValueError("Server returned no completion choices")
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return ModelReply(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            refusal=getattr(choice.message, "refusal", None),
            request_id=getattr(response, "_request_id", None) or response.id,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()


def add_local_arguments(parser):
    parser.add_argument(
        "--base_url",
        default=None,
        help="vLLM endpoint; default LOCAL_LLM_BASE_URL or http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Local backend output-token budget (includes reasoning)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Local backend request timeout in seconds",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="SDK retries for transient local-server errors",
    )
    parser.add_argument(
        "--extra_body",
        type=json_object,
        default={},
        help='Local request JSON, e.g. {"reasoning_effort":"medium"}',
    )


def local_options(args):
    return {
        name: getattr(args, name)
        for name in ("base_url", "max_tokens", "timeout", "max_retries", "extra_body")
    }
