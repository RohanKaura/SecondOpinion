# Local vLLM inference and the SJSU handoff

The client work is ready to test against a real vLLM server. No SJSU login, scheduler
submission, model-weight download, or GPU inference has been performed. All local
HTTP tests use simulated model responses; replay parity does not establish real
model accuracy.

## Environment and local verification

The Mac client is `/Users/rohan/Projects/SecondOpinion/.venv`: Python 3.10.22,
pip 26.2.1. All dependency pins install and `pip check` passes. Pandas 2.3.3,
NumPy 2.2.6 and PyTorch 2.10.0 import successfully; CUDA is unavailable on this Mac.
The earlier Black change remains `25.12.0` -> `25.11.0`. Both releases exist;
25.12.0 could not run under the original Python 3.9 environment. Added
`tiktoken==0.14.0`, which the existing chunk-token audit imports but did not list.

From the repository root:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -v
python scripts/run_local_subset.py --dry_run
python scripts/run_local_subset.py --config configs/qwen38_27b_local.json --dry_run
python scripts/serve_local_vllm.py --print_command
python scripts/validate_local_baseline.py --output baseline-reference.json
```

The tests cover actual localhost HTTP requests through the installed SDK,
configuration and metadata, transient retries, bad requests, timeout/disconnection,
all four Labgen tasks and OpTC edge inference, interrupted and partial jobs,
exclusive output locks, configuration mismatch protection, cross-date resume,
shell quoting, and the entire 88-response GPT-OSS replay. The replay compares
all transmitted prompt hashes with hashes captured from the original harness,
compares transcript bytes (apart from the backend footer), canonicalizes, and
compares the existing scorer's results. Resuming a complete replay sends no requests.

## Current flow and compatibility

Luca's unchanged `scripts/run_gptoss_subset.sh` selects eight Labgen edge scenarios,
sets `TZ=America/Chicago`, and calls `inference/task-query-v2.py` with
`--llm llama --model openai/gpt-oss-120b`. The old shared helper routes those
requests to Together AI. The model name does not determine the provider.

The new v2 path is:

1. Select the scenario's existing OS and task prompt in `task-query-v2.py`.
2. Read the same inclusive line indices and pass the same Python list of lines to
   the unchanged `Prompt.populate`. It is intentionally not converted to joined text.
3. Send the prompt through one generic `LocalClient` to `/v1/chat/completions`.
4. Save the response and metadata durably for each chunk/run pair.
5. Publish the original text transcript format when the scenario's requests finish.
6. Canonicalize with `scripts/canonicalize_outputs.py`, then use the existing v2
   parsers and metrics. No prompt templates, datasets, parser logic, or metric
   formulas were changed. The canonicalizer only gained proper file closing.

Existing GPT, Gemini and Together routes remain available. Cloud providers are now
imported only when selected, so local inference does not initialize Vertex AI or
require cloud credentials. The new CLI/state workflow is for v2; v1 remains legacy.

## Generic local client options

Run inference from `inference/`. Existing task, input, prompt and sampling options
still apply. New controls are:

| Option | Meaning |
| --- | --- |
| `--llm local` | Select the shared vLLM adapter |
| `--model NAME` | Exact name advertised by the server |
| `--base_url URL` | Default `LOCAL_LLM_BASE_URL`, then `http://127.0.0.1:8000/v1` |
| `--max_tokens N` | Output budget; default 16,384, including reasoning tokens |
| `--temp NUMBER` | Existing temperature flag |
| `--extra_body JSON` | Extra request fields, including reasoning effort, sampling and chat-template options |
| `--timeout SECONDS` | Request timeout; default 180, validation configs use 600 |
| `--max_retries N` | SDK transient-error retries; default 2 |
| `--resume` | Skip completed chunk/run pairs and retry request exceptions |
| `--output_file PATH` | Explicit transcript/checkpoint selection |
| `--fail_fast` | Stop on first request exception |
| `--dry_run` | Print the validated plan, without requests or output writes |

`LOCAL_LLM_API_KEY` supplies the local server key if needed; otherwise the client
uses `EMPTY`. It does not fall back to an OpenAI/Together cloud API key or endpoint.
Credentials are not embedded in config files. `extra_body` cannot override model,
messages, temperature, token budget, number of completions, or streaming mode.

Example extra settings:

```bash
--extra_body '{"reasoning_effort":"medium"}'
--extra_body '{"chat_template_kwargs":{"enable_thinking":true},"top_k":20}'
```

`generate_commands_v2.py` also accepts the local options, `--num_runs`, `--temp`,
`--resume` and `--dry_run`, and safely quotes the generated shell arguments. Removed
model-name/provider-name checks, which incorrectly rejected Together-hosted GPT-OSS.

## Logs and restart behavior

For each transcript `NAME.txt`, the output folder contains a hidden `.run_state/`
directory with `NAME.txt.state.json`, `NAME.txt.calls.jsonl` and a lock. The state
contains the actual scenario prompt template, input/prompt hashes, generation
settings and each completed response. Saves use a temporary file, fsync and atomic
replacement. A process lock prevents two writers from using the same output.

Each call record includes UTC time, chunk bounds, run number, elapsed time,
`finish_reason`, `empty_response`, refusal metadata, `input_tokens`, `output_tokens`,
total/reasoning tokens when supplied, truncation, raw JSON validity, response ID,
and error type/status/message on failure. Token fields are null when not supplied.
Logs describe logical calls; the SDK's internal HTTP retries are not separate rows.
Cloud routes returning plain strings have no token/finish metadata.

HTTP/transport exceptions leave missing checkpoint entries and make the command
exit nonzero. Other chunks/runs continue unless `--fail_fast` is set. A keyboard
interrupt also leaves completed entries intact. An incomplete text preview stays
inside `.run_state/`; the normal scorer sees only a fully completed scenario.

For baseline parity, **completed empty, refused, truncated and invalid-JSON
responses are retained, flagged, and not resampled by `--resume`**. Empty/refused
responses receive `response_error` call status. They are not replaced with `{}`.
This deliberately preserves the original experiment's response-quality outcomes;
the scorer itself can interpret an empty answer as no detection. A completed job
can exit zero with these quality flags, so inspect the validation report as well.
Refusals are recognized from API metadata/content-filter finish reasons; prose-only
refusals cannot be identified reliably by a generic adapter.

Repeat the same command with `--resume`. Default output discovery finds the same
checkpoint across dates. If several checkpoints match, supply `--output_file`.
Changing the input, prompt, model, token budget, endpoint, temperature, extra JSON,
line range or timezone requires a separate output. Timeout/retry settings may be
adjusted on resume. Old baseline text files have no run identities and cannot be
resumed automatically. A crash after the server responds but before local persistence
can repeat that uncommitted call; this is not an exactly-once server protocol.

## GPT-OSS validation configuration

`configs/gptoss_local_validation.json` matches the requested settings: model
`openai/gpt-oss-120b`, localhost, 400-line nominal chunks, temperature 0, max output
16,384, explicit `reasoning_effort=medium`, two runs, `America/Chicago`, and:

| Task | Scenario | Chunks |
| --- | --- | ---: |
| classification | attack-linux-scenario2 | 1 |
| classification | benign-scenario3 | 6 |
| exfiltration | exfiltration-scenario1 | 1 |
| exfiltration | benign-scenario3 | 6 |
| lm | lm-scenario2 | 17 |
| lm | benign-scenario3 | 6 |
| persistence | persistence-scenario1 | 1 |
| persistence | benign-scenario3 | 6 |
| Total | 8 scenarios | 44 |

The runner verifies 44 chunks before sending requests: 88 total calls.

After the real server is running **on the same allocated GPU node**, from the repo root:

```bash
python scripts/run_local_subset.py --config configs/gptoss_local_validation.json
# After interruption/request failures:
python scripts/run_local_subset.py --config configs/gptoss_local_validation.json --resume

# Canonicalize before scoring, preserving raw responses:
python scripts/canonicalize_outputs.py --llm local --model openai-gpt-oss-120b
python scripts/validate_local_baseline.py --candidate_llm local --output gptoss-local-comparison.json
```

The comparison script also invokes that same canonicalizer for exactly the selected
eight files, then the existing scorer. Use `--candidate_date YYYY-MM-DD` when multiple
experiments exist. It refuses incomplete/mismatched validation settings. Its exit
status is nonzero if any configured comparison fails.

The stored baseline, dated 2026-10-09, has:

| Measurement | Observed |
| --- | ---: |
| Completed responses | 88 |
| Truncated | 2/88 = 2.27% |
| Nonempty invalid JSON before repair | 9/88 = 10.23% |
| Empty | 2/88 = 2.27% |
| Invalid JSON including empty | 11/88 = 12.50% |
| Unrepaired nonempty replies | 0 |

Historical truncation comes from the console log; per-call token usage and explicit
historical reasoning settings were not archived. The old Together code did not
send `reasoning_effort`, so medium is an explicit requested setting, not a proven
historical override.

Canonicalized baseline results from the unchanged scorer, averaged across runs:

| Task | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: |
| classification | 1 | 0 | 0 | 74 |
| exfiltration | 1 | 0 | 0 | 2133 |
| lm | 0 | 2 | 1 | 8687 |
| persistence | 0.5 | 2.5 | 0.5 | 2257.5 |

The report compares TP and FP within 1 per task. Provisional numerical interpretations
of "close" are 2 percentage points around 2% truncation and 3 percentage points
around 10% nonempty invalid JSON. These are explicit, editable config values rather
than a statistical equivalence claim; confirm the intended tolerance with Luca when
interpreting real results. Empty JSON objects are valid; empty strings are reported
separately. No pass/fail claim about real vLLM predictions has been made.

## Preserved issues and limitations

- The original inclusive chunk end means most nominal 400-line chunks contain 401
  lines and overlap the next chunk by one line. Its loop can also omit a final
  singleton line when the selected length is `400*k + 1`. Both behaviors are
  deliberately preserved for parity and covered by tests.
- The old prompt archive saves only the first template per prompt key, although OS
  instructions can differ by scenario. Resume state now stores each actual scenario
  template; the replay uses original-harness prompt hashes, not that archive as a
  universal template.
- The existing token audit estimates using non-overlapping joined logs and a fixed
  prompt allowance. It flags five chunks across the full dataset (four copies of an
  OpTC scenario and one Labgen classification scenario outside the fixed subset).
  A separate audit of all 44 actual serialized subset prompts using `o200k_harmony`
  found max 96,840 prompt-text tokens, or 113,224 with the 16,384 output allowance.
  That excludes the server's final chat-template overhead; verify actual usage on HPC.
- Legacy OpTC raw inference still lacks timestamp handling; use edge logs. Prompt
  population also does not implement graph input. These paths were not expanded.
- The legacy scorer extracts filename fields by underscore positions. Use a served
  model alias without underscores if a future model ID includes them. GPT-OSS and
  Qwen configs here work with the existing naming convention.
- Python 3.10 is the repo's historical environment; cloud packages warn about its
  support lifetime. Prefer a compatible Python 3.12 environment when preparing HPC.

## What needs SJSU HPC/model access

1. Obtain the account and confirm scheduler/allocation commands, account/partition,
   GPU model/count/VRAM, CUDA driver, allowed containers/modules, storage quotas and
   the approved model cache location. No site-specific values have been invented.
2. Copy the repo and data, not the Mac `.venv`. Create a Linux client environment
   and install `requirements.txt`. Set `TZ=America/Chicago`.
3. Prepare a **separate vLLM server environment/container** compatible with those
   GPUs and model architectures. Installing vLLM into the pinned client environment
   may conflict with PyTorch/Transformers pins, so vLLM is not a client requirement.
   Record the exact vLLM/PyTorch/CUDA versions, weight/tokenizer revisions,
   quantization and launch command. Download real model weights there.
4. On an allocated GPU node, activate the server environment and inspect/launch:

   ```bash
   python scripts/serve_local_vllm.py --print_command
   python scripts/serve_local_vllm.py --tensor_parallel_size 1
   ```

   The prepared GPT-OSS command binds `127.0.0.1:8000`, keeps a 131,072-token context,
   uses one concurrent sequence, and disables model-repository generation defaults
   with `--generation-config vllm`. TP=1 is a starting configuration, not a promise
   it fits the allocated GPU with this context. Set TP to the allocation and resolve
   memory/kernel compatibility there. Do not silently reduce benchmark context/chunks.
5. In a second shell on **that same node**, activate the client environment, verify
   `/v1/models`, run one small smoke request, confirm reasoning/final-content separation
   and token usage, then run the exact GPT-OSS subset and validation commands above.
   Confirm context headroom from actual input token counts; inspect all error/quality
   flags before accepting the experiment.
6. After GPT-OSS validates, use `configs/qwen38_27b_local.json` and the same adapter:

   ```bash
   python scripts/serve_local_vllm.py --config configs/qwen38_27b_local.json --print_command
   python scripts/serve_local_vllm.py --config configs/qwen38_27b_local.json
   python scripts/run_local_subset.py --config configs/qwen38_27b_local.json
   python scripts/canonicalize_outputs.py --llm local --model Qwen-Qwen3.8-27B
   ```

   Qwen's config is a prepared benchmark-comparison starting point: thinking enabled,
   temperature 0, 16,384 output tokens, 262,144 context, and `qwen3` reasoning parsing.
   Confirm parser/model support in the installed vLLM build and available VRAM. These
   parity sampling settings intentionally differ from Qwen's recommended generation
   settings; do not infer Qwen's best attainable performance from that comparison.
   The public model ID was verified; tokenizer/weight/runtime compatibility still
   requires the GPU environment. No second model-specific adapter is needed.

References: [vLLM server and extra request parameters](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/),
[GPT-OSS serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html),
[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Files changed

| File | Change |
| --- | --- |
| `.gitignore` | Ignore `.venv` and generated run state |
| `requirements.txt` | Preserve earlier Black downgrade; add missing tiktoken |
| `inference/inference_helpers.py` | Lazy provider imports and generic local route |
| `inference/task-query-v2.py` | Local options, preflight validation, per-run persistence/resume |
| `inference/generate_commands_v2.py` | Local options and safely quoted commands; remove name-based routing restrictions |
| `inference/llm_base/interact_local.py` | Generic configurable chat-completions adapter and response metadata |
| `inference/run_state.py` | Atomic checkpoints, call logs, output locks, transcript publication |
| `configs/gptoss_local_validation.json` | Exact fixed GPT-OSS validation settings and explicit comparison tolerances |
| `configs/qwen38_27b_local.json` | Next model's client/server starting configuration |
| `scripts/run_local_subset.py` | Generic eight-scenario runner and dry runs |
| `scripts/serve_local_vllm.py` | Generic localhost server command generation/launch |
| `scripts/validate_local_baseline.py` | Canonicalize and compare fixed subset with unchanged scorer |
| `scripts/canonicalize_outputs.py` | Close input file properly; parsing/repair logic unchanged |
| `tests/test_local_inference.py` | Offline adapter, resume, CLI and full baseline replay tests |
| `tests/fixtures/gptoss_prompt_hashes.json` | Original-harness prompt/input hashes for the fixed subset |
| `README.md`, `inference/README.md`, this guide | Usage and handoff documentation |

Original baseline outputs/log/script and all source datasets are intact. Generated
canonical copies are ignored by Git. Changes are local and uncommitted.
