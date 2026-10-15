# HPC validation checklist

Status: local tests pass; **real GPU inference is still untested**. Use branch
`local-vllm-harness`. Obtain an SJSU GPU allocation and compatible model access first;
no scheduler, partition or GPU type is assumed here. Create Linux client and vLLM
server environments separately; do not copy the Mac `.venv`. Commands below start
at the repository root. Keep the server and harness on the **same allocated GPU
node**, using `http://127.0.0.1:8000/v1`.

1. **Launch GPT-OSS in the server environment.** Record GPU/VRAM, CUDA, vLLM version,
   weight/tokenizer revision, quantization and launch command. Inspect, then launch:

   ```bash
   python scripts/serve_local_vllm.py --print_command
   python scripts/serve_local_vllm.py --tensor_parallel_size 1
   ```

   Set the GPU count to the actual allocation; one GPU is only a starting setting.
   The config requests a 131,072-token context and one concurrent sequence. In a
   second shell on that same node, activate the client environment and check:

   ```bash
   curl --fail http://127.0.0.1:8000/v1/models
   ```

   Confirm `openai/gpt-oss-120b` is advertised. Run a small smoke request and inspect
   final-content/reasoning separation and token usage before the full experiment.
   Resolve GPU/runtime issues there; do not silently shrink chunks or context.

2. **Verify the exact baseline settings.** Use
   `configs/gptoss_local_validation.json`: `openai/gpt-oss-120b`, `--llm local`,
   nominal 400-line chunks, temperature **0**, max output **16,384** tokens,
   reasoning effort **medium**, **2 runs**, and `TZ=America/Chicago`.

   ```bash
   export TZ=America/Chicago
   python -m pip check
   python scripts/run_local_subset.py --dry_run
   ```

3. **Run the fixed subset: 8 scenarios / 44 chunks / 88 responses.**

   ```bash
   python scripts/run_local_subset.py --config configs/gptoss_local_validation.json
   # If interrupted or requests failed, rerun with --resume:
   python scripts/run_local_subset.py --config configs/gptoss_local_validation.json --resume
   ```

   Proceed only after all eight scenarios are complete. Inspect `.run_state/` call
   logs for request failures, empty/refused replies, finish reasons, token counts
   and truncation. Completed empty/truncated/refused replies are retained for parity;
   resume retries missing requests, not completed low-quality answers.

4. **Canonicalize, preserving raw outputs.**

   ```bash
   python scripts/canonicalize_outputs.py --llm local --model openai-gpt-oss-120b
   ```

   Canonical copies go into `local-canonout/`.

5. **Score with `--llm local-canon`, not `local`.** The following selects exactly
   the eight configured files and scores their canonical copies with the existing
   parser/metrics. If multiple experiment dates exist, set `date` to the intended
   `YYYY-MM-DD`; selection otherwise fails rather than mixing experiments.

   ```bash
   python - <<'PY'
   import json, subprocess, sys
   from pathlib import Path
   from scripts.validate_local_baseline import select_files
   config = json.loads(Path("configs/gptoss_local_validation.json").read_text())
   date = None
   for task, raw in select_files(config, "local", config["model"].replace("/", "-"), date):
       subprocess.run([
           sys.executable, "metrics_comp/task_computemetrics.py",
           "--dataset", "labgen", "--rep", "edge", "--task", task,
           "--version", "v2", "--llm", "local-canon",
           "--llmoutput_fname", raw.name,
       ], cwd="inference", check=True)
   PY
   ```

6. **Compare against Luca's baseline and inspect the report.**

   ```bash
   python scripts/validate_local_baseline.py --candidate_llm local --output gptoss-local-comparison.json
   ```

   Add `--candidate_date YYYY-MM-DD` if dates are ambiguous. This command checks
   completed response counts/settings, uses the canonicalized outputs, compares
   existing metrics, and exits nonzero if a configured criterion fails.

   | Baseline measurement | Reference |
   | --- | ---: |
   | Truncation | 2/88 = 2.27% |
   | Nonempty invalid JSON before canonicalization | 9/88 = 10.23% |
   | Empty replies, reported separately | 2/88 = 2.27% |
   | Classification TP / FP | 1 / 0 |
   | Exfiltration TP / FP | 1 / 0 |
   | Lateral movement TP / FP | 0 / 2 |
   | Persistence TP / FP | 0.5 / 2.5 |

   Counts are averaged over the two runs. The configured count tolerance is **1 TP
   and 1 FP per task**. Provisional rate tolerances are **2 percentage points around
   2% truncation** and **3 points around 10% nonempty invalid JSON**; confirm these
   interpretations of "close" with Luca when accepting real results. Also inspect
   empty/refusal and request-failure logs; a zero inference exit code alone is not
   validation success. Archive raw/canonical outputs, state/call logs, report and
   server provenance together.

7. **Only after GPT-OSS passes, run Qwen3.8 27B through the same adapter.** Stop the
   GPT-OSS server before reusing port 8000. In the server environment:

   ```bash
   python scripts/serve_local_vllm.py --config configs/qwen38_27b_local.json --print_command
   python scripts/serve_local_vllm.py --config configs/qwen38_27b_local.json
   ```

   Confirm the allocated hardware/vLLM build supports its prepared context and
   reasoning parser. In the client shell on that same node:

   ```bash
   python scripts/run_local_subset.py --config configs/qwen38_27b_local.json --dry_run
   python scripts/run_local_subset.py --config configs/qwen38_27b_local.json
   python scripts/canonicalize_outputs.py --llm local --model Qwen-Qwen3.8-27B
   ```

   Score as in step 5, loading `configs/qwen38_27b_local.json` instead. Continue
   using `--llm local-canon`; no model-specific adapter is needed. Qwen's prepared
   settings are a comparison starting point, not verified GPU settings or a claim
   about its best achievable accuracy.

Known issues deliberately documented and preserved:

- **Chunk overlap:** nominal 400-line chunks can include 401 lines and overlap by
  one. The original loop can also omit a trailing singleton line. Kept for parity.
- **Old prompt archive:** only the first template per prompt key is saved there;
  OS-specific scenario templates can differ. New checkpoints store the actual template.
- **Historical reasoning effort is unknown:** the old Together code did not set it;
  the new config explicitly requests medium.
- **Oversized chunks outside the fixed subset:** the existing approximate token
  audit flags five. The fixed subset's prompt-text audit fits its budget, but final
  server chat-template overhead still needs real token-usage verification.
- **Real GPU inference remains untested:** offline HTTP replay proves integration
  parity, not vLLM model accuracy, memory fit, speed or hardware compatibility.

See [the full handoff notes](LOCAL_VLLM.md) for implementation details and limitations.
