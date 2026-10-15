## Running LLM Inference

V2 also supports `--llm local` for any chat model served by a localhost vLLM server.
See [the local inference/HPC guide](../docs/LOCAL_VLLM.md) for configurable request
settings, per-call logs, resume support, and the fixed GPT-OSS validation workflow.
V2 now checkpoints each response and publishes its transcript when a scenario is
complete; rerun interrupted jobs with `--resume`. Existing text-only outputs cannot
be resumed safely. Prompt templates, chunk boundaries, transcript format and scoring
logic remain compatible with the original baseline.

To evaluate the performance of a given LLM on the benchmark data, you will run an inference script corresponding to your chosen LLM. Make sure to navigate to the `./inference/` directory (relative to the repository root) before running these commands; all paths in the commands below are relative to it.

### General Command Structure
Regardless of which LLM model you are using for inference, you will run a command of the following structure:

```
python <script_name.py> --dataset <dataset> --log_fpath <log_path> --task <task> [additional flags...]
```

#### Required Flags for Inference Scripts

You must set the following flags for each inference script that you run:

* `--dataset`: The benchmark dataset to run LLM inference on. Valid values: "labgen" or "optc".
* `--task`: The type of investigation task for the chosen LLM to conduct on the data. Valid values: "classification", "lm", "persistence", or "exfiltration".
* `--llm`: The LLM back end. Valid values: "gpt" (OpenAI), "gemini" (Vertex AI), "llama" or "kimi" (both served through Together AI), and "local" (v2, localhost vLLM). This value also names the directory the outputs are written to (`<llm>out`).
* `--model`: The specific model of the LLM being used. The paper's models are "gpt-5-mini", "gpt-5", "gemini-2.5-flash", "gemini-2.5-pro", and "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" (with `--llm "llama"`); "moonshotai/kimi-k2.5" (with `--llm "kimi"`) is the current substitute for the deprecated Llama model.
* `--log_fpath`: The path to the input log file you want to run LLM inference on. (These are located in the dataset folders `./labgen_expt_dataset` and `./optc_expt_dataset`. For example, `./labgen_expt_dataset/task_lm/edge/input/` contains lab generated log files that contain lateral movement scenarios represented with a node/edge format).
* `--num_lines`: The number of log lines sent to the LLM in one query; the whole file is processed in consecutive chunks of this size (use `--start_lno`/`--end_lno` to restrict the range). The paper used chunks of 400 lines for the "edge" representation and 1,000 lines for the "raw" representation, which come to roughly 100,000-128,000 tokens.
* `--prompt_key`: A custom text label used to name your generated output file and track your experiment. Its first two `_`-separated parts become the prefix of the output file name (`<prompt_type>_<prompt_version>_...`), and the metrics script recognises v2 outputs by the `_v2_` part, so use the format `{task}_v2_{dataset}_{rep}` (e.g. "lm_v2_labgen_edge").
* `--num_runs`: The number of times to run the inference. We recommend starting with 1. Larger values can help in evaluating the consistency of the AI's responses.
* `--temp`: The temperature setting for models that support it. The paper used 0 wherever the model allows it, for deterministic outputs (the GPT-5 models only accept their default of 1).
* `--custom_prompt` - (Optional) A custom string or a path to a .txt file containing a user-defined prompt for the LLM. If provided, this overrides the default task goal in the prompt template. See `./prompt_template/classification_template_v2.py`, `./prompt_template/exfiltration_template_v2.py`, `./prompt_template/persistence_template_v2.py`, and `./prompt_template/lateralmovement_template_v2.py` for the current default values of the `GOAL` variable for each of the four tasks, which can be overridden with this flag.

#### Where the Outputs Go

V2 checkpoints each response and, when a scenario is complete, publishes all its answers in a single file,
`./<dataset>_expt_dataset/task_<task>/<rep>/output/<llm>out/<prompt_type>_<prompt_version>_<model>_<scenario file name>_<date>_temp-<temp>_numruns-<num_runs>.txt`,
and saves the prompt template under `--prompt_key` in `./<dataset>_expt_dataset/task_<task>/benchmark_<llm>_prompts.jsonl`. That legacy archive retains the first template per key; the v2 checkpoint also records the actual template for each scenario. Incomplete transcripts and per-call logs remain in the output folder's `.run_state/` directory.

### Example Commands by Model

#### Note: The examples below use our "v2" prompt template, which is used as the default in the inference scripts.

All models and tasks share a single v2 inference script, `task-query-v2.py`. Select the model with the `--llm` and `--model` flags. (If you want to run the legacy "v1" prompt instead of the default v2 template, the equivalent single script is `task-query-v1.py`, which takes the same flags.)

**Prompt versions:** v2 is the current prompt (`prompt_v2.py`) and v1 the legacy one (`benchmark_*_prompts.jsonl`); older outputs and commands may still carry the labels "v3" (meaning v2) and "v0" (meaning v1), which are accepted as aliases.

### 1. Gemini Models
#### To run Gemini models, please setup a Vertex AI project by following the steps in [`./GEMINI_SETUP.md`](./GEMINI_SETUP.md).
Use the `task-query-v2.py` script.

#### Classification Example:

```
python task-query-v2.py --dataset "labgen" --log_fpath "./labgen_expt_dataset/task_classification/edge/input/edge_2025-01-13T01-58-to-2025-01-13T02-03_attack-windows-lmscenario1.log" --num_lines 400 --prompt_key "classification_v2_labgen_raw" --num_runs 1 --temp 0 --llm "gemini" --model "gemini-2.5-flash" --task "classification"
```

#### Lateral Movement (LM) Example:

```
python task-query-v2.py --dataset "labgen" --log_fpath "./labgen_expt_dataset/task_lm/edge/input/edge_2024-11-27T14-04-to-2024-11-27T14-10_lm-scenario2.log" --num_lines 400 --prompt_key "lm_v2_labgen_edge" --num_runs 1 --temp 0 --llm "gemini" --model "gemini-2.5-flash" --task "lm"
```

### 2. GPT Models
Use the `task-query-v2.py` script.

#### Lateral Movement (LM) Example:

```
python task-query-v2.py --dataset "optc" --log_fpath "./optc_expt_dataset/task_lm/edge/input/edge_2019-09-18T11-49-38-to-2019-09-18T11-49-58_h201_velox-benign-scenario7.log" --num_lines 400 --prompt_key "lm_v2_optc_edge" --num_runs 1 --temp 0 --llm "gpt" --model "gpt-5-mini" --task "lm"
```

### 3. Llama 4 Maverick and Kimi Models (via Together AI)
The paper's fifth model, Llama 4 Maverick, is served through Together AI. Use the `task-query-v2.py` script with `--llm "llama"`:

#### Classification Example (Llama 4 Maverick, as in the paper):

```
python task-query-v2.py --dataset "optc" --log_fpath "./optc_expt_dataset/task_classification/edge/input/edge_2019-09-24T11-09-18-to-2019-09-24T11-09-39_h501_attack-scenario3.log" --num_lines 400 --prompt_key "classification_v2_optc_edge" --num_runs 1 --temp 0 --llm "llama" --model "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" --task "classification"
```

**Note:** This Llama model is deprecated on our Together AI account, so we currently use Kimi-k2.5 as a substitute (Kimi is not part of the paper). Pass the Kimi `--llm` and `--model` arguments as shown below.

#### Classification Example (Kimi-k2.5 substitute):

```
python task-query-v2.py --dataset "optc" --log_fpath "./optc_expt_dataset/task_classification/edge/input/edge_2019-09-24T11-09-18-to-2019-09-24T11-09-39_h501_attack-scenario3.log" --num_lines 400 --prompt_key "classification_v2_optc_edge" --num_runs 1 --temp 0 --llm "kimi" --model "moonshotai/kimi-k2.5" --task "classification"
```

### Running All Scenarios of a Task

`generate_commands_v2.py` prints one `task-query-v2.py` command for every scenario file registered in `benchmark_scenarios.py` for a dataset, task, representation and scenario type ("attack" or "benign"), using the prompt key `<task>-investigation_v2_<task>_<dataset>_<rep>`:

```
python generate_commands_v2.py --dataset "labgen" --task "lm" --rep "edge" --type "attack" --llm "gemini" --model "gemini-2.5-flash"
```

Run the printed commands (for example by piping them into `sh`). `generate_commands_v1.py` does the same for the legacy v1 prompt.

### Computing Correctness Metrics

Once inference is complete, the LLM predictions are saved as .txt files in the output directories corresponding to the chosen dataset, task, and data representation (e.g. `./labgen_expt_dataset/task_lm/edge/output/`). You can then score them against the dataset's ground truth with `./metrics_comp/task_computemetrics.py`.

#### What the Metrics Script Computes

The script extracts the JSON answers from an LLM output file and compares them with the ground truth. It reports the counts of true positives (TP), false positives (FP), false negatives (FN) and true negatives (TN) and, when run over all scenarios of a task, the resulting **true positive rate, TPR = TP / (TP + FN)**, and **false positive rate, FPR = FP / (FP + TN)**. It does not compute precision or recall.

**F1 score.** The F1 scores reported in the paper are not computed by the script. They are derived from the counts the script prints as

```
F1 = 2·TP / (2·TP + FP + FN)
```

**Lateral movement, persistence, exfiltration.** Every JSON object the LLM reports with verdict `HIGH_SUSPICIOUS` counts as a positive (`--more_defensive 1` also counts `MEDIUM_SUSPICIOUS`; with the legacy v1 prompts, which have no verdict field, every reported object is a positive). A positive is a TP if it matches the ground truth and an FP otherwise: for lateral movement the (timestamp, external host) pair must match exactly; for persistence and exfiltration the timestamp must match and the ground-truth technique name / file path must appear in the reported field. A run of an attack scenario in which no TP was found counts as one FN (for OpTC lateral movement, the FN count is the number of ground-truth external hosts that were not reported). TN is approximated as the number of log lines in the scenario minus the FPs, so the FPR denominator is the total number of log lines, as in the paper (§ 4.3). In a benign scenario every positive is an FP. The counts are averaged over the `--num_runs` runs stored in a file.

**Classification.** The classification prompt asks for exactly one JSON object per chunk; the chunk is labelled malicious if that object's verdict is positive (as above) and benign otherwise, including when the object is empty (with the legacy v1 prompts, the LLM's explicit malicious/benign decision is used). An attack scenario is detected (TP) if at least one of its chunks is labelled malicious and missed (FN) otherwise; in a benign scenario every chunk labelled malicious is an FP and every other chunk a TN, so the FPR denominator is the number of log chunks, as in the paper (§ 4.3).

**Output.** For a single file (`--llmoutput_fname`) the per-scenario counts are printed. With `--all_scenarios 1` the counts are summed over all scenarios of the given `--scenario_type` and the rates are printed as percentages:

* lateral movement / persistence / exfiltration: `Aggregate LLM TPs: TP/(TP+FN) = TPR%` (attack scenarios only) and `Aggregate percentage LLM FPs: FP/(FP+TN) = FPR%`;
* classification: `Aggregate LLM accuracy for ground truth attack scenarios: detected/total = TPR%` (attack) and `Aggregate FPs malicious labels: malicious chunks/total chunks = FPR%` (benign).

With `--quiet 1` only the numbers are printed, as `TP, TP+FN, FP, FP+TN` for lateral movement / persistence / exfiltration and as `detected, total` (attack) or `malicious chunks, total chunks` (benign) for classification, which is convenient for scripting.

#### Ground Truth

The ground truth for a task is the file `./<dataset>_expt_dataset/task_<task>/groundtruth-<task>.jsonl` (an identical copy is in `../data/ground_truth/`). Each line is one JSON object keyed by the scenario name that ends the scenario file name (e.g. `lm-scenario2`), listing the attack's timestamp(s) and the field the LLM must report: the external host (lateral movement), the MITRE ATT&CK technique name (persistence) or the exfiltrated file path (exfiltration). Classification has no ground-truth file: a scenario is an attack or benign according to its file name.

#### Flags for the Metrics Script

* `--dataset`: The dataset which the LLM ran inference on. Valid values: "labgen" or "optc".
* `--task`: The investigation task which the LLM conducted. Valid values: "classification", "lm", "persistence", or "exfiltration".
* `--rep`: The log representation format used during inference. Valid values: "edge" or "raw".
* `--llm`: The LLM back end that generated the outputs, i.e. the `<llm>out` output directory to read. Valid values: "gpt", "gemini", "llama", "kimi".
* `--version`: The prompt version used during inference. Valid values: "v2" or "v1". This tells the script which parsing logic to use to extract the AI's answers. Note: Use "v2" for all new inference runs. The "v1" option is still supported in order to evaluate older outputs from earlier stages of the project. ("v3" and "v0" are accepted as aliases of "v2" and "v1".)
* `--llmoutput_fname`: (Optional) The exact .txt filename of the AI's output that you want to evaluate. (These files can be found in `./labgen_expt_dataset/task_lm/edge/output/` for inference on lab generated log files that contain lateral movement scenarios represented with a node/edge format)
* `--all_scenarios`: (Optional) Set to 1 if you want to evaluate an entire directory of output files at once instead of providing a single file name.
* `--scenario_type`: (Required if using --all_scenarios) Specifies whether to evaluate "benign" or "attack" scenarios.
* `--model`: (Required if using --all_scenarios) The specific model name used during inference (e.g., "gemini-2.5-flash").
* `--more_defensive`: (Optional) Set to 1 to also count `MEDIUM_SUSPICIOUS` verdicts as positives (v2 prompts only). Default 0: only `HIGH_SUSPICIOUS` counts.
* `--quiet`: (Optional) Set to 1 to print only the aggregate numbers described above.
* `--metric`: (Optional) Defaults to "correctness", which is the only metric implemented; the flag is kept for backward compatibility with older commands.

#### Evaluating a Single Inference File
To evaluate a single inference file, you must provide the exact output filename generated during inference using the --llmoutput_fname flag.

```
python ./metrics_comp/task_computemetrics.py \
  --dataset "labgen" \
  --rep "edge" \
  --llm "gemini" \
  --task "persistence" \
  --llmoutput_fname "persistence-investigation_v2_gemini-2.5-flash_edge_2024-11-16T02-00-to-2024-11-16T02-05_persistence-scenario1.log_2026-01-12_temp-0.0_numruns-1.txt" \
  --version "v2"
```

#### Evaluating Multiple Inference Files at Once
If you want to evaluate all inference files produced by a certain model for a given experiment, set `--all_scenarios 1` along with the appropriate `--scenario_type` ("attack"/"benign") and `--model` flags. The script picks up every file in the output directory whose name contains the model name, the scenario type and the prompt-version label (`_v2_`).

```
python ./metrics_comp/task_computemetrics.py \
  --dataset "labgen" \
  --rep "edge" \
  --llm "gemini" \
  --task "persistence" \
  --version "v2" \
  --all_scenarios 1 \
  --scenario_type "attack" \
  --model "gemini-2.5-flash"
```

### Replicating Paper Results
To fully replicate the tables and metrics presented in our paper, inference must be run across all scenario log files in the dataset (found in the input directories of both labgen and optc): use `generate_commands_v2.py` for every dataset, task, representation, scenario type and model. Then run the metrics script with `--all_scenarios 1` twice per experiment, once with `--scenario_type "attack"` (giving TP, FN and the FPs raised on attack scenarios, i.e. the TPR) and once with `--scenario_type "benign"` (giving the FPs raised on benign scenarios, i.e. the FPR). F1 is computed from the printed TP, FP and FN counts with the formula above.
