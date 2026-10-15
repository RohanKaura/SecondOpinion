# AuditBench

AuditBench is a benchmark dataset for evaluating the capabilities of LLMs at investigating security-related system audit logs. This repository accompanies the paper **"Benchmarking and Exploring the Capabilities of LLMs for Attack Investigations"** ([arXiv](https://arxiv.org/pdf/2606.10281)) and provides the benchmark data, the code to run LLM inference on it, and the code to score the LLM outputs against the ground truth.

The paper evaluates five frontier LLMs (GPT-5, Gemini 2.5 Pro, GPT-5 mini, Gemini 2.5 Flash, and Llama 4 Maverick) on four investigation tasks (attack classification, lateral movement, persistence, and exfiltration) over two datasets: 25 lab-generated scenarios (Lab data, Linux and Windows, available in a raw and an edge representation) and 26 scenarios curated from DARPA OpTC (edge representation only). Results are reported as true positive rate (TPR), false positive rate (FPR) and F1 score per task and representation.

**Note:** The paper's Llama 4 Maverick results were produced with `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` served through Together AI. That model is now deprecated on our Together AI account, so the code currently uses Kimi-k2.5 (`moonshotai/kimi-k2.5`) as a substitute for it. Kimi is not part of the paper; the Llama model still runs unchanged with `--llm "llama"` if your Together AI account has access to it.

## Project Structure
```
auditbench/
├─ data/                                   # the benchmark data on its own (see "Data Layout" below)
│  ├─ ground_truth/{lab,optc}/             #   groundtruth-{lm,persistence,exfiltration}.jsonl
│  └─ input/{lab,optc}/task_<task>/{edge,raw}/   #   scenario log files
├─ dataset_extraction/                     # pipeline that turns the collected SPADE/auditd logs into the scenario files (has its own README)
├─ inference/
│  ├─ README.md                            # how to run inference and compute metrics
│  ├─ GEMINI_SETUP.md                      # Vertex AI setup needed for the Gemini models
│  ├─ task-query-v2.py                     # inference script, current (v2) prompt, all models and tasks
│  ├─ task-query-v1.py                     # inference script, legacy (v1) prompt
│  ├─ generate_commands_v2.py              # prints one task-query-v2.py command per scenario of a dataset/task/representation
│  ├─ generate_commands_v1.py              # same for task-query-v1.py
│  ├─ benchmark_scenarios.py               # registry of the scenario files per dataset/task/representation/type, and their OS
│  ├─ prompt_v2.py                         # assembles the v2 prompt from a task template
│  ├─ prompt_template/                     # per-task v2 templates (goal, output format) and the input-format / OS enums
│  ├─ inference_helpers.py                 # sends a prompt to the selected LLM back end
│  ├─ llm_base/                            # API clients: interact_gpt.py (OpenAI), interact_gemini.py (Vertex AI), interact_llama.py (Together AI: Llama/Kimi)
│  ├─ helpers/, labgen_helpers/, optc_helpers/   # log loading, timestamps, output-file naming
│  ├─ labgen_expt_dataset/                 # lab data + ground truth + LLM outputs, laid out for the code (see "Data Layout")
│  ├─ optc_expt_dataset/                   # OpTC data + ground truth + LLM outputs, laid out for the code
│  ├─ metrics_comp/                        # task_computemetrics.py: scores LLM outputs against the ground truth
│  ├─ parser/                              # extracts the JSON answers from LLM output files (one parser per task and prompt version)
│  ├─ error_analysis/                      # TP/FP/FN bookkeeping used by the metrics (one module per prompt version)
│  └─ paper_metrics/                       # small aggregation helpers
├─ requirements.txt
├─ .env                                    # your API keys (never committed, see Installation)
├─ README.md
└─ .gitignore
```

## Data Layout

The benchmark data is stored twice on purpose, and the two copies are identical:

* `data/` contains only the data: the scenario log files under `data/input/{lab,optc}/task_<task>/{edge,raw}/` and the ground truth under `data/ground_truth/{lab,optc}/`. It exists so that someone who is only interested in the dataset can take it without looking into the codebase.
* `inference/labgen_expt_dataset/` and `inference/optc_expt_dataset/` contain the same data arranged in the directory structure the code expects, so that inference and metrics computation run without any path configuration. The LLM outputs are written here as well.

Inside the code layout, every task directory has the same shape (`labgen` = lab-generated data, `optc` = DARPA OpTC data; `<task>` is one of `classification`, `lm` (lateral movement), `persistence`, `exfiltration`):

```
inference/<labgen|optc>_expt_dataset/task_<task>/
├─ groundtruth-<task>.jsonl                 # ground truth (lm, persistence and exfiltration only, see below)
├─ benchmark_<llm>_prompts.jsonl            # the exact prompts that were sent, keyed by --prompt_key
└─ <edge|raw>/
   ├─ input/                                # scenario log files, e.g. edge_2024-11-27T14-04-to-2024-11-27T14-10_lm-scenario2.log
   └─ output/<llm>out/                      # LLM output files
```

* **Scenario files** are named `<rep>_<start time>-to-<end time>_<scenario name>.log`. The scenario name (e.g. `lm-scenario2`, `benign-scenario3`, `attack-linux-scenario2`) says whether the scenario contains an attack and, for OpTC, which host (`h201`, `h501`, ...) it was taken from. The lab data comes in two representations: `edge` (information-flow graph edges produced by SPADE, one edge per line) and `raw` (auditd lines). OpTC data is available in the `edge` representation only.
* **Ground truth** lives in `groundtruth-<task>.jsonl` next to the task's scenario files (and in `data/ground_truth/`). Each line is one JSON object keyed by the scenario name, listing the timestamp(s) of the attack and the field the LLM must get right: the external host for lateral movement, the MITRE ATT&CK technique name for persistence, and the exfiltrated file path for exfiltration. Classification has no ground-truth file: a scenario is an attack or benign according to its file name.
* **LLM output files** are named `<prompt_type>_<prompt_version>_<model>_<scenario file name>_<date>_temp-<temp>_numruns-<num_runs>.txt`, where `<prompt_type>_<prompt_version>` are the first two `_`-separated parts of the `--prompt_key` given at inference time. The metrics script locates and validates files by this name, so keep `_v2_` (or `_v1_` for the legacy prompt) in your prompt keys.

## Installation

1. Create a new conda environment. We recommend using Python 3.10.14 for the conda environment.

```
conda create --name myenv python=3.10.14
conda activate myenv
```

2. Set your API keys as environment variables: Create a `.env` file in the root directory with the following contents:

```
OPENAI_API_KEY=your_openaiapi_key_here
TOGETHER_API_KEY=yourtogetheraiapi_key_here
```

Note: Paste your API keys directly after the = sign. Do not add any quotes or spaces around the keys.

Our code supports 2 kinds of API keys in the `.env` file: an OpenAI API key and a Together AI API key (for running the Llama 4 Maverick or Kimi models). For Gemini models, authentication is handled with the Google Cloud Vertex AI platform instead of with an API key. See [`./inference/GEMINI_SETUP.md`](./inference/GEMINI_SETUP.md) for setup instructions needed to run inference with Gemini models.

3. Install python dependencies using:

```
pip install -r requirements.txt
```

## Repository Features

This repo provides 3 main functionalities:
1. Run inference on our benchmark data with any of the paper's five LLMs, or Kimi-k2.5 as the Llama substitute (`inference/task-query-v2.py`), for any of the four tasks on either dataset.
2. Compute correctness metrics by comparing the LLM outputs with the ground truth (`inference/metrics_comp/task_computemetrics.py`): true/false positive and negative counts per scenario, and the true positive rate (TPR) and false positive rate (FPR) aggregated over all scenarios of a task.
3. Replicate the results associated with our paper.

## Prompt Versions

This repository has exactly **two** prompt versions, which are the Prompt v1 and Prompt v2 of the paper (§ 4.2.2):

* **v2** (current, default): the templated prompt assembled by `inference/prompt_v2.py` from the per-task templates in `inference/prompt_template/*_template_v2.py`. Use it for all new inference runs (`task-query-v2.py`) and metric computations (`--version "v2"`).
* **v1** (legacy): the original hand-written prompts stored per task in `benchmark_<llm>_prompts.jsonl`, run with `task-query-v1.py` and evaluated with `--version "v1"`. It is kept only to evaluate older outputs.

Older outputs and commands may still carry the labels "v3" (meaning v2) and "v0" (meaning v1); both are accepted as aliases.

### Running the Benchmark

For instructions on running inference and computing correctness metrics, refer to [`./inference/README.md`](./inference/README.md). For how the scenario log files were produced from the collected logs, see [`./dataset_extraction/README.md`](./dataset_extraction/README.md).

For the generic `--llm local` vLLM backend, failure logs, resume support, GPT-OSS
validation and Qwen3.8-27B preparation, see [the local inference/HPC guide](docs/LOCAL_VLLM.md).
