from helpers import helpers
from labgen_helpers import helpers as labgenhelpers
from optc_helpers import helpers as optchelpers
from zoneinfo import ZoneInfo
import os
import argparse
import datetime
import json
import math
from pathlib import Path
from contextlib import ExitStack
from llm_base.interact_local import LocalClient, add_local_arguments, local_options
from run_state import digest, legacy_chunks, find_resume_output, run_inference
import inference_helpers
import benchmark_scenarios

from prompt_v2 import Prompt
from prompt_template.input_format import InputFormat
from prompt_template.ostype import OSType
from prompt_template.lateralmovement_template_v2 import LateralMovementPrompt
from prompt_template.persistence_template_v2 import PersistencePrompt
from prompt_template.exfiltration_template_v2 import ExfiltrationPrompt
from prompt_template.classification_template_v2 import ClassificationPrompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run v2-prompt inference for any task on any supported LLM (GPT, Gemini, Llama/Kimi, local vLLM).")
    parser.add_argument("--dataset", type=str, required=True, default="labgen", choices=["labgen", "optc"], help="name of the dataset. 'labgen', 'optc'")
    parser.add_argument("--log_fpath", type=str, required=True, default="./dataset/sorted_provenance_08202024.jsonl",
                        help="path of log file")
    parser.add_argument("--num_lines", type=int, required=True, default=1000,
                        help='number of lines')
    parser.add_argument('--prompt_key', type=str, required=True, default='security v2',
                        help='enter the dictionary key from benchmark_prompts.jsonl file')
    parser.add_argument('--num_runs', type=int, default=10, required=True,
                        help='number of times to run the consistency experiment')
    parser.add_argument('--temp', type=float, default=1, required=True,
                        help='values between 0-2. lower values result in more consistent outputs (e.g. 0.2), while higher values generate more diverse and creative results(e.g. 1.0)')
    parser.add_argument("--start_lno", type=int, required=False, default=0,
                        help="if the input log file is too big, then specify start and end line numbers")
    parser.add_argument("--end_lno", type=int, required=False, default=0,
                        help="if the input log file is too big, then specify start and end line numbers")
    parser.add_argument("--model", type=str, required=False, default='gemini-2.5-flash',
                        help="model name to run the query, e.g. 'gpt-5-mini', 'gemini-2.5-flash', 'moonshotai/kimi-k2.5'")
    parser.add_argument("--llm", type=str, required=False, default='gemini', choices=["gpt", "gemini", "llama", "kimi", "local"],
                        help="LLM backend; local uses a localhost vLLM server")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "lm", "persistence", "exfiltration"], help="task name: 'classification', 'lm', 'persistence', 'exfiltration'")
    parser.add_argument("--custom_prompt", type=str, required=False, default=None,
                        help="Pass a custom prompt string or a path to a .txt file to override the default goal section of the prompt for the selected task.")

    add_local_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="Skip checkpointed chunk/run pairs; retry failures")
    parser.add_argument("--output_file", type=Path, help="Explicit transcript path (also selects a checkpoint across dates)")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed call")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print the plan without requests or output writes")
    args = parser.parse_args()
    if args.num_lines < 1 or args.num_runs < 1 or not math.isfinite(args.temp) or not 0 <= args.temp <= 2:
        parser.error("num_lines/num_runs must be positive; temp must be between 0 and 2")
    if len(args.prompt_key.split("_")) < 2:
        parser.error("prompt_key must contain a prompt version, e.g. classification-investigation_v2_labgen_edge")
    if not Path(args.log_fpath).is_file():
        parser.error(f"Input log does not exist: {args.log_fpath}")

    assert args.dataset in args.log_fpath, "Mismatch in --dataset and --log_fpath flag"

    assert args.task in {'lm', 'persistence', 'exfiltration', 'classification'}, "Undefined task"

    custom_goal_text = None
    if args.custom_prompt:
        if os.path.isfile(args.custom_prompt):
            with open(args.custom_prompt, 'r') as f:
                custom_goal_text = f.read().strip()
        else:
            custom_goal_text = args.custom_prompt

    if args.start_lno and args.end_lno:
        st_lno = args.start_lno
        en_lno = args.end_lno
    elif args.start_lno != 0 and args.end_lno == 0:
        st_lno = args.start_lno
        en_lno = inference_helpers.count_total_lines(args.log_fpath)
    elif args.start_lno == 0 and args.end_lno != 0:
        st_lno = 1
        en_lno = args.end_lno
    else:
        st_lno = 1
        en_lno = inference_helpers.count_total_lines(args.log_fpath)

    total_lines = inference_helpers.count_total_lines(args.log_fpath)
    if args.start_lno < 0 or args.end_lno < 0 or not 1 <= st_lno <= en_lno <= total_lines:
        parser.error("Invalid or empty input line range")
    num_lines = args.num_lines
    rep = args.log_fpath.split("/")[-1].split("_")[0]

    curr_date = datetime.datetime.now(ZoneInfo("US/Central")).date().strftime("%Y-%m-%d")
    prompt_key = args.prompt_key

    output_file = helpers.create_output_filename(prompt_key, args.log_fpath, curr_date, args.temp, args.num_runs, args.model)
    outpath = f"./{args.dataset}_expt_dataset/task_{args.task}/{rep}/output/{args.llm}out/"+output_file

    if args.output_file:
        outpath = args.output_file

    scenario = args.log_fpath.split("/")[-1]
    if args.dataset == "labgen":
        os_type = getattr(benchmark_scenarios, f"{args.dataset}_{args.task}_os")[scenario]
    else:
        os_type = "windows"

    if args.task == "lm":
        task_prompt = Prompt(
            task = "lateral movement",
            os_type = getattr(OSType, os_type),
            goal = custom_goal_text if custom_goal_text else LateralMovementPrompt.GOAL,
            output_description = LateralMovementPrompt.OUTPUT_DESCRIPTION,
            json_output_format = LateralMovementPrompt.OUTPUT_FORMAT,
            log_input_format = getattr(InputFormat, rep)
        )

    if args.task == "persistence":
        task_prompt = Prompt(
            task = "persistence",
            os_type = getattr(OSType, os_type),
            goal = custom_goal_text if custom_goal_text else PersistencePrompt.GOAL,
            output_description = PersistencePrompt.OUTPUT_DESCRIPTION,
            json_output_format = PersistencePrompt.OUTPUT_FORMAT,
            log_input_format = getattr(InputFormat, rep),
            additional_sections=PersistencePrompt.ADDITIONAL_SECTION,
            mitre_output=PersistencePrompt.OUTPUT_FORMAT_MITRE,
        )

    if args.task == "exfiltration":
        task_prompt = Prompt(
            task = "exfiltration",
            os_type = getattr(OSType, os_type),
            output_description = ExfiltrationPrompt.OUTPUT_DESCRIPTION,
            goal = custom_goal_text if custom_goal_text else ExfiltrationPrompt.GOAL,
            json_output_format = ExfiltrationPrompt.OUTPUT_FORMAT,
            log_input_format = getattr(InputFormat, rep),
        )

    if args.task == "classification":
        task_prompt = Prompt(
        task = "attack",
        os_type = getattr(OSType, os_type),
        output_description = ClassificationPrompt.OUTPUT_DESCRIPTION,
        goal = custom_goal_text if custom_goal_text else ClassificationPrompt.GOAL,
        json_output_format = ClassificationPrompt.OUTPUT_FORMAT,
        log_input_format = getattr(InputFormat, rep),
    )

    chunks = list(legacy_chunks(st_lno, en_lno, num_lines))
    with ExitStack() as stack:
        local_client = stack.enter_context(LocalClient(**local_options(args))) if args.llm == "local" else None
        config = {
            "runner_version": "v2-parity-1",
            "dataset": args.dataset, "task": args.task, "representation": rep,
            "input_name": Path(args.log_fpath).name,
            "input_sha256": digest(Path(args.log_fpath).read_bytes()),
            "prompt_sha256": digest(task_prompt.populate("_input_data_")),
            "prompt_template": task_prompt.populate("_input_data_"),
            "prompt_key": args.prompt_key, "llm": args.llm, "model": args.model,
            "temp": args.temp, "num_lines": num_lines, "num_runs": args.num_runs,
            "start_lno": st_lno, "end_lno": en_lno,
            "local": {"base_url": local_client.base_url, "max_tokens": args.max_tokens,
                      "extra_body": args.extra_body} if local_client else None,
            "timezone": os.environ.get("TZ", "system-local"),
        }
        if args.resume and not args.output_file:
            outpath = find_resume_output(outpath, config)
        if args.dry_run:
            print(json.dumps({"config": config, "chunks": len(chunks),
                              "calls": len(chunks) * args.num_runs,
                              "output": str(outpath)}, indent=2))
            raise SystemExit(0)

        fpath = f"./{args.dataset}_expt_dataset/task_{args.task}/benchmark_{args.llm}_prompts.jsonl"
        task_prompt.save_prompt(fpath, args.prompt_key, task_prompt.populate("_input_data_"))

        def make_message(start, end):
            # Preserve the original list-of-lines prompt serialization for baseline parity.
            return task_prompt.populate(helpers.load_logs(args.log_fpath, start, end))

        def time_range(start, end):
            if args.dataset == "labgen":
                get_timestamps = getattr(labgenhelpers, f"get_{rep}log_timestamps")
            elif rep == "edge":
                get_timestamps = optchelpers.get_edgelog_timestamps
            else:
                raise ValueError("OpTC raw timestamps are not implemented; use edge logs")
            first, last = get_timestamps(args.log_fpath, start, end)
            first = helpers.unixts_to_string(first) if first is not None else None
            last = helpers.unixts_to_string(last) if last is not None else None
            return f"{first}-to-{last}"

        def invoke(message):
            if local_client:
                return local_client.generate(args.model, message, args.temp)
            return inference_helpers.call_model(args.llm, args.model, message, args.temp)

        try:
            status = run_inference(outpath, config, chunks, make_message, time_range, invoke,
                                   resume=args.resume, fail_fast=args.fail_fast)
        except ValueError as error:
            parser.exit(2, f"Error: {error}\n")
        raise SystemExit(status)
