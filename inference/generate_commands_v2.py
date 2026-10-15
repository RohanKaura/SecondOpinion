
import importlib
import sys
import argparse
import json
import shlex
from llm_base.interact_local import add_local_arguments

# Dynamically import the benchmark_scenarios module
benchmark_scenarios = importlib.import_module("benchmark_scenarios")

def main():
    parser = argparse.ArgumentParser(description="Generate python commands for LLM evaluation.")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--task', required=True, help='Task name')
    parser.add_argument('--rep', required=True, choices=['raw', 'edge'], help='Representation (raw/edge)')
    parser.add_argument('--type', required=True, help='Type (benign/attack)')
    parser.add_argument('--llm', required=True, choices=['gpt', 'gemini', 'llama', 'kimi', 'local'], help='LLM backend')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--num_lines', type=int, required=False, default=None, help='Number of lines of logs for running inferences.')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--temp', type=float, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Emit inference commands that only print their plans')
    add_local_arguments(parser)
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    rep = args.rep
    typ = args.type
    llm = args.llm
    model = args.model

    allowed_tasks = {"lm", "persistence", "exfiltration", "classification"}
    if task not in allowed_tasks:
        print(f"Error: task must be one of {allowed_tasks}. Got '{task}' instead.")
        sys.exit(1)

    # Provider selection is explicit. Model names need not contain the provider name
    # (Together's GPT-OSS baseline is one example).

    # Construct the variable name
    var_name = f"{dataset}_{task}_{rep}_{typ}"

    # Try to get the list from benchmark_scenarios
    try:
        scenario_list = getattr(benchmark_scenarios, var_name)
    except AttributeError:
        print(f"No such scenario list: {var_name}")
        sys.exit(1)

    # Set num_lines based on rep
    if args.num_lines is None:
        num_lines = 1000 if rep == "raw" else 400
    else:
        num_lines = args.num_lines

    # Prompt version label. Every task uses the single v2 prompt template (prompt_v2.py). The label
    # is embedded in --prompt_key and therefore in the output file name, which the v2 parsers and
    # metrics scripts rely on to recognise v2 outputs.
    version = "v2"

    prompt_key = f"{task}-investigation_{version}_{task}_{dataset}_{rep}"

    # Set temp based on model
    if model in {"gpt-5", "gpt-5-mini"}:
        temp = 1
    elif model in {"gemini-2.5-flash", "gemini-2.5-pro", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}:
        temp = 0
    else:
        temp = 0  # Default to 0 if not specified

    if args.temp is not None:
        temp = args.temp
    if num_lines < 1 or args.num_runs < 1:
        parser.error('num_lines and num_runs must be positive')

    for log_fname in scenario_list:
        # All models and tasks share the single v2 inference script; --llm/--model select the model.
        command = ['python', 'task-query-v2.py', '--dataset', dataset,
                   '--log_fpath', f'./{dataset}_expt_dataset/task_{task}/{rep}/input/{log_fname}',
                   '--num_lines', str(num_lines), '--prompt_key', prompt_key,
                   '--num_runs', str(args.num_runs), '--temp', str(temp),
                   '--llm', llm, '--model', model, '--task', task]
        if llm == 'local':
            if args.base_url:
                command += ['--base_url', args.base_url]
            command += ['--max_tokens', str(args.max_tokens), '--timeout', str(args.timeout),
                        '--max_retries', str(args.max_retries), '--extra_body', json.dumps(args.extra_body)]
        if args.resume:
            command.append('--resume')
        if args.dry_run:
            command.append('--dry_run')
        print(shlex.join(command))
        print()


if __name__ == "__main__":
    main()
