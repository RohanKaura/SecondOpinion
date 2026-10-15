from helpers import helpers
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os


def call_model(llm, model, message, temp, **local_options):
    """Send a single prompt to the requested LLM backend and return its text response.

    Routing is done on the explicit ``--llm`` value rather than on the model name, so
    that models whose name does not contain the provider string (e.g.
    ``moonshotai/kimi-k2.5``) are still dispatched to the correct client.
    """
    if llm == "gpt":
        load_dotenv()
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model=model,
            input=message,
        )
        return response.output_text

    if llm == "gemini":
        from llm_base import interact_gemini

        return interact_gemini.call_gemini(model=model, prompt=message, temp=temp)

    if llm in ("llama", "kimi"):
        from llm_base import interact_llama

        return interact_llama.call_llama(model=model, prompt=message, temp=temp, safety_off=True)

    if llm == "local":
        from llm_base.interact_local import LocalClient

        with LocalClient(**local_options) as client:
            reply = client.generate(model, message, temp)
            reply.require_content()
            return reply.content

    raise ValueError(
        f"Unsupported --llm value: {llm!r}. Expected one of: gpt, gemini, llama, kimi, local."
    )


def get_numrun_outputs(log_fpath, prompt, num_runs, stidx, enidx, temp, llm, model='gemini-1.5-pro', dataset='labgen'):
    """Build the message from a v2 Prompt object and run inference ``num_runs`` times."""
    data = helpers.load_logs(log_fpath, stidx, enidx)

    message = prompt.populate(data)

    results = []
    for _ in tqdm(range(num_runs)):
        results.append(call_model(llm, model, message, temp))

    return results


def count_total_lines(fpath):
    line_cnt = 0
    with open(fpath, 'r') as f:
        for line in f:
            line_cnt += 1
    return line_cnt


def write_completionapi_results(results, outpath, st_lno, numlines, llm, misc=''):
    for content in results:
        helpers.write_output(content, outpath, st_lno, numlines, llm, misc)
