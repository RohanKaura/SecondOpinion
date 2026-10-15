"""Repair model replies that are almost-valid JSON, without touching the originals.

Reads every output file in  inference/*_expt_dataset/task_*/*/output/<llm>out/
and writes a copy to         inference/*_expt_dataset/task_*/*/output/<llm>-canonout/

Only replies that fail to parse as JSON are changed. Each one is repaired if it can
be (string pieces joined with '+', markdown fences, text around the JSON) and
re-written as standard JSON with one field per line, which is what the scorer's
line-by-line parser expects. Replies that are already valid, and empty replies,
are copied unchanged. Replies that still can't be parsed are copied unchanged and
listed at the end.

Run from the repo's top-level folder:
    python scripts/canonicalize_outputs.py --llm llama
Then score with  --llm llama-canon  instead of  --llm llama.
"""
import argparse
import glob
import json
import os
import re

HEADER = re.compile(r"^\*\*\*\*Starting line num: .*\*\*\*\*$", re.M)


def try_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def repair(body):
    """Return (new_body, status) where status is ok | empty | repaired | unrepaired."""
    text = body.strip()
    if not text:
        return body, "empty"
    if try_parse(text) is not None:
        return body, "ok"

    candidate = re.sub(r"```(?:json)?", "", text)                         # markdown fences, any number
    candidate = re.sub(r'"\s*\+\s*\n\s*"', "", candidate)                 # "a" +\n "b"  ->  "ab"
    obj = try_parse(candidate)
    if obj is None:
        obj = collect_json(candidate)        # several objects, or JSON mixed with prose
    if obj is None:
        return body, "unrepaired"
    return "\n" + json.dumps(obj, indent=4, ensure_ascii=False) + "\n", "repaired"


def collect_json(text):
    """Pull every top-level JSON object or array out of text; return one list, or None."""
    dec, found, i = json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i] in "{[":
            try:
                val, end = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            found.extend(val if isinstance(val, list) else [val])
            i = end
        else:
            i += 1
    found = [x for x in found if isinstance(x, dict)]
    return found or None


def process_file(src, dst):
    with open(src, encoding="utf-8") as source:
        raw = source.read()
    headers = list(HEADER.finditer(raw))
    out, report = [], []
    out.append(raw[: headers[0].start()] if headers else raw)
    for i, h in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        block = raw[h.end():end]
        sep = block.rfind("\n-----------------------")       # writer's end-of-reply line
        body, tail = (block[:sep], block[sep:]) if sep != -1 else (block, "")
        new_body, status = repair(body)
        out.append(h.group(0) + new_body + tail)
        report.append((h.group(0), status))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="output folder label used at inference, e.g. llama")
    ap.add_argument("--model", default="", help="only files whose name contains this, e.g. openai-gpt-oss-120b")
    args = ap.parse_args()

    files = sorted(glob.glob(f"inference/*_expt_dataset/task_*/*/output/{args.llm}out/*.txt"))
    files = [f for f in files if args.model in os.path.basename(f)]
    if not files:
        raise SystemExit("No output files found. Run this from the repo's top-level folder.")

    totals, problems = {"ok": 0, "empty": 0, "repaired": 0, "unrepaired": 0}, []
    for src in files:
        dst = src.replace(f"/output/{args.llm}out/", f"/output/{args.llm}-canonout/")
        for header, status in process_file(src, dst):
            totals[status] += 1
            if status in ("repaired", "unrepaired"):
                problems.append((status, os.path.basename(src), header))

    print(f"Files: {len(files)}   Replies: {sum(totals.values())}   " +
          "   ".join(f"{k}: {v}" for k, v in totals.items()))
    for status, name, header in problems:
        print(f"  {status.upper():10s} {name}\n             {header}")
    print(f"\nCopies written to output/{args.llm}-canonout/. Score with --llm {args.llm}-canon")


if __name__ == "__main__":
    main()
