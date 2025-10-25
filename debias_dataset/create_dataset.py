#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create multiple OpenAI Batch API input JSONL files from DiffusionDB parquet.

- Streams prompts directly from parquet on the Hub (no script loaders).
- Applies your cleanup filter.
- Splits into N files, each with M requests (total = N * M).
- Each JSONL line is a POST /v1/chat/completions request asking only for {"axes":[...]}.

Example:
  python make_diffusiondb_batches.py \
    --out-dir batches_axes \
    --num-files 10 \
    --per-file-requests 30000 \
    --parquet hf://datasets/poloclub/diffusiondb/metadata.parquet \
    --dedupe
"""

import argparse
import hashlib
import json
import os
from typing import Iterable, Dict, List
from tqdm import tqdm
from datasets import load_dataset

# -------------------- Cleanup filter (your rules) --------------------

PUNCT_TO_SPACE = [':', ';', '[', ']', '!', ',']
PUNCT_REMOVE   = ['(', ')']

def cleanup_prompts_stream(ds: Iterable[str]) -> Iterable[str]:
    """
    Yield cleaned prompts one-by-one (streaming).
      - replace : ; [ ] ! , with space
      - remove ( )
      - keep longest chunk by '.'
      - token length: 5..9 inclusive
      - drop if contains 'painting' or 'by', or startswith 'in'
      - de-dup
    """
    seen = set()
    for prompt in ds:
        if not isinstance(prompt, str):
            continue
        for p in PUNCT_TO_SPACE:
            prompt = prompt.replace(p, ' ')
        for p in PUNCT_REMOVE:
            prompt = prompt.replace(p, '')
        prompt = max(prompt.split('.'), key=len).strip()

        toks = prompt.split()
        n = len(toks)
        if n <= 4 or n >= 10:
            continue
        if 'painting' in toks or 'by' in toks or prompt.startswith('in'):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        yield prompt

# -------------------- DiffusionDB via parquet (no remote code) --------------------

def iter_prompts_from_parquet(data_files: Dict[str, str], text_field: str = "prompt") -> Iterable[str]:
    """
    data_files: e.g. {"train": "hf://datasets/poloclub/diffusiondb/metadata.parquet"}
                or multiple shards: {"train": ["hf://.../part1.parquet", "hf://.../part2.parquet"]}
    """
    ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    for ex in ds:
        yield ex.get(text_field, "")

# -------------------- Batch request line builder --------------------

SYSTEM_MSG = (
    'You are a careful annotator. Return STRICT JSON {"axes":[...]}. '
    'Allowed axes: ["gender","race_ethnicity","age","religion","nationality",'
    '"socioeconomic_class","disability","body_type","sexual_orientation","political","other","none"]. '
    'If no bias, return {"axes":["none"]}. No extra keys.'
)
USER_TEMPLATE = "PROMPT: {text}\nReturn JSON ONLY."

def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def to_batch_line(prompt: str, idx: int, model: str = "gpt-4o-mini") -> dict:
    """
    One JSON object for the Batch API input JSONL (POST /v1/chat/completions).
    """
    cid = f"axes-{idx:09d}-{short_hash(prompt)}"
    return {
        "custom_id": cid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_TEMPLATE.format(text=prompt)}
            ]
        }
    }

# -------------------- Main: write N files × M requests --------------------

def main():
    ap = argparse.ArgumentParser(description="Create multiple Batch API input JSONL files from DiffusionDB parquet.")
    ap.add_argument("--out-dir", required=True, help="Directory to write batch JSONL files.")
    ap.add_argument("--num-files", type=int, default=10, help="Number of batch files to produce.")
    ap.add_argument("--per-file-requests", type=int, default=30000, help="Requests per file.")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini).")
    ap.add_argument("--text-field", default="prompt", help="Prompt field in parquet (default: 'prompt').")
    ap.add_argument("--parquet", nargs="+", default=["hf://datasets/poloclub/diffusiondb/metadata.parquet"],
                    help="One or more parquet URIs or paths. Default is the 2M metadata table.")
    ap.add_argument("--dedupe", action="store_true", help="Deduplicate prompts across shards/files.")
    args = ap.parse_args()

    total_needed = args.num_files * args.per_file_requests
    os.makedirs(args.out_dir, exist_ok=True)

    # Build a single streaming iterator over all selected parquet file(s)
    data_files = {"train": args.parquet if len(args.parquet) > 1 else args.parquet[0]}
    stream = cleanup_prompts_stream(iter_prompts_from_parquet(data_files, args.text_field))

    # Optional global dedupe across shards/files
    seen_global = set()
    global_idx = 0
    written_total = 0

    # Open first file
    file_index = 1
    per_file_count = 0
    current_path = os.path.join(args.out_dir, f"axes_input_{file_index:02d}.jsonl")
    out_f = open(current_path, "w", encoding="utf-8")
    print(f"Writing {current_path}")

    try:
        for prompt in tqdm(stream, desc="Streaming & building batches"):
            if args.dedupe and prompt in seen_global:
                continue
            if args.dedupe:
                seen_global.add(prompt)

            # Write one request line
            line = to_batch_line(prompt, global_idx, model=args.model)
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")

            global_idx += 1
            written_total += 1
            per_file_count += 1

            # Rotate file if we hit per-file-requests
            if per_file_count >= args.per_file_requests:
                out_f.close()
                if file_index >= args.num_files:
                    break
                file_index += 1
                per_file_count = 0
                current_path = os.path.join(args.out_dir, f"axes_input_{file_index:02d}.jsonl")
                out_f = open(current_path, "w", encoding="utf-8")
                print(f"Writing {current_path}")

            # Stop once we’ve produced all needed
            if written_total >= total_needed:
                break
    finally:
        try:
            out_f.close()
        except Exception:
            pass

    print(f"Done. Wrote {written_total} requests across {min(args.num_files, (written_total + args.per_file_requests - 1) // args.per_file_requests)} file(s) in {args.out_dir}.")

if __name__ == "__main__":
    main()
