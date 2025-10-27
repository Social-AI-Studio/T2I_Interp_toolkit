#!/usr/bin/env python3
import os, sys, re, json, glob, argparse
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import HfApi

# ---------- IO ----------
def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON in {path}:{ln}: {e}", file=sys.stderr)

# ---------- Parsing ----------
PROMPT_RE = re.compile(
    r"(?is)^\s*PROMPT:\s*(.*?)\s*(?:Return\s+JSON\s+ONLY\.?|$)"
)

def extract_prompt_from_user_content(content: str) -> str:
    """
    Robustly extract user prompt:
    - If 'PROMPT:' present, take text after it, stopping before 'Return JSON ONLY' if present.
    - Else, use entire content as prompt.
    """
    if not isinstance(content, str):
        return ""
    m = PROMPT_RE.search(content)
    if m:
        return m.group(1).strip()
    return content.strip()

def collect_prompts(input_paths: List[str]) -> Dict[str, str]:
    """custom_id -> prompt (from inputs)"""
    cid2prompt: Dict[str, str] = {}
    for p in input_paths:
        for ex in read_jsonl(p):
            cid = ex.get("custom_id")
            if not cid:
                continue
            body = (ex.get("body") or {})
            messages = body.get("messages") or []
            # take the last user message (common pattern)
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                cid2prompt[cid] = ""
                continue
            content = user_msgs[-1].get("content", "")
            prompt = extract_prompt_from_user_content(content)
            cid2prompt[cid] = prompt
    return cid2prompt

def parse_axes_from_batch_row(row: dict) -> Optional[List[str]]:
    """Extract model-produced axes list from a batch output row."""
    resp = row.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return None
    content = content.strip()
    try:
        obj = json.loads(content)
    except Exception:
        return None
    axes = obj.get("axes")
    if isinstance(axes, list) and all(isinstance(a, str) for a in axes):
        # keep as-is (e.g., ["none"] allowed)
        return [a.strip() for a in axes]
    return None

def collect_bias_axes_from_batches(batch_paths: List[str]) -> Dict[str, List[str]]:
    """custom_id -> bias_axes (from model outputs)"""
    out: Dict[str, List[str]] = {}
    for p in batch_paths:
        for row in read_jsonl(p):
            cid = row.get("custom_id")
            if not cid:
                continue
            axes = parse_axes_from_batch_row(row)
            if axes is not None:
                out[cid] = axes
    return out

# ---------- HF dataset ----------
def build_rows(cid2prompt: Dict[str, str], cid2axes: Dict[str, List[str]]) -> List[dict]:
    ids = sorted(set(cid2prompt.keys()) | set(cid2axes.keys()))
    rows = []
    for cid in ids:
        rows.append({
            "prompt": cid2prompt.get(cid, ""),
            "bias_axes": cid2axes.get(cid)  # may be None if no output yet
        })
    return rows

def dataset_from_rows(rows: List[dict]) -> Dataset:
    feats = Features({
        "prompt": Value("string"),
        "bias_axes": Sequence(Value("string")),
    })
    return Dataset.from_list(rows).cast(feats)

def push_hf(ds: Dataset, repo_id: str, private: bool = True, token: Optional[str] = None):
    token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    ds.push_to_hub(repo_id, private=private, token=token)
    print(f"[OK] Pushed {len(ds)} rows to hf:{repo_id}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with axes_input*.jsonl and batch_*.jsonl")
    ap.add_argument("--repo_id", required=True, help="username/dataset_name")
    ap.add_argument("--out_jsonl", default="axes_min.jsonl", help="Write merged JSONL with 2 columns")
    ap.add_argument("--public", action="store_true", help="Make HF dataset public")
    ap.add_argument("--only_paired", action="store_true",
                    help="Keep only rows that have BOTH prompt and bias_axes")
    args = ap.parse_args()

    folder = Path(args.folder)
    input_paths = sorted(glob.glob(str(folder / "axes_input*.jsonl")))
    batch_paths = sorted(glob.glob(str(folder / "batch_*.jsonl")))

    if not input_paths:
        print(f"[WARN] No inputs at {folder/'axes_input*.jsonl'}", file=sys.stderr)
    if not batch_paths:
        print(f"[WARN] No batches at {folder/'batch_*.jsonl'}", file=sys.stderr)

    cid2prompt = collect_prompts(input_paths)
    cid2axes   = collect_bias_axes_from_batches(batch_paths)

    rows = build_rows(cid2prompt, cid2axes)
    if args.only_paired:
        rows = [r for r in rows if r["prompt"] and r["bias_axes"]]

    # Write JSONL preview
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {args.out_jsonl} with {len(rows)} rows")

    # Push to HF
    ds = dataset_from_rows(rows)
    push_hf(ds, repo_id=args.repo_id, private=(not args.public))

if __name__ == "__main__":
    main()
