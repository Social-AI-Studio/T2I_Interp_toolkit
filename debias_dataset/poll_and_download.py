#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, argparse
from pathlib import Path
from typing import Optional, Iterable, Dict, Any

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

TERMINAL_STATES = {"completed", "failed", "expired", "cancelling", "cancelled"}

# ---------- env & client ----------

def get_client() -> OpenAI:
    load_dotenv(find_dotenv(), override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")
    return OpenAI(api_key=api_key)

# ---------- helpers: submission, polling, listing, download ----------

def _batch_to_dict(b):
    # Works for OpenAI SDK pydantic models and plain dicts
    try:
        return b.model_dump()                  # new SDK
    except Exception:
        try:
            return json.loads(b.json())        # older pydantic models
        except Exception:
            try:
                return b.__dict__              # fallback
            except Exception:
                return {"repr": repr(b)}


def upload_batch_file(client: OpenAI, jsonl_path: str) -> str:
    with open(jsonl_path, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    return up.id

def create_batch(client: OpenAI, input_file_id: str, endpoint: str, window: str = "24h") -> str:
    b = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=window,
    )
    return b.id

def retrieve_batch(client: OpenAI, batch_id: str):
    return client.batches.retrieve(batch_id)

def list_batches(client: OpenAI, limit: int = 100) -> Iterable[Any]:
    # SDK returns a cursor-like paginator
    return client.batches.list(limit=limit)

def poll_until_terminal(client: OpenAI, batch_id: str, interval: float = 5.0) -> Any:
    while True:
        b = retrieve_batch(client, batch_id)
        print(f"[{batch_id}] status:", b.status)
        if b.status in TERMINAL_STATES:
            return b
        time.sleep(interval)

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _parse_one_result_line(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns {"id": custom_id, "axes": [...]} or None for non-success.
    Supports /v1/chat/completions (choices[0].message.content) and /v1/responses (output_text).
    """
    if "response" not in obj:
        return None

    custom_id = obj.get("custom_id")

    # Try Chat Completions
    try:
        msg = obj["response"]["body"]["choices"][0]["message"]["content"]
        axes = json.loads(msg).get("axes", ["none"])
        return {"id": custom_id, "axes": axes}
    except Exception:
        pass

    # Try Responses API
    try:
        # Some SDKs return "output_text"; in raw REST result inside batch, body may have "output"
        body = obj["response"]["body"]
        if "output" in body:
            # /v1/responses structured output
            # Collect text parts only
            texts = []
            for item in body["output"]:
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
            msg = "\n".join(texts).strip()
        else:
            # fallback: stitched text field sometimes present
            msg = body.get("output_text", "")

        if msg:
            axes = json.loads(msg).get("axes", ["none"])
            return {"id": custom_id, "axes": axes}
    except Exception:
        pass

    return None

def download_and_parse_results(client: OpenAI, batch, out_dir: str) -> str:
    """
    Saves raw output shard(s) and a parsed axes JSONL.
    Returns path to parsed file.
    """
    ensure_dir(out_dir)
    out_parsed = os.path.join(out_dir, f"{batch.id}_parsed.jsonl")
    out_raw_prefix = os.path.join(out_dir, f"{batch.id}_raw")

    output_files = batch.output_files or []
    if not output_files:
        raise RuntimeError(f"[{batch.id}] No output files attached")

    parsed = 0
    with open(out_parsed, "w", encoding="utf-8") as parsed_f:
        for idx, fmeta in enumerate(output_files):
            fid = fmeta["id"] if isinstance(fmeta, dict) else fmeta.id
            content = client.files.content(fid)

            raw_path = f"{out_raw_prefix}_{idx}.jsonl"
            with open(raw_path, "w", encoding="utf-8") as raw_f:
                for line in content.text.splitlines():
                    raw_f.write(line + "\n")
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if "response" in obj:
                        rec = _parse_one_result_line(obj)
                        if rec is not None:
                            parsed_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            parsed += 1
                    elif "error" in obj:
                        # Keep errors in raw; skip parsing
                        pass

    print(f"[{batch.id}] Wrote parsed={parsed} to {out_parsed}")
    return out_parsed

# ---------- CLI commands ----------

def cmd_submit(args):
    client = get_client()
    file_id = upload_batch_file(client, args.input)
    batch_id = create_batch(client, file_id, endpoint=args.endpoint, window=args.window)
    print("Batch created:", batch_id)

def cmd_submit_and_poll(args):
    client = get_client()
    file_id = upload_batch_file(client, args.input)
    batch_id = create_batch(client, file_id, endpoint=args.endpoint, window=args.window)
    print("Batch created:", batch_id)
    b = poll_until_terminal(client, batch_id, interval=args.interval)
    if b.status != "completed":
        raise RuntimeError(f"[{batch_id}] ended with status={b.status}")
    download_and_parse_results(client, b, out_dir=args.out_dir)

def cmd_poll(args):
    client = get_client()
    b = poll_until_terminal(client, args.batch_id, interval=args.interval)
    if b.status != "completed":
        raise RuntimeError(f"[{args.batch_id}] ended with status={b.status}")
    download_and_parse_results(client, b, out_dir=args.out_dir)

def cmd_poll_all(args):
    client = get_client()
    # Collect all non-terminal batches first
    active = []
    for b in list_batches(client, limit=args.limit):
        if b.status not in TERMINAL_STATES:
            active.append(b)

    if not active:
        print("No active batches to poll.")
        return

    # Show full objects before polling
    print("Active batches (full objects):")
    for b in active:
        print(json.dumps(_batch_to_dict(b), indent=2, ensure_ascii=False, default=str))

    # Poll each until done, then download
    for b in active:
        bid = b.id
        bdone = poll_until_terminal(client, bid, interval=args.interval)
        if bdone.status == "completed":
            download_and_parse_results(client, bdone, out_dir=args.out_dir)
        else:
            print(f"[{bid}] ended with status={bdone.status} (no download)")

def cmd_list(args):
    client = get_client()
    for b in list_batches(client, limit=args.limit):
        if args.status and b.status != args.status:
            continue
        print(b.id, b.status, getattr(b, "request_counts", None))

def cmd_download(args):
    client = get_client()
    b = retrieve_batch(client, args.batch_id)
    if b.status != "completed":
        raise RuntimeError(f"[{args.batch_id}] not completed (status={b.status})")
    download_and_parse_results(client, b, out_dir=args.out_dir)

# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description="OpenAI Batch helper: submit, poll, poll-all, list, download.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # submit
    sp = sub.add_parser("submit", help="Upload JSONL and create a batch (no polling).")
    sp.add_argument("--input", required=True, help="Batch input JSONL path.")
    sp.add_argument("--endpoint", default="/v1/chat/completions", help="API endpoint for the batch.")
    sp.add_argument("--window", default="24h", help="Completion window (e.g., 24h).")
    sp.set_defaults(func=cmd_submit)

    # submit-and-poll
    sp = sub.add_parser("submit-and-poll", help="Submit then poll until done, then download+parse.")
    sp.add_argument("--input", required=True, help="Batch input JSONL path.")
    sp.add_argument("--endpoint", default="/v1/chat/completions", help="API endpoint for the batch.")
    sp.add_argument("--window", default="24h", help="Completion window.")
    sp.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds.")
    sp.add_argument("--out-dir", default="batch_results", help="Directory for outputs.")
    sp.set_defaults(func=cmd_submit_and_poll)

    # poll
    sp = sub.add_parser("poll", help="Poll a specific batch id until terminal, then download+parse if completed.")
    sp.add_argument("--batch-id", required=True, help="Batch id to poll.")
    sp.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds.")
    sp.add_argument("--out-dir", default="batch_results", help="Directory for outputs.")
    sp.set_defaults(func=cmd_poll)

    # poll-all
    sp = sub.add_parser("poll-all", help="Poll all active batches, then download+parse completed ones.")
    sp.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds.")
    sp.add_argument("--out-dir", default="batch_results", help="Directory for outputs.")
    sp.add_argument("--limit", type=int, default=100, help="Max batches to list/poll.")
    sp.set_defaults(func=cmd_poll_all)

    # list
    sp = sub.add_parser("list", help="List batches (optionally filter by status).")
    sp.add_argument("--status", default="", help="Filter by status (queued|running|completed|...).")
    sp.add_argument("--limit", type=int, default=100, help="Max batches to list.")
    sp.set_defaults(func=cmd_list)

    # download
    sp = sub.add_parser("download", help="Download+parse results for a completed batch id.")
    sp.add_argument("--batch-id", required=True, help="Completed batch id.")
    sp.add_argument("--out-dir", default="batch_results", help="Directory for outputs.")
    sp.set_defaults(func=cmd_download)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
