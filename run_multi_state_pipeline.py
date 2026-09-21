#!/usr/bin/env python3
"""Run multi-state streaming pipeline for Chronicling America data.

Streams batches to compressed JSONL and pushes directly to Hugging Face
with atomic commits and automatic scratch space purging.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from loc_chronicling_america.pipeline import BatchPipeline

# Load .env if present
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(description="Run multi-state Chronicling America streaming export pipeline")
    parser.add_argument("--states", nargs="+", default=["NE", "NJ", "NY", "PA", "CA"], help="States to process")
    parser.add_argument("--limit-per-state", type=int, default=None, help="Max batches per state (default: all pending)")
    parser.add_argument("--hf-repo", default="Tim-Pinecone/LOC-Chronicling-America", help="Hugging Face repo id")
    parser.add_argument("--output-dir", default="./export_data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("=" * 80, flush=True)
    print("Starting Multi-State Chronicling America Pipeline", flush=True)
    print(f"Target States      : {', '.join(args.states)}", flush=True)
    print(f"Batches per State  : {'ALL pending' if args.limit_per_state is None else args.limit_per_state}", flush=True)
    print(f"Hugging Face Repo  : https://huggingface.co/datasets/{args.hf_repo}", flush=True)
    print(f"Output Directory   : {out_dir.resolve()}", flush=True)
    print(f"Zero-Bloat Scratch : Enabled (keep_tar=False)", flush=True)
    print("=" * 80, flush=True)

    pipeline = BatchPipeline(
        output_dir=out_dir,
        scratch_dir=out_dir / "scratch",
        hf_repo=args.hf_repo,
        hf_token=os.environ.get("HF_TOKEN"),
        keep_tar=False,
        purge_local_after_upload=False,
    )

    # Ensure any failed batches are reset to pending for retry
    with pipeline.db._get_connection() as conn:
        conn.execute("UPDATE pipeline_batches SET status = 'pending', error_message = NULL WHERE status = 'failed'")
        conn.commit()

    overall_start = time.time()
    total_successful = 0
    total_failed = 0
    total_pages = 0

    for idx, state in enumerate(args.states, start=1):
        pending_list = pipeline.db.get_pending_pipeline_batches(state=state, limit=args.limit_per_state)
        batch_count = len(pending_list)
        print(f"\n>>> [{idx}/{len(args.states)}] Starting State: {state} ({batch_count} pending batches) ...", flush=True)

        if batch_count == 0:
            print(f"    No pending batches for {state}. Moving to next state.", flush=True)
            continue

        try:
            res = pipeline.run(state=state, limit_batches=args.limit_per_state)
            s_count = res.get("successful_batches", 0)
            f_count = res.get("failed_batches", 0)
            p_count = res.get("pages_extracted", 0)
            elapsed = res.get("elapsed_seconds", 0)

            total_successful += s_count
            total_failed += f_count
            total_pages += p_count

            print(f">>> State {state} Finished in {elapsed / 60:.1f}m: {s_count} succeeded, {f_count} failed, {p_count:,} pages", flush=True)

        except Exception as e:
            print(f"!!! Error processing state {state}: {e}", flush=True)

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 80, flush=True)
    print("Multi-State Pipeline Execution Complete!", flush=True)
    print(f"Total Successful Batches : {total_successful}", flush=True)
    print(f"Total Failed Batches     : {total_failed}", flush=True)
    print(f"Total Pages Extracted    : {total_pages:,}", flush=True)
    print(f"Overall Elapsed Time     : {overall_elapsed / 60:.2f} minutes ({overall_elapsed:.1f}s)", flush=True)
    print(f"Hugging Face Dataset     : https://huggingface.co/datasets/{args.hf_repo}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
