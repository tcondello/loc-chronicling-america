#!/usr/bin/env python3
"""Run multi-state streaming pipeline for Chronicling America data.

Processes 10 batches each for NE, NJ, NY, PA, CA, streaming to compressed JSONL
and pushing directly to Hugging Face with atomic commits and scratch space purging.
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

TARGET_STATES = ["NE", "NJ", "NY", "PA", "CA"]
BATCHES_PER_STATE = 10
HF_REPO = "Tim-Pinecone/LOC-Chronicling-America"
OUTPUT_DIR = Path("./export_data")


def main():
    print("=" * 80, flush=True)
    print(f"Starting Multi-State Chronicling America Pipeline", flush=True)
    print(f"Target States      : {', '.join(TARGET_STATES)} ({BATCHES_PER_STATE} batches each, 50 total)", flush=True)
    print(f"Hugging Face Repo  : https://huggingface.co/datasets/{HF_REPO}", flush=True)
    print(f"Output Directory   : {OUTPUT_DIR.resolve()}", flush=True)
    print(f"Zero-Bloat Scratch : Enabled (keep_tar=False)", flush=True)
    print("=" * 80, flush=True)

    pipeline = BatchPipeline(
        output_dir=OUTPUT_DIR,
        scratch_dir=OUTPUT_DIR / "scratch",
        hf_repo=HF_REPO,
        hf_token=os.environ.get("HF_TOKEN"),
        keep_tar=False,
        purge_local_after_upload=False,
    )

    overall_start = time.time()
    total_successful = 0
    total_failed = 0
    total_pages = 0

    for idx, state in enumerate(TARGET_STATES, start=1):
        print(f"\n>>> [{idx}/{len(TARGET_STATES)}] Starting State: {state} ({BATCHES_PER_STATE} batches) ...", flush=True)
        t_state_start = time.time()

        try:
            res = pipeline.run(state=state, limit_batches=BATCHES_PER_STATE)
            s_count = res.get("successful_batches", 0)
            f_count = res.get("failed_batches", 0)
            p_count = res.get("pages_extracted", 0)
            elapsed = res.get("elapsed_seconds", 0)

            total_successful += s_count
            total_failed += f_count
            total_pages += p_count

            print(f">>> State {state} Finished in {elapsed:.1f}s: {s_count} succeeded, {f_count} failed, {p_count:,} pages", flush=True)

        except Exception as e:
            print(f"!!! Error processing state {state}: {e}", flush=True)

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 80, flush=True)
    print("Pipeline Execution Complete!", flush=True)
    print(f"Total Successful Batches : {total_successful} / {len(TARGET_STATES) * BATCHES_PER_STATE}", flush=True)
    print(f"Total Failed Batches     : {total_failed}", flush=True)
    print(f"Total Pages Extracted    : {total_pages:,}", flush=True)
    print(f"Overall Elapsed Time     : {overall_elapsed / 60:.2f} minutes ({overall_elapsed:.1f}s)", flush=True)
    print(f"Hugging Face Dataset     : https://huggingface.co/datasets/{HF_REPO}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
