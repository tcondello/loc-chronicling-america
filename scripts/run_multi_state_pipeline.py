#!/usr/bin/env python3
"""Run multi-state streaming pipeline for Chronicling America data.

Streams batches to compressed Apache Parquet and pushes directly to Hugging Face
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


HIGH_VALUE_36H_STATES = [
    "AK", "NE", "IL", "CA", "NY", "PA", "TX", "OH", "FL", "WA", "CO", "MO", "NC", "MA", "VA", "MN", "GA", "MI"
]


def main():
    parser = argparse.ArgumentParser(description="Run multi-state Chronicling America streaming export pipeline")
    parser.add_argument("--states", nargs="+", default=["NE", "NJ", "NY", "PA", "CA"], help="States to process (default: NE NJ NY PA CA)")
    parser.add_argument("--high-value-36h", action="store_true", help="Process prioritized high-value states totaling ~36 hours (AK, NE, IL, CA, NY, PA, TX, OH, FL, WA, CO, MO, NC, MA, VA, MN, GA, MI)")
    parser.add_argument("--all-states", action="store_true", help="Process ALL states in the catalog")
    parser.add_argument("--limit-per-state", type=int, default=None, help="Max batches per state (default: all pending)")
    parser.add_argument("--purge-local-after-upload", action="store_true", help="Delete local Parquet files after uploading to Hugging Face to conserve disk")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly reset all failed batches back to pending for retry")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO", "Tim-Pinecone/LOC-Chronicling-America"), help="Hugging Face repo id")
    parser.add_argument("--output-dir", default="./export_data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    pipeline = BatchPipeline(
        output_dir=out_dir,
        scratch_dir=out_dir / "scratch",
        hf_repo=args.hf_repo,
        hf_token=os.environ.get("HF_TOKEN"),
        keep_tar=False,
        purge_local_after_upload=args.purge_local_after_upload,
    )

    # Auto-initialize catalog if running on a fresh machine
    if pipeline.db.count_batches() == 0:
        print("Catalog database is empty. Auto-synchronizing metadata from Library of Congress...", flush=True)
        from loc_chronicling_america.client import ChroniclingAmerica
        client = ChroniclingAmerica(catalog_db=pipeline.db, db_path=pipeline.db.db_path)
        b_count = client.sync_batches()
        t_count = client.sync_titles()
        print(f"✓ Initialized catalog with {b_count:,} batches and {t_count:,} titles.", flush=True)

    # Determine target states
    if args.high_value_36h:
        target_states = HIGH_VALUE_36H_STATES
    elif args.all_states or (len(args.states) == 1 and args.states[0].upper() == "ALL"):
        with pipeline.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT state FROM batches WHERE state IS NOT NULL ORDER BY state")
            target_states = [r[0] for r in cur.fetchall()]
    else:
        target_states = args.states

    print("=" * 80, flush=True)
    print("Starting Multi-State Chronicling America Pipeline", flush=True)
    print(f"Target States      : {', '.join(target_states)} ({len(target_states)} total)", flush=True)
    print(f"Batches per State  : {'ALL pending' if args.limit_per_state is None else args.limit_per_state}", flush=True)
    print(f"Hugging Face Repo  : https://huggingface.co/datasets/{args.hf_repo}", flush=True)
    print(f"Output Directory   : {out_dir.resolve()}", flush=True)
    print(f"Zero-Bloat Scratch : Enabled (keep_tar=False)", flush=True)
    print(f"Purge Local Parquet: {'Enabled (--purge-local-after-upload)' if args.purge_local_after_upload else 'Disabled (retaining local copy)'}", flush=True)
    print("=" * 80, flush=True)

    # Handle manual retries if requested
    if args.retry_failed:
        re_count = pipeline.db.mark_failed_batches_pending()
        if re_count > 0:
            print(f"✓ Reset {re_count} failed batches to pending (--retry-failed).", flush=True)

    # Cleanly recover any interrupted batches without infinite crash loops
    reset_pending, marked_failed = pipeline.db.reset_interrupted_batches(max_attempts=2)
    if reset_pending > 0 or marked_failed > 0:
        print(f"Pipeline recovery: {reset_pending} interrupted batches queued for retry, {marked_failed} exceeding max attempts marked failed.", flush=True)

    overall_start = time.time()
    total_successful = 0
    total_failed = 0
    total_pages = 0

    for idx, state in enumerate(target_states, start=1):
        pending_list = pipeline.db.get_pending_pipeline_batches(state=state, limit=args.limit_per_state)
        batch_count = len(pending_list)
        print(f"\n>>> [{idx}/{len(target_states)}] Starting State: {state} ({batch_count} pending batches) ...", flush=True)

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

    # Finalize catalog index and Hugging Face Dataset Card for all processed states
    print("\nFinalizing master catalog and Hugging Face Dataset Card...", flush=True)
    pipeline.update_catalog_index()
    if pipeline.hf_manager:
        readme_path = pipeline.output_dir / "README.md"
        states = [d.name for d in (pipeline.output_dir / "newspapers").iterdir() if d.is_dir()]
        pipeline.hf_manager.generate_readme(readme_path, states=sorted(states))
        pipeline.hf_manager.upload_file(readme_path, "README.md")
        cat_p = pipeline.output_dir / "catalog.parquet"
        if cat_p.exists():
            pipeline.hf_manager.upload_file(cat_p, "catalog.parquet")

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
