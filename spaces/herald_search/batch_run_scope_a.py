#!/usr/bin/env python3
"""Batch runner orchestrating Scope A ingestion for all 5 expanded publications.

Executes sequential streaming ingestion with atomic checkpoints and zero disk accumulation:
1. Chicago Eagle (344 issues)
2. The Beatrice Daily Express (2,570 issues)
3. The Evening World (2,230 issues)
4. The San Francisco Call (2,157 issues)
5. The Sun (2,895 issues)

Usage:
    python spaces/herald_search/batch_run_scope_a.py
    python spaces/herald_search/batch_run_scope_a.py --batch-size 250
    python spaces/herald_search/batch_run_scope_a.py --newspapers chicago_eagle,the_beatrice_daily_express
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from pinecone import Pinecone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config
from stream_ingest import process_newspaper

load_dotenv()
load_dotenv(HERE / ".env")

SCOPE_A_NEWSPAPERS = [
    "chicago_eagle",
    "the_beatrice_daily_express",
    "the_evening_world",
    "the_san_francisco_call",
    "the_sun",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Scope A ingestion across the 5 expanded broadsheets")
    parser.add_argument(
        "--newspapers",
        default=",".join(SCOPE_A_NEWSPAPERS),
        help=f"Comma-separated list of newspapers to ingest (default: {','.join(SCOPE_A_NEWSPAPERS)})",
    )
    parser.add_argument(
        "--years",
        default=",".join(config.TARGET_YEARS),
        help=f"Comma-separated list of target years (default: {','.join(config.TARGET_YEARS)})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of chunks per OpenAI embedding & Pinecone upsert batch (default: 200)",
    )
    parser.add_argument(
        "--limit-issues-per-paper",
        type=int,
        default=None,
        help="Optional issue limit per newspaper (for testing/canary runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and chunk without calling OpenAI or Pinecone APIs",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Wipe checkpoint files and re-ingest from issue 1",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not args.dry_run:
        if not api_key:
            print("Error: Missing PINECONE_API_KEY in environment.", file=sys.stderr)
            sys.exit(1)
        if not openai_key:
            print("Error: Missing OPENAI_API_KEY in environment.", file=sys.stderr)
            sys.exit(1)

    pc = Pinecone(api_key=api_key) if not args.dry_run else None
    oai = OpenAI(api_key=openai_key) if not args.dry_run else None
    index = pc.Index(config.INDEX_NAME) if not args.dry_run else None
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    target_newspapers = [n.strip() for n in args.newspapers.split(",") if n.strip()]
    target_years = [y.strip() for y in args.years.split(",") if y.strip()]

    print("=" * 80)
    print("🚀 STARTING SCOPE A FULL INGESTION PIPELINE")
    print(f"  Target Publications: {len(target_newspapers)} broadsheets ({', '.join(target_newspapers)})")
    print(f"  Target Years       : {len(target_years)} years ({', '.join(target_years)})")
    print(f"  Batch Size         : {args.batch_size} chunks/flush")
    print(f"  Embedding Model    : {config.EMBED_MODEL} (1536 dims)")
    print(f"  Pinecone Index     : {config.INDEX_NAME}")
    print(f"  Dry-Run Mode       : {args.dry_run}")
    print("=" * 80)

    t0_global = time.time()
    newspaper_stats = []

    for idx, np_slug in enumerate(target_newspapers, 1):
        if np_slug not in config.NEWSPAPERS:
            print(f"⚠ Skipping unknown newspaper: {np_slug}")
            continue

        meta = config.NEWSPAPERS[np_slug]
        print(f"\n[{idx}/{len(target_newspapers)}] Starting {meta['title']} ({meta['region']})...")
        t0_paper = time.time()

        process_newspaper(
            np_slug=np_slug,
            years=target_years,
            batch_size=args.batch_size,
            limit_issues=args.limit_issues_per_paper,
            dry_run=args.dry_run,
            reset_checkpoint=args.reset_checkpoint,
            oai=oai,
            index=index,
            api=api,
        )

        elapsed_paper = time.time() - t0_paper
        newspaper_stats.append((meta["title"], elapsed_paper))

    total_elapsed = time.time() - t0_global

    print("\n" + "=" * 80)
    print("🏁 ALL REQUESTED INGESTION WORK COMPLETED!")
    print(f"  Total Duration: {total_elapsed / 60:.1f} minutes ({total_elapsed / 3600:.2f} hours)")
    print("  Publication Durations:")
    for title, dur in newspaper_stats:
        print(f"    - {title}: {dur / 60:.1f}m")
    print("=" * 80)


if __name__ == "__main__":
    main()
