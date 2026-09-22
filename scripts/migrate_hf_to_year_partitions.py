#!/usr/bin/env python3
"""Migrate flat Hugging Face dataset layout to year-partitioned layout.

Converts:
  newspapers/{state}/{newspaper_slug}/{filename}.parquet
To:
  newspapers/{state}/{newspaper_slug}/{year}/{filename}.parquet

Uses Hugging Face's server-side CommitOperationCopy and CommitOperationDelete,
requiring zero local file downloads or bandwidth.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.utils import HfHubHTTPError

# Auto-load .env
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate Hugging Face Parquet files to year-partitioned directory layout"
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO", "Tim-Pinecone/LOC-Chronicling-America"),
        help="Hugging Face repo ID (default: HF_REPO env or Tim-Pinecone/LOC-Chronicling-America)",
    )
    parser.add_argument(
        "--newspaper",
        default=None,
        help="Filter to a specific newspaper slug (e.g. omaha_daily_bee)",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Filter to a specific state slug (e.g. nebraska)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Number of files to move per commit (default: 250, which is 500 copy+delete ops)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum total files to migrate in this run (default: all matching)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and list files to move without executing commits",
    )
    return parser.parse_args()


def get_moves(
    files: List[str], state_filter: str = None, newspaper_filter: str = None
) -> List[Tuple[str, str]]:
    """Identifies flat parquet files that need to be moved to a year subfolder."""
    moves = []
    # Match: newspapers/{state}/{slug}/{filename}.parquet
    pattern = re.compile(r"^newspapers/([^/]+)/([^/]+)/([^/]+_(\d{4})_\d{2}_\d{2}\.parquet)$")

    for f in files:
        m = pattern.match(f)
        if not m:
            continue
        state, slug, filename, year = m.groups()
        if state_filter and state.lower() != state_filter.lower():
            continue
        if newspaper_filter and slug.lower() != newspaper_filter.lower():
            continue

        dest = f"newspapers/{state}/{slug}/{year}/{filename}"
        moves.append((f, dest))

    return moves


def commit_batch_with_retry(
    api: HfApi,
    repo_id: str,
    operations: list,
    commit_message: str,
    max_retries: int = 5,
):
    for attempt in range(1, max_retries + 1):
        try:
            return api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
        except HfHubHTTPError as e:
            if attempt == max_retries:
                raise
            sleep_time = attempt * 5
            print(f"  [Warning] Commit failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Connecting to Hugging Face dataset: {args.repo_id}")
    print("Listing repository files (this may take a few moments)...", flush=True)
    all_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    print(f"Total files currently in repository: {len(all_files):,}")

    moves = get_moves(
        all_files, state_filter=args.state, newspaper_filter=args.newspaper
    )

    if args.max_files:
        moves = moves[: args.max_files]

    total_moves = len(moves)
    print(f"\nIdentified {total_moves:,} flat Parquet files to migrate to year partitions.")

    if total_moves == 0:
        print("No files need migration.")
        return

    # Breakdown by newspaper
    from collections import Counter

    paper_counts = Counter("/".join(src.split("/")[:3]) for src, _ in moves)
    print("\nFiles to move by newspaper (top 10):")
    for paper, count in paper_counts.most_common(10):
        print(f"  {count:5d} files in {paper}")
    if len(paper_counts) > 10:
        print(f"  ... and {len(paper_counts) - 10} more newspapers")

    if args.dry_run:
        print("\n[DRY RUN] Sample operations:")
        for src, dest in moves[:5]:
            print(f"  Copy: {src} -> {dest}")
            print(f"  Del : {src}")
        print("\nDry run complete. No changes made.")
        return

    # Execute commits in batches
    batch_size = args.batch_size
    num_batches = (total_moves + batch_size - 1) // batch_size
    print(
        f"\nStarting migration in {num_batches} commit batch(es) of up to {batch_size} files each..."
    )

    start_time = time.time()
    for batch_idx in range(num_batches):
        batch_slice = moves[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        operations = []
        for src, dest in batch_slice:
            operations.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dest))
            operations.append(CommitOperationDelete(path_in_repo=src))

        commit_msg = (
            f"refactor(partition): migrate {len(batch_slice)} files to year partitions "
            f"({batch_idx + 1}/{num_batches})"
        )

        t0 = time.time()
        commit_info = commit_batch_with_retry(
            api=api,
            repo_id=args.repo_id,
            operations=operations,
            commit_message=commit_msg,
        )
        t_batch = time.time() - t0

        elapsed = time.time() - start_time
        processed = min((batch_idx + 1) * batch_size, total_moves)
        pct = (processed / total_moves) * 100
        rate = processed / elapsed if elapsed > 0 else 0
        rem_sec = (total_moves - processed) / rate if rate > 0 else 0

        print(
            f"  [{batch_idx + 1}/{num_batches}] ({pct:5.1f}%) "
            f"Migrated {len(batch_slice)} files in {t_batch:.1f}s | "
            f"Rate: {rate:.1f} files/s | ETA: {rem_sec / 60:.1f}m"
        )

    print(
        f"\nMigration completed successfully! Migrated {total_moves:,} files in "
        f"{(time.time() - start_time) / 60:.2f} minutes."
    )


if __name__ == "__main__":
    main()
