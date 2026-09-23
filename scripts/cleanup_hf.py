#!/usr/bin/env python3
"""Clean up Hugging Face dataset files and reset SQLite pipeline state.

Provides granular control to wipe specific states or reset the entire dataset
so that batches can be re-extracted with updated schemas or URL mappings.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from loc_chronicling_america.db import CatalogDB
from loc_chronicling_america.hf import HuggingFaceDatasetManager

# Load .env if present
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def clean_state(
    api: HfApi,
    repo_id: str,
    state: str,
    db: CatalogDB,
    dry_run: bool = False,
) -> None:
    """Delete a single state's files from Hugging Face and reset its SQLite batches."""
    clean_st = state.lower().replace(" ", "_").replace("-", "_")
    path_in_repo = f"newspapers/{clean_st}"

    print(f"\n--- Cleaning State: {clean_st} ---")
    if dry_run:
        print(f"  [DRY-RUN] Would delete remote folder '{path_in_repo}' from {repo_id}")
    else:
        try:
            print(f"  Deleting remote folder '{path_in_repo}' on Hugging Face...")
            api.delete_folder(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Clean up state {clean_st} data",
            )
            print(f"  ✓ Deleted remote {path_in_repo}")
        except Exception as e:
            print(f"  Notice: Could not delete remote '{path_in_repo}' (may already be deleted): {e}")

    # Reset in SQLite database
    if dry_run:
        print(f"  [DRY-RUN] Would reset SQLite batches for state '{state}' to pending")
    else:
        with db._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE pipeline_batches
                SET status = 'pending',
                    pages_extracted = 0,
                    output_files = '[]',
                    started_at = NULL,
                    completed_at = NULL,
                    error_message = NULL,
                    attempts = 0
                WHERE batch_name IN (
                    SELECT name FROM batches WHERE UPPER(state) = UPPER(?)
                )
                """,
                (state,),
            )
            count = cur.rowcount
            conn.commit()
            print(f"  ✓ Reset {count} batches in SQLite for state '{state}' to 'pending'.")


def clean_all(
    api: HfApi,
    repo_id: str,
    db: CatalogDB,
    recreate_repo: bool = False,
    dry_run: bool = False,
) -> None:
    """Wipe all newspaper data from Hugging Face and reset all SQLite batches."""
    print("\n--- Full Dataset Cleanup ---")

    if recreate_repo:
        if dry_run:
            print(f"  [DRY-RUN] Would delete and recreate repo {repo_id}")
        else:
            print(f"  Recreating entire Hugging Face repository '{repo_id}' for clean Git history...")
            try:
                api.delete_repo(repo_id=repo_id, repo_type="dataset")
                print("  ✓ Deleted remote repo")
            except Exception as e:
                print(f"  Notice during delete_repo: {e}")

            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            print("  ✓ Created fresh empty repo")
            # Push clean README
            manager = HuggingFaceDatasetManager(repo_id=repo_id, token=api.token)
            readme_p = Path("/tmp/README_clean.md")
            manager.generate_readme(readme_p, states=[])
            manager.upload_file(readme_p, "README.md")
            print("  ✓ Initialized fresh README.md")
    else:
        if dry_run:
            print(f"  [DRY-RUN] Would delete remote folder 'newspapers' from {repo_id}")
            print(f"  [DRY-RUN] Would delete remote 'catalog.parquet' from {repo_id}")
        else:
            try:
                api.delete_folder(
                    path_in_repo="newspapers",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Wipe all newspaper parquet shards for fresh regeneration",
                )
                print("  ✓ Deleted remote 'newspapers/' directory.")
            except Exception as e:
                print(f"  Notice: Could not delete remote 'newspapers/' (may already be empty): {e}")

            try:
                api.delete_file(
                    path_in_repo="catalog.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Remove stale catalog index",
                )
                print("  ✓ Deleted remote 'catalog.parquet'")
            except Exception:
                pass

    # Reset SQLite database
    if dry_run:
        print("  [DRY-RUN] Would reset all batches in SQLite to 'pending'")
    else:
        with db._get_connection() as conn:
            conn.execute(
                """
                UPDATE pipeline_batches
                SET status = 'pending',
                    pages_extracted = 0,
                    output_files = '[]',
                    started_at = NULL,
                    completed_at = NULL,
                    error_message = NULL,
                    attempts = 0
                """
            )
            conn.commit()
        print("  ✓ Reset ALL pipeline batches in SQLite to 'pending'.")


def main():
    parser = argparse.ArgumentParser(description="Clean up Hugging Face dataset files and reset SQLite state")
    parser.add_argument("--repo", default=os.environ.get("HF_REPO", "Tim-Pinecone/LOC-Chronicling-America"), help="Hugging Face repo ID")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face API token")
    parser.add_argument("--state", help="Specific state to wipe (e.g. 'alaska', 'AK', 'california', 'CA')")
    parser.add_argument("--all", action="store_true", help="Wipe all newspaper shards from Hugging Face")
    parser.add_argument("--recreate-repo", action="store_true", help="Delete and re-create the entire Hugging Face repo for a zero-history clean slate")
    parser.add_argument("--reset-db-only", action="store_true", help="Only reset SQLite pipeline queue without touching Hugging Face")
    parser.add_argument("--output-dir", default="./export_data", help="Local export directory to purge")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without deleting anything")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.reset_db_only and not args.dry_run:
        raise ValueError("HF_TOKEN is required via --token, environment variable, or .env file.")

    api = HfApi(token=token)
    db = CatalogDB()

    print("=" * 80)
    print("Chronicling America Hugging Face & State Cleanup Utility")
    print(f"Target Repo : https://huggingface.co/datasets/{args.repo}")
    print(f"Mode        : {'DRY RUN' if args.dry_run else 'LIVE EXECUTION'}")
    print("=" * 80)

    # 1. Clean local export directory if requested
    out_dir = Path(args.output_dir)
    if out_dir.exists() and (args.all or args.recreate_repo):
        newspapers_local = out_dir / "newspapers"
        if newspapers_local.exists() and not args.dry_run:
            shutil.rmtree(newspapers_local)
            print(f"✓ Removed local directory {newspapers_local}")
        scratch_local = out_dir / "scratch"
        if scratch_local.exists() and not args.dry_run:
            shutil.rmtree(scratch_local)
            print(f"✓ Removed local scratch {scratch_local}")

    # 2. Reset database only
    if args.reset_db_only:
        with db._get_connection() as conn:
            if args.state:
                conn.execute(
                    "UPDATE pipeline_batches SET status = 'pending', attempts = 0 WHERE batch_name IN (SELECT name FROM batches WHERE UPPER(state) = UPPER(?))",
                    (args.state,),
                )
                print(f"✓ Reset SQLite batches for state '{args.state}' to 'pending'.")
            else:
                conn.execute("UPDATE pipeline_batches SET status = 'pending', attempts = 0")
                print("✓ Reset ALL SQLite batches to 'pending'.")
        return

    # 3. Handle state-level or full cleanup
    if args.state:
        clean_state(api, args.repo, args.state, db, dry_run=args.dry_run)
    elif args.all or args.recreate_repo:
        clean_all(api, args.repo, db, recreate_repo=args.recreate_repo, dry_run=args.dry_run)
    else:
        print("Please specify --state <state>, --all, or --recreate-repo.")
        parser.print_help()
        return

    print("\n" + "=" * 80)
    print("Summary of Pipeline Batches in SQLite:")
    print(db.get_pipeline_summary())
    print("=" * 80)


if __name__ == "__main__":
    main()
