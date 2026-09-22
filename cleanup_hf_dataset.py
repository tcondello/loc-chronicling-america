#!/usr/bin/env python3
"""Clean up legacy Hugging Face dataset files, local export data, and SQLite pipeline queue.

Removes legacy data/ directory and old catalog files from the Hugging Face repo,
cleans local export directories, and resets the SQLite database state machine.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import CommitOperationDelete, HfApi
from loc_chronicling_america.db import CatalogDB

# Load .env if present
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

HF_REPO = "Tim-Pinecone/LOC-Chronicling-America"
HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_DIR = Path("./export_data")


def main():
    print("=" * 80)
    print("Chronicling America Dataset & Local Storage Cleanup")
    print(f"Target Repo: https://huggingface.co/datasets/{HF_REPO}")
    print("=" * 80)

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not found in environment or .env file!")

    api = HfApi(token=HF_TOKEN)

    # Step 1: Clean Hugging Face Repo
    print("\n1. Inspecting remote repository on Hugging Face...")
    remote_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    print(f"   Found {len(remote_files)} files in repository.")

    # We want to delete all files in data/ and legacy catalog files
    files_to_delete = [
        f for f in remote_files
        if f.startswith("data/") or f in ("catalog.parquet", "catalog.jsonl")
    ]

    if files_to_delete:
        print(f"   Deleting {len(files_to_delete)} legacy files in atomic commit...")
        # Hugging Face create_commit supports operations
        # For a large number of deletes, chunk them into batches of 100 to avoid request body limits
        chunk_size = 100
        for i in range(0, len(files_to_delete), chunk_size):
            chunk = files_to_delete[i:i + chunk_size]
            operations = [CommitOperationDelete(path_in_repo=f) for f in chunk]
            api.create_commit(
                repo_id=HF_REPO,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Clean up legacy dataset files ({i + 1} to {min(i + len(chunk), len(files_to_delete))})",
            )
            print(f"   ✓ Deleted chunk {i + 1}-{min(i + len(chunk), len(files_to_delete))} / {len(files_to_delete)}")
        print("   ✓ Remote cleanup complete!")
    else:
        print("   No legacy files found on Hugging Face.")

    # Step 2: Clean Local Export Directory
    print("\n2. Cleaning local export directory...")
    if OUTPUT_DIR.exists():
        legacy_data = OUTPUT_DIR / "data"
        if legacy_data.exists():
            shutil.rmtree(legacy_data)
            print(f"   ✓ Deleted local {legacy_data}")
        scratch_dir = OUTPUT_DIR / "scratch"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
            print(f"   ✓ Deleted local {scratch_dir}")
        for f in (OUTPUT_DIR / "catalog.parquet", OUTPUT_DIR / "catalog.jsonl"):
            if f.exists():
                f.unlink(missing_ok=True)
                print(f"   ✓ Deleted local {f.name}")

    # Step 3: Reset SQLite Pipeline Queue
    print("\n3. Resetting SQLite pipeline queue...")
    db = CatalogDB()
    with db._get_connection() as conn:
        conn.execute(
            """
            UPDATE pipeline_batches
            SET status = 'pending',
                pages_extracted = 0,
                output_files = '[]',
                started_at = NULL,
                completed_at = NULL,
                error_message = NULL
            """
        )
        conn.commit()
    print("   ✓ Reset all pipeline batches in SQLite to 'pending'.")
    print("   Current queue summary:", db.get_pipeline_summary())

    print("\n" + "=" * 80)
    print("Cleanup Complete! Ready for fresh Parquet pipeline execution.")
    print("=" * 80)


if __name__ == "__main__":
    main()
