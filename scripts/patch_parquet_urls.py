#!/usr/bin/env python3
"""Batch update Parquet files in-place with precomputed direct image, PDF, and item URLs.

Avoids re-downloading or re-parsing gigabytes of raw Library of Congress OCR archives.
Reads existing Parquet files, resolves direct batch asset URLs via cached NDNP manifests,
and updates 'image_url', 'pdf_url', 'loc_item_url', and 'reel_id' using zero-copy PyArrow.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import disable_progress_bars
from loc_chronicling_america.batch import Batch

disable_progress_bars()

# Load .env if present (check cwd, script parent, and standard app dir)
env_candidates = [
    Path(".env"),
    Path(__file__).resolve().parent.parent / ".env",
    Path("/home/ubuntu/loc-chronicling-america/.env"),
]
for env_file in env_candidates:
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break


class ParquetUrlPatcher:
    """Precomputes batch manifest lookups and updates Parquet URL columns in-place."""

    def __init__(self, repo_id: str = "Tim-Pinecone/LOC-Chronicling-America", token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            print("WARNING: HF_TOKEN is not set! Writes and commits to Hugging Face will fail.", flush=True)
        self.api = HfApi(token=self.token)
        self.manifest_cache: Dict[str, Dict[Tuple[str, str, int, int], Dict[str, Any]]] = {}

    def get_manifest_mapping(self, batch_name: str) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
        """Fetch and cache asset mappings from the batch's manifest file."""
        if batch_name in self.manifest_cache:
            return self.manifest_cache[batch_name]

        b = Batch(batch_name)
        urls = b.get_manifest_asset_urls()
        self.manifest_cache[batch_name] = urls
        return urls

    def patch_table(self, table: pa.Table) -> Tuple[pa.Table, bool]:
        """Patch image_url, pdf_url, loc_item_url, and reel_id in a PyArrow Table.

        Returns:
            (updated_table, modified_bool)
        """
        if len(table) == 0:
            return table, False

        batch_name = table["source_batch"][0].as_py() if "source_batch" in table.column_names else None
        if not batch_name:
            return table, False

        urls = self.get_manifest_mapping(batch_name)
        n = len(table)

        new_img: List[str] = []
        new_pdf: List[Optional[str]] = []
        new_item: List[str] = []
        new_reel: List[str] = []
        modified = False

        col_names = table.column_names
        curr_imgs = table["image_url"].to_pylist() if "image_url" in col_names else [None] * n
        curr_pdfs = table["pdf_url"].to_pylist() if "pdf_url" in col_names else [None] * n
        curr_items = table["loc_item_url"].to_pylist() if "loc_item_url" in col_names else [None] * n
        curr_reels = table["reel_id"].to_pylist() if "reel_id" in col_names else [""] * n

        lccns = table["lccn"].to_pylist()
        dates = table["date"].to_pylist()
        editions = table["edition"].to_pylist()
        sequences = table["sequence"].to_pylist()

        for i in range(n):
            lccn = str(lccns[i])
            date_val = str(dates[i])
            try:
                ed = int(editions[i])
            except (ValueError, TypeError):
                ed = 1
            try:
                seq = int(sequences[i])
            except (ValueError, TypeError):
                seq = 1

            asset = urls.get((lccn, date_val, ed, seq), {})
            target_img = asset.get("image_url") or curr_imgs[i]
            target_pdf = asset.get("pdf_url") or curr_pdfs[i]
            target_reel = asset.get("reel_id") or curr_reels[i] or ""
            target_item = f"https://www.loc.gov/item/{lccn}/{date_val}/ed-{ed}/"

            if (
                target_img != curr_imgs[i]
                or target_pdf != curr_pdfs[i]
                or target_item != curr_items[i]
                or target_reel != curr_reels[i]
            ):
                modified = True

            new_img.append(target_img)
            new_pdf.append(target_pdf)
            new_item.append(target_item)
            new_reel.append(target_reel)

        if not modified:
            return table, False

        # Zero-copy replace the updated columns
        new_table = table
        col_idx = {name: idx for idx, name in enumerate(new_table.column_names)}

        if "image_url" in col_idx:
            new_table = new_table.set_column(col_idx["image_url"], "image_url", pa.array(new_img, pa.string()))
        else:
            new_table = new_table.append_column("image_url", pa.array(new_img, pa.string()))

        col_idx = {name: idx for idx, name in enumerate(new_table.column_names)}
        if "pdf_url" in col_idx:
            new_table = new_table.set_column(col_idx["pdf_url"], "pdf_url", pa.array(new_pdf, pa.string()))
        else:
            new_table = new_table.append_column("pdf_url", pa.array(new_pdf, pa.string()))

        col_idx = {name: idx for idx, name in enumerate(new_table.column_names)}
        if "loc_item_url" in col_idx:
            new_table = new_table.set_column(col_idx["loc_item_url"], "loc_item_url", pa.array(new_item, pa.string()))
        else:
            new_table = new_table.append_column("loc_item_url", pa.array(new_item, pa.string()))

        col_idx = {name: idx for idx, name in enumerate(new_table.column_names)}
        if "reel_id" in col_idx:
            new_table = new_table.set_column(col_idx["reel_id"], "reel_id", pa.array(new_reel, pa.string()))
        else:
            new_table = new_table.append_column("reel_id", pa.array(new_reel, pa.string()))

        return new_table, True

    def patch_local_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Patch a single local Parquet file in-place."""
        table = pq.read_table(file_path)
        new_table, modified = self.patch_table(table)
        if modified and not dry_run:
            pq.write_table(new_table, file_path, compression="snappy")
        return modified

    def patch_local_dir(self, dir_path: Path, dry_run: bool = False, workers: int = 8) -> int:
        """Recursively scan and patch local Parquet files."""
        files = list(dir_path.rglob("*.parquet"))
        print(f"Found {len(files)} local Parquet files in {dir_path}")
        if not files:
            return 0

        patched_count = 0
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.patch_local_file, f, dry_run): f for f in files}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    if fut.result():
                        patched_count += 1
                except Exception as e:
                    f = futures[fut]
                    print(f"Error patching {f}: {e}")

        elapsed = time.time() - t0
        print(f"✓ Patched {patched_count}/{len(files)} files in {elapsed:.2f}s ({elapsed/len(files)*1000:.1f}ms/file)")
        return patched_count

    def commit_with_retry(
        self,
        operations: List[CommitOperationAdd],
        commit_message: str,
        max_retries: int = 5,
    ) -> None:
        """Commit operations to Hugging Face with exponential backoff on Git conflicts."""
        for attempt in range(1, max_retries + 1):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_message,
                )
                return
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait_s = attempt * 3
                print(f"  Warning: Commit attempt {attempt} failed ({e}). Retrying in {wait_s}s...")
                time.sleep(wait_s)

    def list_hf_states(self) -> List[str]:
        """List all state folder names under newspapers/ on Hugging Face."""
        items = list(self.api.list_repo_tree(repo_id=self.repo_id, path_in_repo="newspapers", repo_type="dataset", recursive=False))
        states: List[str] = []
        for item in items:
            name = item.path.split("/")[-1]
            if name and not name.startswith("."):
                states.append(name)
        return sorted(states)

    def _process_remote_file(
        self,
        remote_path: str,
        tmp_path: Path,
        idx: int,
    ) -> Optional[Tuple[CommitOperationAdd, Path]]:
        try:
            cache_dir = tmp_path / "cache"
            local_dl = self.api.hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=remote_path,
                cache_dir=str(cache_dir),
            )
            table = pq.read_table(local_dl)
            new_table, modified = self.patch_table(table)
            if modified:
                out_file = tmp_path / f"patched_{idx}_{os.getpid()}.parquet"
                pq.write_table(new_table, out_file, compression="snappy")
                return (
                    CommitOperationAdd(
                        path_in_repo=remote_path,
                        path_or_fileobj=str(out_file),
                    ),
                    out_file,
                )
        except Exception as e:
            print(f"Failed processing {remote_path}: {e}")
        return None

    def patch_hf_state(
        self,
        state: str,
        batch_size: int = 250,
        workers: int = 8,
        dry_run: bool = False,
    ) -> None:
        """Patch Parquet files in a Hugging Face state partition in committed batches."""
        import shutil
        clean_st = state.lower().replace(" ", "_").replace("-", "_")
        prefix = f"newspapers/{clean_st}"
        print(f"\n--- Scanning Hugging Face partition: {prefix} ---", flush=True)

        all_entries = list(self.api.list_repo_tree(repo_id=self.repo_id, path_in_repo=prefix, repo_type="dataset", recursive=True))
        parquet_paths = [e.path for e in all_entries if e.path.endswith(".parquet")]
        print(f"Found {len(parquet_paths)} Parquet files in {prefix}", flush=True)
        if not parquet_paths:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            total_files = len(parquet_paths)
            for chunk_start in range(0, total_files, batch_size):
                chunk = parquet_paths[chunk_start : chunk_start + batch_size]
                chunk_end = min(chunk_start + len(chunk), total_files)
                pending_operations: List[CommitOperationAdd] = []
                files_to_cleanup: List[Path] = []

                print(f"  [{chunk_end:,}/{total_files:,} ({chunk_end/total_files*100:.1f}%)] Downloading and inspecting {len(chunk)} files...", flush=True)

                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self._process_remote_file, path, tmp_path, chunk_start + i): path
                        for i, path in enumerate(chunk)
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        if res:
                            op, out_file = res
                            pending_operations.append(op)
                            files_to_cleanup.append(out_file)

                if pending_operations:
                    if dry_run:
                        print(f"    [DRY-RUN] Would commit {len(pending_operations)} files to Hugging Face...", flush=True)
                    else:
                        print(f"    Committing {len(pending_operations)} patched files to {self.repo_id}...", flush=True)
                        self.commit_with_retry(
                            operations=pending_operations,
                            commit_message=f"Patch direct raw asset URLs for {clean_st} (chunk {chunk_end}/{total_files})",
                        )
                        print(f"    ✓ Committed {len(pending_operations)} files", flush=True)
                else:
                    print(f"    - All {len(chunk)} files in chunk already have correct URLs (skipped)", flush=True)

                # Clean up disk space
                for f in files_to_cleanup:
                    f.unlink(missing_ok=True)
                shutil.rmtree(tmp_path / "cache", ignore_errors=True)

        print(f"✓ Completed patching partition for {clean_st}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch update Parquet files with direct NDNP asset URLs.")
    parser.add_argument("--local-dir", type=str, help="Local directory containing Parquet files to patch")
    parser.add_argument("--hf-state", type=str, help="State name to patch directly on Hugging Face (e.g. 'alaska')")
    parser.add_argument("--all-states", action="store_true", help="Patch ALL states currently in Hugging Face")
    parser.add_argument("--repo-id", default="Tim-Pinecone/LOC-Chronicling-America", help="Hugging Face repo ID")
    parser.add_argument("--batch-size", type=int, default=250, help="Commit batch size for Hugging Face uploads")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for local patching")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without modifying files or uploading")

    args = parser.parse_args()

    patcher = ParquetUrlPatcher(repo_id=args.repo_id)

    if args.all_states or (args.hf_state and args.hf_state.lower() == "all"):
        states = patcher.list_hf_states()
        print(f"Discovered {len(states)} states on Hugging Face: {', '.join(states)}")
        for idx, st in enumerate(states, start=1):
            print(f"\n========================================================")
            print(f"Processing State [{idx}/{len(states)}]: {st}")
            print(f"========================================================")
            patcher.patch_hf_state(st, batch_size=args.batch_size, workers=args.workers, dry_run=args.dry_run)
    elif args.hf_state:
        patcher.patch_hf_state(args.hf_state, batch_size=args.batch_size, workers=args.workers, dry_run=args.dry_run)
    elif args.local_dir:
        patcher.patch_local_dir(Path(args.local_dir), dry_run=args.dry_run, workers=args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
