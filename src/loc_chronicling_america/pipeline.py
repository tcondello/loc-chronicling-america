"""Streaming batch transmutation pipeline for Chronicling America.

Converts NDNP bulk .tar.bz2 archives into a researcher-friendly State/Newspaper/Year
hierarchy of compressed Pinecone-compatible JSONL (.jsonl.gz) files without local disk bloat.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .batch import Batch
from .db import CatalogDB
from .downloader import Downloader, get_default_downloader
from .hf import HuggingFaceDatasetManager
from .models import BatchInfo, TitleInfo


def slugify(text: str) -> str:
    """Create a clean, URL-safe slug from a string."""
    clean = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", clean)


class BatchPipeline:
    """Orchestrates streaming download, parsing, and JSONL export across batches."""

    def __init__(
        self,
        output_dir: Path | str = "./export_data",
        scratch_dir: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
        downloader: Optional[Downloader] = None,
        hf_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        keep_tar: bool = False,
        purge_local_after_upload: bool = False,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.scratch_dir = Path(scratch_dir or (self.output_dir / "scratch")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self.db = CatalogDB(db_path=db_path)
        self.downloader = downloader or get_default_downloader()
        self.keep_tar = keep_tar
        self.purge_local_after_upload = purge_local_after_upload

        self.hf_manager: Optional[HuggingFaceDatasetManager] = None
        if hf_repo:
            self.hf_manager = HuggingFaceDatasetManager(repo_id=hf_repo, token=hf_token)
            self.hf_manager.ensure_repo_exists()

        self._title_cache: Dict[str, TitleInfo] = {}

    def _get_title_info(self, lccn: str, batch_awardee: Optional[str] = None) -> TitleInfo:
        """Resolve title metadata for an LCCN using SQLite cache or fallback."""
        if lccn in self._title_cache:
            return self._title_cache[lccn]

        title = self.db.get_title(lccn)
        if title:
            self._title_cache[lccn] = title
            return title

        # Try fetching from live LoC item endpoint if not in local SQLite
        try:
            url = f"https://www.loc.gov/item/{lccn}/?fo=json"
            data = self.downloader.fetch_json(url)
            item = data.get("item", {})
            raw_title = item.get("title") or item.get("newspaper_title") or f"Newspaper {lccn}"
            title_name = raw_title[0] if isinstance(raw_title, list) and raw_title else str(raw_title)

            loc_state = item.get("location_state")
            st_val = None
            if isinstance(loc_state, list) and loc_state:
                st_val = str(loc_state[0])
            elif isinstance(loc_state, dict):
                st_val = loc_state.get("label") or loc_state.get("value")
            elif loc_state:
                st_val = str(loc_state)

            loc_city = item.get("location_city")
            ct_val = loc_city[0] if isinstance(loc_city, list) and loc_city else str(loc_city or "Unknown")

            eth = item.get("subject_ethnicity")
            ethnicity_str = eth.get("label") if isinstance(eth, dict) else (str(eth[0]) if isinstance(eth, list) and eth else (str(eth) if eth else None))

            fetched_title = TitleInfo(
                lccn=lccn,
                name=title_name,
                state=st_val or (batch_awardee.upper() if batch_awardee and len(batch_awardee) == 2 else "Unknown"),
                city=ct_val,
                ethnicity=ethnicity_str,
            )
            self.db.upsert_title(fetched_title)
            self._title_cache[lccn] = fetched_title
            return fetched_title
        except Exception:
            pass

        # Fallback based on awardee/LCCN
        clean_name = f"Newspaper {lccn}"
        state_code = batch_awardee.upper() if batch_awardee and len(batch_awardee) == 2 else None
        state_name = state_code or "Unknown"

        fallback = TitleInfo(
            lccn=lccn,
            name=clean_name,
            state=state_name,
            city="Unknown",
        )
        self._title_cache[lccn] = fallback
        return fallback


    def process_batch(
        self,
        batch_info: BatchInfo,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[int, List[str]]:
        """Process a single batch through the streaming pipeline.

        1. Marks batch as processing in SQLite.
        2. Downloads the .tar.bz2 bulk archive to scratch space.
        3. Iterates through all ocr.txt members.
        4. Writes pages to state/newspaper/year .jsonl.gz files.
        5. Deletes the .tar.bz2 to reclaim disk space.
        6. Uploads to Hugging Face if configured.
        7. Marks batch as completed in SQLite.

        Returns:
            Tuple of (pages_extracted_count, list_of_relative_output_files).
        """
        batch_name = batch_info.name
        self.db.mark_pipeline_batch_processing(batch_name)

        batch_obj = Batch(name=batch_name, info=batch_info, downloader=self.downloader)
        tar_path = self.scratch_dir / f"{batch_name}.tar.bz2"

        try:
            # Step 1: Download bulk tarball
            if not tar_path.exists():
                batch_obj.download(dest_dir=self.scratch_dir, show_progress=False)

            # Step 2: Buffer pages in memory grouped by target file path
            # target_rel_path -> list of document dicts
            file_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            pages_count = 0

            awardee = batch_info.awardee

            for item in batch_obj.iter_archive(archive_path=tar_path, extract_xml=False):
                if not item.text:
                    continue

                title = self._get_title_info(item.lccn, batch_awardee=awardee)
                state_slug = slugify(title.state or awardee or "unknown")
                city_slug = slugify(title.city or "unknown")
                title_slug = slugify(title.name)

                folder_name = f"{title_slug}_{city_slug}_{item.lccn}"
                year_str = item.year
                rel_path = f"data/{state_slug}/{folder_name}/{year_str}.jsonl.gz"

                doc_id = f"{item.lccn}_{item.date}_ed-{item.edition}_seq-{item.sequence}"

                # Pinecone Document Schema format
                doc = {
                    "_id": doc_id,
                    "text": item.text.strip(),
                    "title": f"{title.name}, {item.date} - Page {item.sequence}",
                    "newspaper_title": title.name,
                    "newspaper_slug": title_slug,
                    "lccn": item.lccn,
                    "date": item.date,
                    "year": int(item.year) if item.year.isdigit() else 0,
                    "month": int(item.month) if item.month.isdigit() else 0,
                    "day": int(item.day) if item.day.isdigit() else 0,
                    "edition": item.edition,
                    "sequence": item.sequence,
                    "city": title.city or "Unknown",
                    "state": title.state or "Unknown",
                    "char_count": len(item.text),
                    "word_count": len(item.text.split()),
                    "loc_item_url": item.loc_url,
                    "source_batch": batch_name,
                }
                if title.ethnicity:
                    doc["ethnicity"] = title.ethnicity

                file_buffers[rel_path].append(doc)
                pages_count += 1

                if progress_callback and pages_count % 500 == 0:
                    progress_callback(batch_name, pages_count, 0)

            # Step 3: Write buffered documents to compressed .jsonl.gz files
            written_files: List[str] = []
            for rel_path, docs in file_buffers.items():
                dest_file = self.output_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Append gzip-compressed JSON lines
                with gzip.open(dest_file, "at", encoding="utf-8") as gz_out:
                    for d in docs:
                        gz_out.write(json.dumps(d, ensure_ascii=False) + "\n")

                written_files.append(rel_path)

            # Step 4: Delete raw .tar.bz2 to preserve scratch disk space
            if not self.keep_tar and tar_path.exists():
                tar_path.unlink(missing_ok=True)

            # Step 5: Upload to Hugging Face if configured
            if self.hf_manager and written_files:
                batch_uploads = [(self.output_dir / rel_path, rel_path) for rel_path in written_files]
                self.hf_manager.upload_files_atomic(
                    local_files=batch_uploads,
                    commit_message=f"Add {pages_count} pages from batch {batch_name}",
                )
                if self.purge_local_after_upload:
                    for rel_path in written_files:
                        (self.output_dir / rel_path).unlink(missing_ok=True)

            # Step 6: Mark completed in SQLite
            self.db.mark_pipeline_batch_completed(batch_name, pages_count, written_files)
            return pages_count, written_files

        except Exception as e:
            # Clean up on failure
            if not self.keep_tar and tar_path.exists():
                tar_path.unlink(missing_ok=True)
            self.db.mark_pipeline_batch_failed(batch_name, str(e))
            raise

    def update_catalog_index(self) -> Tuple[Path, Optional[Path]]:
        """Scan generated output files to construct a master catalog.parquet and catalog.jsonl."""
        data_dir = self.output_dir / "data"
        if not data_dir.exists():
            return self.output_dir / "catalog.jsonl", None

        # Gather metadata on all generated newspaper titles
        titles_summary: Dict[str, Dict[str, Any]] = {}

        for gz_file in data_dir.glob("*/*/*.jsonl.gz"):
            # Path: data/{state}/{folder}/{year}.jsonl.gz
            state_slug = gz_file.parent.parent.name
            folder_name = gz_file.parent.name
            parts = folder_name.split("_")
            lccn = parts[-1] if len(parts) >= 2 else folder_name

            title_info = self._get_title_info(lccn)

            if lccn not in titles_summary:
                titles_summary[lccn] = {
                    "lccn": lccn,
                    "newspaper_title": title_info.name,
                    "state": title_info.state,
                    "city": title_info.city,
                    "ethnicity": title_info.ethnicity,
                    "relative_path": f"data/{state_slug}/{folder_name}/",
                    "file_count": 0,
                    "total_size_bytes": 0,
                    "years": set(),
                }

            summary = titles_summary[lccn]
            summary["file_count"] += 1
            summary["total_size_bytes"] += gz_file.stat().st_size
            year_match = re.match(r"^(\d{4})\.jsonl\.gz$", gz_file.name)
            if year_match:
                summary["years"].add(int(year_match.group(1)))

        records = []
        for s in titles_summary.values():
            years_list = sorted(list(s["years"]))
            start_yr = years_list[0] if years_list else None
            end_yr = years_list[-1] if years_list else None
            records.append(
                {
                    "lccn": s["lccn"],
                    "newspaper_title": s["newspaper_title"],
                    "state": s["state"],
                    "city": s["city"],
                    "ethnicity": s["ethnicity"],
                    "relative_path": s["relative_path"],
                    "file_count": s["file_count"],
                    "total_size_mb": round(s["total_size_bytes"] / (1024 * 1024), 2),
                    "start_year": start_yr,
                    "end_year": end_yr,
                    "years_covered": f"{start_yr}-{end_yr}" if start_yr else "Unknown",
                }
            )

        jsonl_path = self.output_dir / "catalog.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        parquet_path = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(records)
            parquet_path = self.output_dir / "catalog.parquet"
            pq.write_table(table, parquet_path)
        except Exception:
            pass

        return jsonl_path, parquet_path

    def run(
        self,
        state: Optional[str] = None,
        limit_batches: Optional[int] = None,
        batch_name: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Execute the pipeline across queued batches.

        Args:
            state: Optional state filter (e.g. 'NE').
            limit_batches: Maximum number of batches to process in this run.
            batch_name: Optional specific batch identifier to run.
            resume: Whether to skip already completed batches.
        """
        batches_to_run: List[BatchInfo] = []

        if batch_name:
            b_info = self.db.get_batch(batch_name)
            if not b_info:
                b_info = BatchInfo(
                    name=batch_name,
                    archive_url=f"https://chroniclingamerica.loc.gov/data/ocr/{batch_name}.tar.bz2",
                    raw_batch_url=f"https://chroniclingamerica.loc.gov/data/batches/{batch_name}/",
                )
            batches_to_run.append(b_info)
        else:
            batches_to_run = self.db.get_pending_pipeline_batches(state=state, limit=limit_batches)

        total_batches = len(batches_to_run)
        total_pages = 0
        successful = 0
        failed = 0

        t0 = time.time()
        for idx, b_info in enumerate(batches_to_run, start=1):
            b_name = b_info.name
            size_mb = (b_info.size_bytes or 0) / (1024 * 1024)
            print(f"  [{idx}/{total_batches}] Batch '{b_name}' ({size_mb:.1f} MB) downloading & extracting...", flush=True)
            try:
                pages, files = self.process_batch(b_info)
                total_pages += pages
                successful += 1
                print(f"  [{idx}/{total_batches}] ✓ Batch '{b_name}' uploaded: {pages:,} pages across {len(files)} files", flush=True)
            except Exception as e:
                failed += 1
                print(f"  [{idx}/{total_batches}] ✗ Batch '{b_name}' failed: {e}", flush=True)
                continue

        # Update master index
        self.update_catalog_index()

        # Update Hugging Face README if configured
        if self.hf_manager:
            readme_path = self.output_dir / "README.md"
            self.hf_manager.generate_readme(readme_path)
            self.hf_manager.upload_file(readme_path, "README.md")
            if (self.output_dir / "catalog.parquet").exists():
                self.hf_manager.upload_file(self.output_dir / "catalog.parquet", "catalog.parquet")

        elapsed = time.time() - t0
        return {
            "batches_processed": len(batches_to_run),
            "successful_batches": successful,
            "failed_batches": failed,
            "pages_extracted": total_pages,
            "elapsed_seconds": round(elapsed, 2),
            "pipeline_summary": self.db.get_pipeline_summary(),
        }
