"""Streaming batch transmutation pipeline for Chronicling America.

Converts NDNP bulk .tar.bz2 archives into a researcher-friendly State/Newspaper/Issue
hierarchy of compressed Apache Parquet files without local disk bloat.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from .batch import Batch
from .db import CatalogDB
from .downloader import Downloader, get_default_downloader
from .hf import HuggingFaceDatasetManager
from .models import BatchInfo, TitleInfo


def clean_newspaper_title(raw_title: str) -> str:
    """Extract clean title by removing parenthetical location and date ranges."""
    clean = re.sub(r"\s*\([^)]*\).*", "", raw_title).strip()
    clean = re.sub(r"\s*\d{4}[-–][\d\?]{4}$", "", clean).strip()
    clean = re.sub(r"\s*\d{4}$", "", clean).strip()
    return clean or raw_title


def clean_slug(text: str) -> str:
    """Create a clean, underscore-separated slug from a string."""
    clean = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "_", clean)


def slugify(text: str) -> str:
    """Create a clean, URL-safe slug with hyphens from a string."""
    clean = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", clean)


PARQUET_PAGE_SCHEMA = pa.schema([
    ("_id", pa.string()),
    ("text", pa.string()),
    ("title", pa.string()),
    ("newspaper_title", pa.string()),
    ("newspaper_slug", pa.string()),
    ("lccn", pa.string()),
    ("date", pa.string()),
    ("year", pa.int32()),
    ("month", pa.int32()),
    ("day", pa.int32()),
    ("edition", pa.string()),
    ("sequence", pa.int32()),
    ("city", pa.string()),
    ("state", pa.string()),
    ("ethnicity", pa.string()),
    ("char_count", pa.int32()),
    ("word_count", pa.int32()),
    ("loc_page_url", pa.string()),
    ("loc_item_url", pa.string()),
    ("pdf_url", pa.string()),
    ("image_url", pa.string()),
    ("source_batch", pa.string()),
    ("awardee", pa.string()),
    ("awardee_code", pa.string()),
    ("reel_id", pa.string()),
    ("ocr_engine", pa.string()),
    ("page_width", pa.int32()),
    ("page_height", pa.int32()),
    ("page_unit", pa.string()),
    ("words", pa.list_(pa.string())),
    ("boxes", pa.list_(pa.list_(pa.int16()))),
    ("word_confidences", pa.list_(pa.float32())),
    (
        "lines",
        pa.list_(
            pa.struct([
                ("box", pa.list_(pa.int16())),
                ("text", pa.string()),
            ])
        ),
    ),
    (
        "blocks",
        pa.list_(
            pa.struct([
                ("block_id", pa.string()),
                ("box", pa.list_(pa.int16())),
                ("text", pa.string()),
            ])
        ),
    ),
])

AWARDEE_NAMES: Dict[str, str] = {
    "vi": "Library of Virginia",
    "nbu": "University of Nebraska-Lincoln",
    "iune": "University of Illinois at Urbana-Champaign",
    "curiv": "University of California, Riverside",
    "nn": "The New York Public Library",
    "pst": "Penn State University",
    "njr": "Rutgers University",
    "dlc": "Library of Congress",
    "whi": "Wisconsin Historical Society",
    "ohi": "Ohio History Connection",
    "mnhi": "Minnesota Historical Society",
    "khi": "Kansas Historical Society",
    "txdn": "University of North Texas",
    "fu": "University of Florida",
    "gu": "University of Georgia",
    "az": "Arizona State Library",
    "oru": "University of Oregon",
    "wa": "Washington State Library",
    "cohi": "History Colorado",
    "iahi": "State Historical Society of Iowa",
    "mdu": "University of Maryland",
    "kyu": "University of Kentucky",
    "lu": "Louisiana State University",
    "ncu": "University of North Carolina at Chapel Hill",
    "scu": "University of South Carolina",
    "tu": "University of Tennessee",
    "uuml": "University of Utah",
    "vtu": "University of Vermont",
    "wvu": "West Virginia University",
    "wyu": "University of Wyoming",
}


class BatchPipeline:
    """Orchestrates streaming download, parsing, and Parquet export across batches."""

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
        fallback = TitleInfo(
            lccn=lccn,
            name=clean_name,
            state=batch_awardee.upper() if batch_awardee and len(batch_awardee) == 2 else "Unknown",
            city="Unknown",
        )
        self._title_cache[lccn] = fallback
        return fallback

    def is_batch_already_uploaded(self, batch_info: BatchInfo, batch_obj: Batch) -> bool:
        """Check if a batch's parquet files are already present in Hugging Face."""
        if not self.hf_manager:
            return False

        try:
            issues = batch_obj.get_issue_refs()
            if not issues:
                return False

            # Check up to 3 sample issues (start, middle, end) to verify batch presence
            sample_indices = [0]
            if len(issues) > 1:
                sample_indices.append(len(issues) // 2)
            if len(issues) > 2:
                sample_indices.append(len(issues) - 1)

            for idx in sample_indices:
                issue = issues[idx]
                title = self._get_title_info(issue.lccn, batch_awardee=batch_info.awardee)
                clean_name = clean_newspaper_title(title.name)
                newspaper_slug = clean_slug(clean_name)
                state_slug = clean_slug(title.state or batch_info.awardee or "unknown")

                parts = issue.issue_date.split("-")
                if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit() or not parts[2].isdigit():
                    return False
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                rel_path = f"newspapers/{state_slug}/{newspaper_slug}/{y:04d}/{state_slug}_{newspaper_slug}_{y:04d}_{m:02d}_{d:02d}.parquet"

                if not self.hf_manager.file_exists(rel_path):
                    return False

            return True
        except Exception:
            return False

    def process_batch(
        self,
        batch_info: BatchInfo,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[int, List[str]]:
        """Process a single batch through the streaming pipeline.

        1. Checks if batch is already uploaded to Hugging Face (skips if present).
        2. Marks batch as processing in SQLite.
        3. Downloads the .tar.bz2 bulk archive to scratch space.
        4. Iterates through all ocr.txt and ALTO XML members with multi-core parsing.
        5. Writes pages to state/newspaper/issue .parquet files.
        6. Deletes the .tar.bz2 to reclaim disk space.
        7. Uploads to Hugging Face if configured.
        8. Marks batch as completed in SQLite.

        Returns:
            Tuple of (pages_extracted_count, list_of_relative_output_files).
        """
        batch_name = batch_info.name
        batch_obj = Batch(name=batch_name, info=batch_info, downloader=self.downloader)

        # Step 0: Remote HF pre-check to prevent duplicate downloads and processing
        if self.is_batch_already_uploaded(batch_info, batch_obj):
            print(f"  ⚡ Batch '{batch_name}' already exists on Hugging Face ({self.hf_manager.repo_id}). Skipping.", flush=True)
            self.db.mark_pipeline_batch_completed(batch_name, pages_extracted=batch_info.page_count or 0, output_files=[])
            return (batch_info.page_count or 0, [])

        self.db.mark_pipeline_batch_processing(batch_name)
        tar_path = self.scratch_dir / f"{batch_name}.tar.bz2"

        try:
            # Step 1: Download bulk tarball
            if not tar_path.exists():
                batch_obj.download(dest_dir=self.scratch_dir, show_progress=False)

            # Step 2: Buffer pages in memory grouped by target file path
            file_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            pages_count = 0

            awardee = batch_info.awardee
            awardee_code = awardee or "unknown"
            awardee_full = AWARDEE_NAMES.get(awardee_code.lower(), awardee_code)

            for item in batch_obj.iter_archive(archive_path=tar_path, extract_xml=True):
                if not item.text:
                    continue

                title = self._get_title_info(item.lccn, batch_awardee=awardee)
                clean_name = clean_newspaper_title(title.name)
                newspaper_slug = clean_slug(clean_name)
                state_slug = clean_slug(title.state or awardee or "unknown")

                year_int = int(item.year) if str(item.year).isdigit() else 0
                month_int = int(item.month) if str(item.month).isdigit() else 0
                day_int = int(item.day) if str(item.day).isdigit() else 0

                # Issue-level Parquet: newspapers/{state}/{newspaper_slug}/{year}/{state}_{newspaper_slug}_{year}_{month}_{day}.parquet
                filename = f"{state_slug}_{newspaper_slug}_{year_int:04d}_{month_int:02d}_{day_int:02d}.parquet"
                rel_path = f"newspapers/{state_slug}/{newspaper_slug}/{year_int:04d}/{filename}"

                doc_id = f"{item.lccn}_{item.date}_ed-{item.edition}_seq-{item.sequence}"

                # Real working LoC Page Viewer URL and Direct Asset URLs
                loc_page_url = f"https://www.loc.gov/resource/{item.lccn}/{item.date}/ed-{item.edition}/?sp={item.sequence}"
                pdf_url = f"https://chroniclingamerica.loc.gov/lccn/{item.lccn}/{item.date}/ed-{item.edition}/seq-{item.sequence}.pdf"
                image_url = f"https://chroniclingamerica.loc.gov/lccn/{item.lccn}/{item.date}/ed-{item.edition}/seq-{item.sequence}.jp2"

                layout = item.layout_data or {}

                doc = {
                    "_id": doc_id,
                    "text": item.text.strip(),
                    "title": f"{clean_name}, {item.date} - Page {item.sequence}",
                    "newspaper_title": clean_name,
                    "newspaper_slug": newspaper_slug,
                    "lccn": item.lccn,
                    "date": item.date,
                    "year": year_int,
                    "month": month_int,
                    "day": day_int,
                    "edition": str(item.edition),
                    "sequence": int(item.sequence) if str(item.sequence).isdigit() else 1,
                    "city": title.city or "Unknown",
                    "state": title.state or "Unknown",
                    "ethnicity": title.ethnicity or None,
                    "char_count": len(item.text),
                    "word_count": len(item.text.split()),
                    "loc_page_url": loc_page_url,
                    "loc_item_url": loc_page_url,
                    "pdf_url": pdf_url,
                    "image_url": image_url,
                    "source_batch": batch_name,
                    "awardee": awardee_full,
                    "awardee_code": awardee_code,
                    "reel_id": item.reel_id or "",
                    "ocr_engine": layout.get("ocr_engine") or "Unknown",
                    "page_width": layout.get("page_width", 0),
                    "page_height": layout.get("page_height", 0),
                    "page_unit": layout.get("page_unit", "inch1200"),
                    "words": layout.get("words", []),
                    "boxes": layout.get("boxes", []),
                    "word_confidences": layout.get("word_confidences", []),
                    "lines": layout.get("lines", []),
                    "blocks": layout.get("blocks", []),
                }

                file_buffers[rel_path].append(doc)
                pages_count += 1

                if progress_callback and pages_count % 500 == 0:
                    progress_callback(batch_name, pages_count, 0)

            # Step 3: Write buffered documents to compressed Parquet files
            written_files: List[str] = []
            for rel_path, docs in file_buffers.items():
                dest_file = self.output_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                new_table = pa.Table.from_pylist(docs, schema=PARQUET_PAGE_SCHEMA)

                if dest_file.exists():
                    try:
                        existing_table = pq.read_table(dest_file)
                        combined_table = pa.concat_tables([existing_table, new_table])
                        pq.write_table(combined_table, dest_file, compression="snappy")
                    except Exception:
                        pq.write_table(new_table, dest_file, compression="snappy")
                else:
                    pq.write_table(new_table, dest_file, compression="snappy")

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
        newspapers_dir = self.output_dir / "newspapers"
        if not newspapers_dir.exists():
            return self.output_dir / "catalog.jsonl", None

        # Gather metadata on all generated newspaper titles
        titles_summary: Dict[str, Dict[str, Any]] = {}

        for pq_file in sorted(newspapers_dir.glob("**/*.parquet")):
            rel_parts = pq_file.relative_to(newspapers_dir).parts
            if len(rel_parts) < 2:
                continue
            state_slug = rel_parts[0]
            title_slug = rel_parts[1]

            # Read first row to extract canonical metadata
            lccn = None
            title_name = None
            city = None
            state = None
            ethnicity = None
            num_rows = 0

            try:
                table = pq.read_table(pq_file, columns=["lccn", "newspaper_title", "city", "state", "ethnicity", "year"])
                num_rows = table.num_rows
                if num_rows > 0:
                    first_row = table.slice(0, 1).to_pylist()[0]
                    lccn = first_row.get("lccn")
                    title_name = first_row.get("newspaper_title")
                    city = first_row.get("city")
                    state = first_row.get("state")
                    ethnicity = first_row.get("ethnicity")
            except Exception:
                continue

            if not lccn:
                lccn = title_slug

            if lccn not in titles_summary:
                titles_summary[lccn] = {
                    "lccn": lccn,
                    "newspaper_title": title_name or title_slug,
                    "newspaper_slug": title_slug,
                    "state": state or state_slug,
                    "city": city or "Unknown",
                    "ethnicity": ethnicity,
                    "relative_path": f"newspapers/{state_slug}/{title_slug}/",
                    "file_count": 0,
                    "total_pages": 0,
                    "total_size_bytes": 0,
                    "years": set(),
                }

            summary = titles_summary[lccn]
            summary["file_count"] += 1
            summary["total_pages"] += num_rows
            summary["total_size_bytes"] += pq_file.stat().st_size
            year_match = re.search(r"_(\d{4})(?:_\d{2}_\d{2})?\.parquet$", pq_file.name)
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
                    "newspaper_slug": s["newspaper_slug"],
                    "state": s["state"],
                    "city": s["city"],
                    "ethnicity": s["ethnicity"],
                    "relative_path": s["relative_path"],
                    "file_count": s["file_count"],
                    "total_pages": s["total_pages"],
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
            finally:
                time.sleep(2.0)

        # Update master index
        self.update_catalog_index()

        # Update Hugging Face README and catalog if configured
        if self.hf_manager:
            readme_path = self.output_dir / "README.md"
            states_set = set()
            newspapers_dir = self.output_dir / "newspapers"
            if newspapers_dir.exists():
                states_set.update(d.name for d in newspapers_dir.iterdir() if d.is_dir())

            # Also check remote repository so purged local directories are retained
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.hf_manager.token)
                repo_files = api.list_repo_files(repo_id=self.hf_manager.repo_id, repo_type="dataset")
                for f in repo_files:
                    if f.startswith("newspapers/"):
                        p = f.split("/")
                        if len(p) > 2:
                            states_set.add(p[1])
            except Exception:
                pass

            states = sorted(list(states_set))
            self.hf_manager.generate_readme(readme_path, states=states)
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
