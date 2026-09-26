#!/usr/bin/env python3
"""Streaming ingestion engine for Library of Congress Chronicling America.

Streams issue Parquet files directly from Hugging Face Hub (Tim-Pinecone/LOC-Chronicling-America),
chunks page text using Chonkie, generates 1,536-dim embeddings via OpenAI text-embedding-3-small,
and upserts records into Pinecone Serverless (herald-hybrid-fts) partitioned by year namespace.

Features:
- Zero local disk accumulation: downloaded parquet files are immediately purged after chunking.
- Exponential backoff retry on OpenAI / Pinecone rate limits.
- Atomic checkpointing per newspaper (.checkpoint_{slug}.json) for resume capability.
- Appends to local JSONL so local calendar manifest stays synchronized with newly ingested dates.
"""

import argparse
import concurrent.futures
import functools
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from chonkie import RecursiveChunker
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from openai import OpenAI
from pinecone import Pinecone

# Ensure current directory is on sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config
from iiif import jp2_to_iiif_url

load_dotenv()
load_dotenv(HERE / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description="Stream, chunk, embed, and ingest LoC broadsheets into Pinecone")
    parser.add_argument(
        "--newspaper",
        default="chicago_eagle",
        help="Newspaper slug to process, or 'all_5' for the 5 expanded publications",
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
        "--limit-issues",
        type=int,
        default=None,
        help="Optional max number of issues to process (useful for test runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and chunk issues, but skip OpenAI embedding and Pinecone upsert",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Reset completion checkpoint and start from first issue",
    )
    return parser.parse_args()


class CheckpointManager:
    """Tracks completed issue paths per newspaper to guarantee idempotency and easy resume."""

    def __init__(self, checkpoint_file: Path):
        self.file = checkpoint_file
        self.completed: Set[str] = set()
        self.load()

    def load(self):
        if self.file.exists():
            try:
                data = json.loads(self.file.read_text(encoding="utf-8"))
                self.completed = set(data.get("completed_issues", []))
            except Exception as e:
                print(f"Warning: Failed loading checkpoint {self.file}: {e}")
                self.completed = set()

    def save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.file.with_suffix(".tmp")
        temp_file.write_text(json.dumps({"completed_issues": sorted(list(self.completed))}, indent=2), encoding="utf-8")
        temp_file.replace(self.file)

    def is_done(self, issue_path: str) -> bool:
        return issue_path in self.completed

    def mark_done(self, issue_path: str):
        self.completed.add(issue_path)


def embed_batch_with_retry(oai: OpenAI, texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """Embed texts using OpenAI text-embedding-3-small with exponential backoff on rate limits."""
    cleaned = [t if t.strip() else " " for t in texts]
    all_embeddings: List[List[float]] = []
    sub_size = 100

    for i in range(0, len(cleaned), sub_size):
        slice_texts = cleaned[i : i + sub_size]
        delay = 1.5
        success = False
        for attempt in range(max_retries):
            try:
                resp = oai.embeddings.create(model=config.EMBED_MODEL, input=slice_texts)
                all_embeddings.extend([item.embedding for item in resp.data])
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    print(f"  [OpenAI 429 Rate Limit] Backing off {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2.0
                elif attempt == max_retries - 1:
                    print(f"  [OpenAI Error] Permanent failure after {max_retries} attempts: {e}")
                    raise e
                else:
                    print(f"  [OpenAI Warning] Retryable error ({e}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 1.5
        if not success:
            raise RuntimeError(f"Failed to generate embeddings for batch starting at {i}")

    return all_embeddings


def fetch_and_chunk_issue(
    fpath: str,
    token: Optional[str],
    prefix: str,
    np_slug: str,
    np_conf: Dict[str, Any],
    chunker: RecursiveChunker,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Download a single issue Parquet file, chunk pages with Chonkie, and purge disk files."""
    local_parquet = None
    try:
        local_parquet = hf_hub_download(
            repo_id=config.HF_REPO,
            filename=fpath,
            repo_type="dataset",
            token=token,
        )
        table = pq.read_table(local_parquet)
        rows = table.to_pylist()
    except Exception as e:
        print(f"  ⚠ Failed downloading/reading {fpath}: {e}")
        return fpath, None
    finally:
        # Purge downloaded file and underlying HF cache blob immediately
        if local_parquet:
            try:
                real_p = os.path.realpath(local_parquet)
                if os.path.islink(local_parquet):
                    os.unlink(local_parquet)
                elif os.path.exists(local_parquet):
                    os.unlink(local_parquet)
                if os.path.exists(real_p):
                    os.unlink(real_p)
            except Exception:
                pass

    seen_page_keys: Set[Tuple[str, str, int]] = set()
    issue_docs: List[Dict[str, Any]] = []
    for row in rows:
        raw_text = (row.get("text") or "").strip()
        if not raw_text or len(raw_text) < 30:
            continue

        date_str = str(row.get("date") or "1910-01-01")
        seq = int(row.get("sequence") or 1)
        ed = str(row.get("edition") or "1")

        page_key = (date_str, ed, seq)
        if page_key in seen_page_keys:
            continue
        seen_page_keys.add(page_key)

        month_int = int(row.get("month") or 1)
        day_int = int(row.get("day") or 1)
        year_int = int(row.get("year") or 1910)

        season = config.SEASON_MAP.get(month_int, "Unknown")
        raw_jp2 = row.get("image_url") or ""
        iiif_thumb = jp2_to_iiif_url(raw_jp2, size="600,") if raw_jp2 else ""

        parent_page_id = f"{prefix}#{date_str}#ed-{ed}#p{seq:02d}"
        page_chunks = chunker(raw_text)
        chunk_count = len(page_chunks)

        for c_idx, c in enumerate(page_chunks):
            chunk_text = c.text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{parent_page_id}#c{c_idx:02d}"
            doc = {
                "_id": chunk_id,
                "text": chunk_text,
                "parent_page_id": parent_page_id,
                "newspaper_title": row.get("newspaper_title") or np_conf["title"],
                "newspaper_slug": np_slug,
                "region": np_conf["region"],
                "city": row.get("city") or np_conf["city"],
                "state": row.get("state") or np_conf["state_name"],
                "lccn": row.get("lccn") or np_conf["lccn"],
                "date": date_str,
                "year": year_int,
                "month": month_int,
                "day": day_int,
                "season": season,
                "sequence": seq,
                "edition": ed,
                "chunk_index": c_idx,
                "chunk_count": chunk_count,
                "iiif_thumb_url": iiif_thumb,
                "image_url": raw_jp2,
                "file_path": raw_jp2,
                "loc_page_url": row.get("loc_page_url") or "",
                "pdf_url": row.get("pdf_url") or "",
            }
            issue_docs.append(doc)

    return fpath, issue_docs


def process_newspaper(
    np_slug: str,
    years: List[str],
    batch_size: int,
    limit_issues: Optional[int],
    dry_run: bool,
    reset_checkpoint: bool,
    oai: Optional[OpenAI],
    index: Optional[Any],
    api: HfApi,
):
    if np_slug not in config.NEWSPAPERS:
        print(f"Error: Unknown newspaper slug '{np_slug}'")
        return

    np_conf = config.NEWSPAPERS[np_slug]
    prefix = np_conf["prefix"]
    path_prefix = np_conf["path_prefix"]
    token = os.environ.get("HF_TOKEN")

    chk_file = HERE / "data" / f".checkpoint_{np_slug}.json"
    if reset_checkpoint and chk_file.exists():
        chk_file.unlink()

    checkpoint = CheckpointManager(chk_file)

    jsonl_output = HERE / "data" / f"chunks_{np_slug}.jsonl"
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📰 Ingestion Stream: {np_conf['title']} ({np_conf['region']})")
    print(f"  Prefix Path  : {path_prefix}")
    print(f"  Target Years : {years}")
    print(f"  Checkpoint   : {chk_file} ({len(checkpoint.completed)} previously completed)")
    print(f"  Dry-Run Mode : {dry_run}")
    print("=" * 80)

    # 1. Discover all parquet files for target years in HF repo
    all_issue_files: List[str] = []
    print(f"Discovering Parquet files in {config.HF_REPO}/{path_prefix}...")

    for yr in sorted(years):
        yr_path = f"{path_prefix}/{yr}"
        try:
            tree = list(api.list_repo_tree(repo_id=config.HF_REPO, path_in_repo=yr_path, repo_type="dataset"))
            parquets = sorted([item.path for item in tree if item.path.endswith(".parquet")])
            all_issue_files.extend(parquets)
            print(f"  Year {yr}: {len(parquets)} issues found")
        except Exception as e:
            print(f"  Year {yr}: None found or error ({e})")

    # Filter out already completed issues
    pending_files = [f for f in all_issue_files if not checkpoint.is_done(f)]
    if limit_issues:
        pending_files = pending_files[:limit_issues]

    total_pending = len(pending_files)
    print(f"\nTotal issues to process: {total_pending} (out of {len(all_issue_files)} total available in HF)")
    if total_pending == 0:
        print(f"All issues for {np_conf['title']} have already been ingested! Checkpoint is up to date.")
        return

    chunker = RecursiveChunker(
        tokenizer=config.TOKENIZER,
        chunk_size=config.CHUNK_TARGET_TOKENS,
        min_characters_per_chunk=24,
    )

    t_start = time.time()
    total_chunks_processed = 0
    total_issues_processed = 0

    pending_docs: List[Dict[str, Any]] = []
    pending_issue_paths: List[str] = []

    # Pipeline Parquet downloads & chunking using ThreadPoolExecutor
    worker_fn = functools.partial(
        fetch_and_chunk_issue,
        token=token,
        prefix=prefix,
        np_slug=np_slug,
        np_conf=np_conf,
        chunker=chunker,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        issue_stream = executor.map(worker_fn, pending_files)

        with open(jsonl_output, "a", encoding="utf-8") as jsonl_fp:
            for issue_idx, (fpath, issue_docs) in enumerate(issue_stream, 1):
                if issue_docs is None:
                    # Skip checkpointing on failed download
                    continue

                if issue_docs:
                    for doc in issue_docs:
                        jsonl_fp.write(json.dumps(doc) + "\n")
                    jsonl_fp.flush()
                    pending_docs.extend(issue_docs)

                pending_issue_paths.append(fpath)

                # Flush & Upsert when batch size threshold reached or at last issue
                if len(pending_docs) >= batch_size or issue_idx == total_pending:
                    # Deduplicate any duplicate chunk IDs preserving order
                    deduped_pending = {}
                    for d in pending_docs:
                        deduped_pending[d["_id"]] = d
                    pending_docs = list(deduped_pending.values())

                    if not dry_run and oai and index and pending_docs:
                        # 1. Generate embeddings
                        texts = [d["text"] for d in pending_docs]
                        embeddings = embed_batch_with_retry(oai, texts)
                        for doc, emb in zip(pending_docs, embeddings):
                            doc["embedding"] = emb

                        # 2. Group by year namespace and upsert
                        by_year = defaultdict(list)
                        for d in pending_docs:
                            by_year[str(d.get("year", "1910"))].append(d)

                        for yr_ns, yr_docs in by_year.items():
                            res = index.documents.batch_upsert(
                                namespace=yr_ns,
                                documents=yr_docs,
                                batch_size=min(len(yr_docs), 30),
                                max_concurrency=8,
                                show_progress=False,
                            )
                            if getattr(res, "has_errors", False):
                                print(f"  ⚠ Upsert warnings in namespace {yr_ns}: {getattr(res, 'errors', res)}")
                                retryable = [
                                    item for err in getattr(res, "errors", []) if getattr(err, "retryable", False) for item in err.items
                                ]
                                if retryable:
                                    print(f"  Retrying {len(retryable)} documents in namespace {yr_ns}...")
                                    time.sleep(1)
                                    index.documents.batch_upsert(
                                        namespace=yr_ns,
                                        documents=retryable,
                                        batch_size=10,
                                        max_concurrency=4,
                                        show_progress=False,
                                    )

                    # 3. Mark issues as checkpointed
                    for p in pending_issue_paths:
                        checkpoint.mark_done(p)
                    checkpoint.save()

                    total_chunks_processed += len(pending_docs)
                    total_issues_processed += len(pending_issue_paths)

                    elapsed = time.time() - t_start
                    rate = total_chunks_processed / elapsed if elapsed > 0 else 0
                    pct = (issue_idx / total_pending) * 100
                    rem_issues = total_pending - issue_idx
                    est_rem_chunks = rem_issues * 30  # estimated avg
                    rem_seconds = est_rem_chunks / rate if rate > 0 else 0

                    print(
                        f"[{issue_idx:4d}/{total_pending:4d} ({pct:5.1f}%)] "
                        f"Flushed {len(pending_docs):3d} chunks | "
                        f"Total Chunks: {total_chunks_processed:,} | "
                        f"Rate: {rate:.1f} chunks/s | "
                        f"ETA: {rem_seconds/60:.1f}m",
                        flush=True,
                    )

                    pending_docs = []
                    pending_issue_paths = []

    total_time = time.time() - t_start
    print(f"\n🎉 Finished ingesting {np_conf['title']}!")
    print(f"   Processed Issues : {total_issues_processed:,}")
    print(f"   Processed Chunks : {total_chunks_processed:,}")
    print(f"   Elapsed Time     : {total_time/60:.1f} minutes\n")


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
    target_years = [y.strip() for y in args.years.split(",") if y.strip()]

    target_newspapers = []
    if args.newspaper in ("all_5", "all"):
        target_newspapers = [
            "chicago_eagle",
            "the_beatrice_daily_express",
            "the_evening_world",
            "the_san_francisco_call",
            "the_sun",
        ]
    else:
        target_newspapers = [args.newspaper]

    for np_slug in target_newspapers:
        process_newspaper(
            np_slug=np_slug,
            years=target_years,
            batch_size=args.batch_size,
            limit_issues=args.limit_issues,
            dry_run=args.dry_run,
            reset_checkpoint=args.reset_checkpoint,
            oai=oai,
            index=index,
            api=api,
        )


if __name__ == "__main__":
    main()
