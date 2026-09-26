#!/usr/bin/env python3
"""Stage 1: Extract, filter, and chunk historical newspaper issues.

Supports:
- Los Angeles Herald (California, West Coast)
- The Evening World (New York, East Coast)

Features:
- Downloads Parquet issues on demand from Hugging Face Hub (cached locally).
- Drops heavy OCR layout arrays (words, boxes, lines, blocks, confidences).
- Recursively chunks page OCR text using Chonkie (~500 tokens, cl100k_base).
- Formats each chunk with Pinecone Structured IDs:
    Parent Page ID: <prefix>#<YYYY-MM-DD>#ed-<edition>#p<sequence>
    Chunk ID      : <prefix>#<YYYY-MM-DD>#ed-<edition>#p<sequence>#c<chunk_idx>
- Writes chunks to JSONL with checkpointing.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pyarrow.parquet as pq
from chonkie import RecursiveChunker
from huggingface_hub import HfApi, hf_hub_download

import config
from iiif import jp2_to_iiif_url


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk historical newspaper Parquet datasets into Pinecone-ready JSONL"
    )
    parser.add_argument(
        "--newspaper",
        default="the_evening_world",
        choices=list(config.NEWSPAPERS.keys()),
        help=f"Newspaper to process ({', '.join(config.NEWSPAPERS.keys())})",
    )
    parser.add_argument(
        "--years",
        default=",".join(config.TARGET_YEARS),
        help=f"Comma-separated list of years to process (default: {','.join(config.TARGET_YEARS)})",
    )
    parser.add_argument(
        "--months",
        default=",".join(config.TARGET_MONTHS),
        help=f"Comma-separated list of months to process (default: {','.join(config.TARGET_MONTHS)})",
    )
    parser.add_argument(
        "--match-dates-from",
        default=str(config.CHUNKS_PATH),
        help="Path to an existing chunks JSONL file to match exact dates from (e.g. data/chunks.jsonl)",
    )
    parser.add_argument(
        "--limit-issues-per-year",
        type=int,
        default=3,
        help="Number of issue files per year (default: 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (default: data/chunks_{newspaper}.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_TARGET_TOKENS,
        help=f"Target tokens per chunk (default: {config.CHUNK_TARGET_TOKENS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run and overwrite existing chunks and checkpoints",
    )
    return parser.parse_args()


def get_matching_files(
    api: HfApi,
    repo_id: str,
    newspaper_slug: str,
    years: List[str],
    match_dates_path: Optional[Path],
    limit_per_year: int = 3,
) -> Dict[str, List[str]]:
    """Discover available Parquet files matching target dates."""
    np_conf = config.NEWSPAPERS[newspaper_slug]
    path_prefix = np_conf["path_prefix"]
    matched: Dict[str, List[str]] = {}

    # Extract target dates from existing reference chunks if available
    ref_dates_by_year = defaultdict(list)
    if match_dates_path and match_dates_path.exists():
        with open(match_dates_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                    y = str(int(c.get("year", 0)))
                    d = str(c.get("date", ""))
                    if d and d not in ref_dates_by_year[y]:
                        ref_dates_by_year[y].append(d)
                except Exception:
                    pass

    target_mds = [
        "01-01", "01-02", "01-03", "01-04",
        "07-01", "07-02", "07-03", "07-04",
        "12-01", "12-02", "12-03",
        "08-01", "08-02",
    ]

    print(f"Discovering matching issue files in {path_prefix}...")

    for yr in sorted(years):
        path_in_repo = f"{path_prefix}/{yr}"
        try:
            tree = list(
                api.list_repo_tree(
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type="dataset",
                    recursive=False,
                )
            )
            parquet_files = sorted([item.path for item in tree if item.path.endswith(".parquet")])
            file_map = {}
            for p in parquet_files:
                m = re.search(r"(\d{4})_(\d{2})_(\d{2})\.parquet$", p)
                if m:
                    file_map[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = p

            ref_dates = sorted(ref_dates_by_year.get(yr, []))
            chosen = []

            # 1. First priority: Exact matches with year reference dates
            for d in ref_dates:
                if d in file_map and file_map[d] not in chosen:
                    chosen.append(file_map[d])

            # 2. Second priority: Preferred target month-days (01-01..04, 07-01..04, 12-01..03)
            if len(chosen) < limit_per_year:
                for md in target_mds:
                    cand_d = f"{yr}-{md}"
                    if cand_d in file_map and file_map[cand_d] not in chosen:
                        chosen.append(file_map[cand_d])
                    if len(chosen) >= limit_per_year:
                        break

            # 3. Third priority: Any issues in target months (01, 07, 12, 08)
            if len(chosen) < limit_per_year:
                for m_prefix in [f"{yr}-01", f"{yr}-07", f"{yr}-12", f"{yr}-08"]:
                    for d, p in sorted(file_map.items()):
                        if d.startswith(m_prefix) and p not in chosen:
                            chosen.append(p)
                        if len(chosen) >= limit_per_year:
                            break
                    if len(chosen) >= limit_per_year:
                        break

            # 4. Fourth priority: Any available files in that year
            if len(chosen) < limit_per_year:
                for c in parquet_files:
                    if c not in chosen:
                        chosen.append(c)
                    if len(chosen) >= limit_per_year:
                        break

            matched[yr] = chosen[:limit_per_year]
            print(f"  Year {yr}: {len(matched[yr])} matching issue file(s)")
        except Exception as e:
            print(f"  Year {yr}: Failed to inspect or does not exist ({e})")
            matched[yr] = []

    return matched


def main():
    args = parse_args()
    np_slug = args.newspaper
    np_conf = config.NEWSPAPERS[np_slug]

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        if np_slug == "los_angeles_herald":
            out_path = config.CHUNKS_PATH.resolve()
        elif np_slug == "the_evening_world":
            out_path = config.CHUNKS_NY_PATH.resolve()
        else:
            out_path = (config.DATA_DIR / f"chunks_{np_slug}.jsonl").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_file = config.DATA_DIR / f".prepare_{np_slug}_done"

    if done_file.exists() and not args.force:
        print(f"Prepare phase for '{np_slug}' already marked complete ({done_file}).")
        print("Pass --force to re-generate chunks. Exiting.")
        return

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    ref_path = Path(args.match_dates_from).resolve() if args.match_dates_from else None

    print("=" * 70)
    print(f"Stage 1: Chonkie Recursive Chunking for {np_conf['title']} ({np_conf['region']})")
    print(f"Source HF Repo    : {config.HF_REPO}")
    print(f"Target Years      : {years}")
    print(f"Aligning with     : {ref_path}")
    print(f"Chunk Target Size : {args.chunk_size} tokens ({config.TOKENIZER})")
    print(f"Destination File  : {out_path}")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    year_files = get_matching_files(
        api=api,
        repo_id=config.HF_REPO,
        newspaper_slug=np_slug,
        years=years,
        match_dates_path=ref_path,
        limit_per_year=args.limit_issues_per_year,
    )

    total_files = sum(len(flist) for flist in year_files.values())
    print(f"\nTotal issue files to process: {total_files}")
    if total_files == 0:
        print("No files matched the criteria. Exiting.")
        return

    chunker = RecursiveChunker(
        tokenizer=config.TOKENIZER,
        chunk_size=args.chunk_size,
        min_characters_per_chunk=24,
    )

    t0 = time.time()
    total_pages = 0
    total_chunks = 0
    processed_files = 0

    mode = "w" if (args.force or not out_path.exists()) else "a"
    prefix = np_conf["prefix"]

    with open(out_path, mode, encoding="utf-8") as out_f:
        for yr, files in year_files.items():
            print(f"\nProcessing Year {yr} ({len(files)} issues)...")
            yr_pages = 0
            yr_chunks = 0

            for fpath in files:
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
                    continue

                for row in rows:
                    raw_text = (row.get("text") or "").strip()
                    if not raw_text or len(raw_text) < 30:
                        continue

                    total_pages += 1
                    yr_pages += 1

                    date_str = str(row.get("date") or f"{yr}-01-01")
                    seq = int(row.get("sequence") or 1)
                    ed = str(row.get("edition") or "1")
                    month_int = int(row.get("month") or 1)
                    day_int = int(row.get("day") or 1)
                    year_int = int(row.get("year") or int(yr))

                    season = config.SEASON_MAP.get(month_int, "Unknown")
                    raw_jp2 = row.get("image_url") or ""
                    iiif_thumb = jp2_to_iiif_url(raw_jp2, size="600,") if raw_jp2 else ""

                    # Structured Parent Page ID per Pinecone Guidelines
                    parent_page_id = f"{prefix}#{date_str}#ed-{ed}#p{seq:02d}"

                    # Chunk page text using Chonkie
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
                            "edition": ed,
                            "sequence": seq,
                            "chunk_index": c_idx,
                            "chunk_count": chunk_count,
                            "token_count": c.token_count,
                            "loc_page_url": row.get("loc_page_url") or "",
                            "pdf_url": row.get("pdf_url") or "",
                            "image_url": raw_jp2,
                            "iiif_thumb_url": iiif_thumb,
                        }

                        out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        total_chunks += 1
                        yr_chunks += 1

                processed_files += 1
                elapsed = time.time() - t0
                rate = processed_files / elapsed if elapsed > 0 else 0
                print(
                    f"  [{processed_files:2d}/{total_files:2d}] {fpath.split('/')[-1]} -> "
                    f"{yr_pages} pages, {yr_chunks} chunks ({rate:.1f} issues/s)",
                    flush=True,
                )

            print(f"  ✓ Year {yr} complete: {yr_pages} pages -> {yr_chunks} chunks.")

    done_file.write_text(f"done={total_chunks}\n")
    print("\n" + "=" * 70)
    print(f"Prepare phase for '{np_slug}' finished successfully!")
    print(f"Total issue files processed : {processed_files}")
    print(f"Total pages extracted       : {total_pages:,}")
    print(f"Total chunks written        : {total_chunks:,}")
    print(f"Output destination          : {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
