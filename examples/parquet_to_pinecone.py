#!/usr/bin/env python3
"""Convert Chronicling America Parquet datasets into Pinecone Document Schema JSONL.

Reads any single .parquet file or directory of Parquet files and exports them
into Pinecone-compliant JSONL format ready for direct indexing or bulk object
storage import (S3/GCS/Azure -> index.start_import).

Usage:
    # Convert a single newspaper year shard
    python examples/parquet_to_pinecone.py --input export_data/newspapers/nebraska/the-monitor/nebraska_the-monitor_omaha_1915.parquet --output ./pinecone_output

    # Convert an entire state or directory
    python examples/parquet_to_pinecone.py --input export_data/newspapers/nebraska/ --output ./pinecone_output --chunk-lines 50000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import pyarrow.parquet as pq


def clean_pinecone_document(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure document record conforms strictly to Pinecone schema rules."""
    doc: Dict[str, Any] = {"_id": str(row["_id"]), "text": str(row.get("text", "")).strip()}

    for k, v in row.items():
        if k in ("_id", "text"):
            continue
        if v is None:
            continue
        # Pinecone metadata keys cannot start with _ or $
        if k.startswith("_") or k.startswith("$"):
            continue
        # Only accept primitive types or string lists
        if isinstance(v, (str, int, float, bool)):
            doc[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            doc[k] = v

    return doc


def main():
    parser = argparse.ArgumentParser(description="Convert Chronicling America Parquet files to Pinecone Document JSONL")
    parser.add_argument("--input", "-i", required=True, help="Path to .parquet file or directory containing .parquet files")
    parser.add_argument("--output", "-o", required=True, help="Destination directory for output JSONL files")
    parser.add_argument("--chunk-lines", type=int, default=100000, help="Max lines per JSONL shard (default: 100,000)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        parquet_files = [input_path]
    else:
        parquet_files = sorted(list(input_path.glob("**/*.parquet")))

    if not parquet_files:
        print(f"Error: No .parquet files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Chronicling America Parquet -> Pinecone JSONL Converter")
    print(f"Found {len(parquet_files)} Parquet file(s) to process.")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    total_records = 0
    file_index = 0
    current_lines = 0
    current_out_file = None
    current_handle = None

    try:
        for p_file in parquet_files:
            table = pq.read_table(p_file)
            rows = table.to_pylist()

            for raw_row in rows:
                if current_handle is None or current_lines >= args.chunk_lines:
                    if current_handle:
                        current_handle.close()
                    current_out_file = output_dir / f"pinecone_docs_{file_index:05d}.jsonl"
                    current_handle = open(current_out_file, "w", encoding="utf-8")
                    file_index += 1
                    current_lines = 0

                doc = clean_pinecone_document(raw_row)
                current_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
                current_lines += 1
                total_records += 1

                if total_records % 10000 == 0:
                    print(f"  Processed {total_records:,} documents...", flush=True)

    finally:
        if current_handle:
            current_handle.close()

    print("\n" + "=" * 70)
    print(f"Conversion complete!")
    print(f"Total documents converted : {total_records:,}")
    print(f"Total JSONL shards written: {file_index}")
    print(f"Output path               : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
