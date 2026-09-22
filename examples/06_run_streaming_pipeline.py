#!/usr/bin/env python3
"""Example 6: Run the streaming batch transmutation pipeline.

Demonstrates streaming Chronicling America bulk .tar.bz2 batches directly into
a state/newspaper/issue hierarchy of compressed Apache Parquet (.parquet) files
with normalized OCR bounding boxes, zero local disk bloat, and automatic recovery.
"""

import argparse
from pathlib import Path
from loc_chronicling_america import BatchPipeline, ChroniclingAmerica


def main():
    parser = argparse.ArgumentParser(description="Run Chronicling America streaming export pipeline")
    parser.add_argument("--batch", default="nbu_indescribablebeast_ver01", help="Batch name to process")
    parser.add_argument("--output-dir", default="./pipeline_demo_output", help="Output directory")
    parser.add_argument("--status", action="store_true", help="Print pipeline queue summary and exit")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    pipeline = BatchPipeline(
        output_dir=out_dir,
        keep_tar=False,  # Purge .tar.bz2 immediately after extraction (<3 GB scratch disk)
    )

    if args.status:
        summary = pipeline.db.get_pipeline_summary()
        print("=== Pipeline Queue Summary ===")
        for k, v in summary.items():
            print(f"  {k:<20}: {v}")
        return

    print("=" * 70)
    print("Chronicling America -> Apache Parquet Streaming Pipeline")
    print(f"Target Batch: {args.batch}")
    print(f"Output Path : {out_dir.resolve()}")
    print("=" * 70)

    # Ensure title metadata for this batch is populated in SQLite catalog
    client = ChroniclingAmerica()
    monitor_title = client.get_title("00225879")
    if not monitor_title:
        print("Caching title metadata for The Monitor (Omaha, Neb.)...")
        client.search_titles("The Monitor", state="Nebraska")

    print(f"\nProcessing batch '{args.batch}'...")
    print("1. Downloading archive to scratch space...")
    print("2. Parsing METS issue structure and ALTO OCR bounding boxes...")
    print("3. Writing issue-level Parquet files (state/newspaper/issue)...")
    print("4. Purging raw .tar.bz2 to preserve local disk...")

    res = pipeline.run(batch_name=args.batch)

    print("\n" + "=" * 70)
    print("Pipeline Execution Summary:")
    print(f"  Batches Processed : {res['batches_processed']}")
    print(f"  Pages Extracted   : {res['pages_extracted']}")
    print(f"  Elapsed Time      : {res['elapsed_seconds']}s")
    print("=" * 70)

    # Show generated files
    news_dir = out_dir / "newspapers"
    if news_dir.exists():
        print("\nGenerated State & Newspaper Issue Parquet Hierarchy:")
        for p_file in sorted(news_dir.glob("**/*.parquet"))[:10]:
            size_kb = p_file.stat().st_size / 1024
            print(f"  ✓ {p_file.relative_to(out_dir)} ({size_kb:.1f} KB)")

    cat_parquet = out_dir / "catalog.parquet"
    if cat_parquet.exists():
        print(f"\nMaster Index Table Generated: {cat_parquet} ({cat_parquet.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
