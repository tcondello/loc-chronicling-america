#!/usr/bin/env python3
"""Example 5: Download 100 newspaper documents and export as Pinecone-compatible JSONL.

This script demonstrates downloading historical newspaper page records with full OCR text
and formatting them into JSON Lines (.jsonl) conforming to the Pinecone Document Schema format:
https://docs.pinecone.io/guides/index-data/import-data#prepare-document-schema-files-jsonl

Pinecone Document Schema Format:
- Each line is a single JSON document.
- Required `_id`: Non-empty string, unique within the namespace.
- Full-text search field: JSON string (e.g. `text` or `body`).
- Metadata fields: Stored and auto-indexed (strings, numbers, booleans, list[str]).
- Directory layout: <import_dir>/<namespace>/0.jsonl (or `__default__/0.jsonl`).
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from loc_chronicling_america import ChroniclingAmerica
from loc_chronicling_america.models import IssueRecord, PageRecord


def validate_pinecone_document(doc: Dict[str, Any], seen_ids: set, text_field: str = "text") -> None:
    """Validate that a document strictly conforms to Pinecone's Document Schema requirements."""
    # 1. Required _id
    if "_id" not in doc:
        raise ValueError("Document missing required '_id' field.")
    doc_id = doc["_id"]
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"Invalid '_id': must be a non-empty string, got: {doc_id!r}")
    if doc_id in seen_ids:
        raise ValueError(f"Duplicate '_id' detected: {doc_id}")
    seen_ids.add(doc_id)

    # 2. Schema field (full-text search string)
    if text_field not in doc:
        raise ValueError(f"Document missing declared schema field '{text_field}'")
    if not isinstance(doc[text_field], str):
        raise ValueError(f"Schema full-text field '{text_field}' must be a string, got {type(doc[text_field])}")

    # 3. Metadata rules
    for k, v in doc.items():
        if k == "_id":
            continue
        if k.startswith("_"):
            raise ValueError(f"Metadata key '{k}' starts with '_' (reserved for internal use)")
        if k.startswith("$"):
            raise ValueError(f"Metadata key '{k}' starts with '$' (reserved for filter operators)")
        if v is None:
            raise ValueError(f"Null value found for metadata key '{k}'")
        if not isinstance(v, (str, int, float, bool, list)):
            raise ValueError(f"Metadata key '{k}' has unsupported type: {type(v)}")
        if isinstance(v, list):
            if not all(isinstance(elem, str) for elem in v):
                raise ValueError(f"List metadata key '{k}' must contain only strings (arrays of numbers are rejected by Pinecone)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Chronicling America newspaper pages and save as Pinecone JSONL"
    )
    parser.add_argument(
        "--lccn",
        default="00225879",
        help="LCCN of newspaper to download (default: '00225879' - The Monitor, Omaha)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of document records to download (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="./example_output/pinecone_export",
        help="Output directory for JSONL files (default: ./example_output/pinecone_export)",
    )
    parser.add_argument(
        "--namespace",
        default="__default__",
        help="Target namespace for Pinecone bulk import (default: '__default__')",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Schema full-text field name in document (default: 'text')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent workers for fetching page OCR text (default: 6)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pinecone bulk import folder structure: <import_dir>/<namespace>/0.jsonl
    ns_dir = out_dir / "import_data" / args.namespace
    ns_dir.mkdir(parents=True, exist_ok=True)
    import_jsonl_path = ns_dir / "0.jsonl"
    flat_jsonl_path = out_dir / "pinecone_documents.jsonl"
    schema_path = out_dir / "pinecone_schema.json"

    print("=" * 70)
    print("Pinecone Document Schema Exporter for Chronicling America")
    print(f"Target Newspaper LCCN : {args.lccn}")
    print(f"Target Document Count : {args.limit}")
    print(f"Output Directory      : {out_dir.resolve()}")
    print(f"Namespace Subdirectory: {ns_dir.relative_to(out_dir)}/0.jsonl")
    print("=" * 70)

    client = ChroniclingAmerica()

    # Step 1: Discover issues for the title
    print(f"\n1. Querying issues for LCCN '{args.lccn}'...")
    # Estimate issues needed (typically 4 to 8 pages per issue)
    issues_to_fetch = max(10, (args.limit // 4) + 5)
    issue_metas = client.get_issues(args.lccn, limit=issues_to_fetch, sort="date")
    print(f"Found {len(issue_metas)} available issues in chronological order.")

    if not issue_metas:
        print("No issues found. Exiting.")
        sys.exit(1)

    # Step 2: Resolve issues and gather pages until target document limit reached
    print(f"\n2. Resolving issues and collecting page records (target: {args.limit})...")
    collected_pages: List[PageRecord] = []
    issue_cache: Dict[str, IssueRecord] = {}

    for imeta in issue_metas:
        issue_url = imeta["url"]
        try:
            issue_rec = client.resolve(issue_url)
            issue_cache[issue_rec.date] = issue_rec
            for page in issue_rec.pages:
                collected_pages.append(page)
                if len(collected_pages) >= args.limit:
                    break
            print(f"  - Resolved {issue_rec.date}: {len(issue_rec.pages)} pages (Total collected: {len(collected_pages)})")
        except Exception as e:
            print(f"  ! Warning: Failed to resolve {issue_url}: {e}")

        if len(collected_pages) >= args.limit:
            break

    target_pages = collected_pages[: args.limit]
    print(f"\nSuccessfully queued {len(target_pages)} pages across {len(issue_cache)} issues.")

    # Step 3: Concurrently fetch plain text OCR for all pages
    print(f"\n3. Fetching full plain text OCR for {len(target_pages)} documents using {args.workers} workers...")
    t0 = time.time()

    def fetch_page_text(p: PageRecord) -> PageRecord:
        p.get_text()
        return p

    processed_pages: List[PageRecord] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_page_text, p) for p in target_pages]
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            page = f.result()
            processed_pages.append(page)
            completed += 1
            if completed % 10 == 0 or completed == len(target_pages):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Fetched {completed:3d}/{len(target_pages)} OCR texts ({completed*100//len(target_pages):3d}%) - {rate:.1f} docs/sec")

    # Re-sort to maintain chronological and sequence order
    processed_pages.sort(key=lambda p: (p.date, p.edition, p.sequence))
    t1 = time.time()
    print(f"Completed downloading {len(processed_pages)} OCR documents in {t1 - t0:.2f} seconds!")

    # Step 4: Convert to Pinecone Document Schema format and validate
    print("\n4. Converting to Pinecone Document Schema format and validating...")
    documents: List[Dict[str, Any]] = []
    seen_ids = set()

    for page in processed_pages:
        issue = issue_cache.get(page.date)
        extra_meta = {
            "newspaper_title": issue.newspaper_title if issue else "The Monitor",
            "batch_name": issue.batch_name if issue else "Unknown",
            "page_count": issue.page_count if issue else len(processed_pages),
            "loc_item_url": issue.loc_item_url if issue else page.url,
            "city": issue.city if (issue and issue.city) else "Omaha",
            "state": issue.state if (issue and issue.state) else "Nebraska",
        }
        doc = page.to_pinecone_document(
            text_field=args.text_field,
            extra_metadata=extra_meta,
        )
        # Strict validation against Pinecone rules
        validate_pinecone_document(doc, seen_ids, text_field=args.text_field)
        documents.append(doc)

    print(f"All {len(documents)} documents validated successfully with 0 schema violations!")

    # Step 5: Save JSONL files
    print("\n5. Writing JSONL files...")
    # 1. Write flat consolidated JSONL
    with open(flat_jsonl_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved consolidated JSONL : {flat_jsonl_path} ({flat_jsonl_path.stat().st_size / 1024:.1f} KB)")

    # 2. Write Pinecone import directory structure
    with open(import_jsonl_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved Pinecone import file: {import_jsonl_path}")

    # 3. Write Pinecone schema definition
    pinecone_schema = {
        "name": "chronicling-america-schema",
        "description": "Pinecone Document Schema for Chronicling America historical newspaper records",
        "schema": {
            "fields": {
                args.text_field: {
                    "type": "string",
                    "full_text_search": {
                        "language": "en"
                    }
                }
            }
        },
        "metadata_fields_auto_indexed": [
            "title",
            "newspaper_title",
            "lccn",
            "date",
            "year",
            "month",
            "day",
            "edition",
            "sequence",
            "page_count",
            "batch_name",
            "city",
            "state",
            "url",
            "image_url",
            "pdf_url",
            "char_count",
            "word_count"
        ]
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(pinecone_schema, f, indent=2)
    print(f"  ✓ Saved index schema template: {schema_path}")

    # Step 6: Print sample document
    print("\n" + "=" * 70)
    print("Sample Pinecone Document (First record in JSONL):")
    print("-" * 70)
    sample = dict(documents[0])
    # Truncate text preview for clean console display
    preview_len = 240
    if len(sample[args.text_field]) > preview_len:
        sample[args.text_field] = sample[args.text_field][:preview_len] + f"... [{len(sample[args.text_field])} chars total]"
    print(json.dumps(sample, indent=2))
    print("=" * 70)
    print(f"\nDone! Successfully downloaded and formatted {len(documents)} Pinecone documents.")
    print("You can now sync the 'import_data/' folder to S3/GCS/Azure and call index.start_import(...)")


if __name__ == "__main__":
    main()
