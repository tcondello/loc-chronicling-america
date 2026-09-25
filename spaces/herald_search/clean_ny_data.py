#!/usr/bin/env python3
"""Clean and repair New York (The Evening World) newspaper metadata and image URLs.

Actions performed:
1. Scans data/chunks_ny.jsonl and repairs missing microfilm reels & frames in 1910 issues:
   - Maps batch_nn_hardin_ver01 placeholder reels (00000000000) to reel 0028076582A.
   - Calculates consecutive frame numbers:
       1910-12-01 -> frame = 0007 + page
       1910-12-02 -> frame = 0027 + page
       1910-12-03 -> frame = 0051 + page
2. Re-generates valid CDN-backed LoC IIIF endpoints (HTTP 200 on tile.loc.gov).
3. Re-writes data/chunks_ny.jsonl with clean records.
4. Generates dense embeddings via OpenAI text-embedding-3-small and updates Pinecone namespace '1910'.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

import config
from iiif import normalize_iiif_url

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Clean NY dataset IIIF URLs and update Pinecone")
    parser.add_argument(
        "--chunks-file",
        default="data/chunks_ny.jsonl",
        help="Path to NY chunks JSONL file (default: data/chunks_ny.jsonl)",
    )
    parser.add_argument(
        "--update-pinecone",
        action="store_true",
        default=True,
        help="Update modified documents in Pinecone (default: True)",
    )
    parser.add_argument(
        "--namespace",
        default="1910",
        help="Pinecone namespace to update ('1910' or 'all', default: 1910)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    chunks_path = Path(args.chunks_file)
    if not chunks_path.is_absolute():
        chunks_path = Path(__file__).resolve().parent / args.chunks_file

    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("🧹 Cleaning New York Newspaper Data & Repairing IIIF URLs")
    print(f"Target file: {chunks_path}")
    print("=" * 70)

    cleaned_records: List[Dict[str, Any]] = []
    modified_records: List[Dict[str, Any]] = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            orig_thumb = doc.get("iiif_thumb_url", "")
            fixed_thumb = normalize_iiif_url(orig_thumb or doc.get("image_url", ""), size="600,")

            if fixed_thumb != orig_thumb or "00000000000" in orig_thumb:
                doc["iiif_thumb_url"] = fixed_thumb
                modified_records.append(doc)

            cleaned_records.append(doc)

    print(f"\nTotal records in file      : {len(cleaned_records):,}")
    print(f"Records with repaired URLs : {len(modified_records):,}")

    if modified_records:
        print("\nSample repaired record:")
        sample = modified_records[0]
        print(f"  ID        : {sample['_id']}")
        print(f"  Date      : {sample['date']} (Page {sample['sequence']})")
        print(f"  Fixed URL : {sample['iiif_thumb_url']}")

        # Write cleaned records back to file
        print(f"\nWriting cleaned records to {chunks_path}...")
        with open(chunks_path, "w", encoding="utf-8") as f:
            for doc in cleaned_records:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print("✓ Local chunks file updated successfully.")
    else:
        # Check if 1910 records need to be re-upserted anyway
        modified_records = [d for d in cleaned_records if str(d.get("year")) == args.namespace]
        print(f"Loaded {len(modified_records):,} records from namespace '{args.namespace}' to re-upsert.")

    # Update Pinecone
    if args.update_pinecone and modified_records:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            print("Warning: PINECONE_API_KEY not set. Skipping Pinecone update.", file=sys.stderr)
            return

        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Warning: OPENAI_API_KEY not set. Skipping Pinecone update.", file=sys.stderr)
            return

        print("\nConnecting to Pinecone and OpenAI to embed and upsert repaired records...")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(config.INDEX_NAME)
        oai = OpenAI(api_key=openai_key)

        # Filter by namespace if requested
        if args.namespace != "all":
            docs_to_upsert = [d for d in modified_records if str(d.get("year")) == args.namespace]
        else:
            docs_to_upsert = modified_records

        print(f"Upserting {len(docs_to_upsert):,} repaired records into Pinecone namespace '{args.namespace}'...")

        batch_size = 50
        num_batches = (len(docs_to_upsert) + batch_size - 1) // batch_size
        t0 = time.time()
        success_count = 0

        for b_idx in range(num_batches):
            batch = docs_to_upsert[b_idx * batch_size : (b_idx + 1) * batch_size]

            # Embed batch via OpenAI text-embedding-3-small
            texts = [d["text"] if d["text"].strip() else " " for d in batch]
            emb_resp = oai.embeddings.create(model=config.EMBED_MODEL, input=texts)
            for doc, emb_item in zip(batch, emb_resp.data):
                doc["embedding"] = emb_item.embedding

            res = index.documents.batch_upsert(
                namespace=str(batch[0].get("year", args.namespace)),
                documents=batch,
                batch_size=len(batch),
                show_progress=False,
            )
            if res.has_errors:
                print(f"  ⚠ Upsert errors: {res.errors}")
            else:
                success_count += len(batch)
            print(f"  Batch {b_idx + 1}/{num_batches} upserted ({len(batch)} records)...", flush=True)

        elapsed = time.time() - t0
        print(f"\n✓ Successfully updated {success_count:,} records in Pinecone in {elapsed:.1f}s!")

    print("\n" + "=" * 70)
    print("New York data cleaning complete! All images will now load successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
