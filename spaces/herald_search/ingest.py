#!/usr/bin/env python3
"""Stage 2: Create the hybrid FTS index and ingest embedded chunks.

- Connects to Pinecone Serverless and declares Document Schema:
    - text: string, full_text_search with English stemming
    - embedding: dense_vector, 1536 dims, cosine metric
- Reads prepared chunks from data/chunks.jsonl.
- Groups chunks by year and upserts into per-year namespaces ("1890", "1905", "1910", etc.).
- Generates 1536-dim dense vectors with OpenAI text-embedding-3-small.
- Batch-offset checkpointing (.ingest_offset) so interruptions resume without re-embedding.
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
from pinecone import Pinecone, SchemaBuilder, ServerlessSpec

import config

load_dotenv()


def build_schema():
    """Declare only searchable fields in schema; filterable metadata is auto-indexed."""
    return (
        SchemaBuilder()
        .add_string_field("text", full_text_search={"language": "en", "stemming": True})
        .add_dense_vector_field("embedding", dimension=config.EMBED_DIM, metric="cosine")
        .build()
    )


def ensure_index(pc: Pinecone, index_name: str) -> None:
    """Ensure the Pinecone index exists with the correct document schema."""
    if pc.indexes.exists(index_name):
        print(f"Index '{index_name}' already exists.")
        return

    print(f"Creating Pinecone index '{index_name}' with Document Schema (FTS + Dense)...")
    schema = build_schema()
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    pc.indexes.create(
        name=index_name,
        schema=schema,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    while True:
        desc = pc.indexes.describe(index_name)
        if desc.status.ready:
            break
        print("  Waiting for index initialization...", flush=True)
        time.sleep(3)

    print(f"Index '{index_name}' is ready.")


def load_offset(offset_path: Path) -> int:
    if offset_path.exists():
        try:
            return int(offset_path.read_text().strip())
        except Exception:
            return 0
    return 0


def save_offset(offset_path: Path, offset: int) -> None:
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    offset_path.write_text(str(offset))


def embed_batch(oai: OpenAI, texts: List[str]) -> List[List[float]]:
    """Embed texts using OpenAI text-embedding-3-small."""
    # Ensure text is not empty or pure whitespace
    cleaned_texts = [t if t.strip() else " " for t in texts]
    resp = oai.embeddings.create(model=config.EMBED_MODEL, input=cleaned_texts)
    return [item.embedding for item in resp.data]


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into Pinecone hybrid FTS index")
    parser.add_argument("--chunks-file", type=str, default=str(config.CHUNKS_PATH), help="Path to chunks JSONL file")
    parser.add_argument("--offset-file", type=str, default=None, help="Path to checkpoint offset file")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Chunks per upsert batch")
    parser.add_argument("--reset-offset", action="store_true", help="Reset ingest checkpoint to beginning")
    parser.add_argument("--limit-records", type=int, default=None, help="Max records to ingest in this run")
    args = parser.parse_args()

    chunks_path = Path(args.chunks_file)
    if args.offset_file:
        offset_path = Path(args.offset_file)
    else:
        if chunks_path.stem == "chunks":
            offset_path = config.INGEST_OFFSET_PATH
        else:
            offset_path = chunks_path.parent / f".ingest_offset_{chunks_path.stem}"

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)

    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_path}. Run prepare.py first.", file=sys.stderr)
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    oai = OpenAI(api_key=openai_key)

    ensure_index(pc, config.INDEX_NAME)
    index = pc.Index(config.INDEX_NAME)

    if args.reset_offset and offset_path.exists():
        offset_path.unlink()

    offset = load_offset(offset_path)
    print(f"Loading chunks from {chunks_path} (starting from record offset {offset:,})...")

    with open(chunks_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_records = len(all_lines)
    print(f"Total chunks in file: {total_records:,}")

    if offset >= total_records:
        print(f"All records have already been ingested according to {offset_path}.")
        print("Pass --reset-offset to re-ingest.")
        return

    remaining_lines = all_lines[offset:]
    if args.limit_records:
        remaining_lines = remaining_lines[: args.limit_records]

    num_batches = (len(remaining_lines) + args.batch_size - 1) // args.batch_size
    print(f"Ingesting {len(remaining_lines):,} chunks in {num_batches} batch(es) of {args.batch_size}...")

    t0 = time.time()
    processed_count = 0

    for b_idx in range(num_batches):
        batch_slice = remaining_lines[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]
        docs: List[Dict[str, Any]] = [json.loads(line) for line in batch_slice]

        # 1. Embed texts via OpenAI
        texts = [d["text"] for d in docs]
        embeddings = embed_batch(oai, texts)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb

        # 2. Group by year namespace
        by_year: Dict[str, List[Dict[str, Any]]] = {}
        for d in docs:
            yr_str = str(d.get("year", "1910"))
            by_year.setdefault(yr_str, []).append(d)

        # 3. Batch upsert into each year's namespace
        for yr_ns, yr_docs in by_year.items():
            res = index.documents.batch_upsert(
                namespace=yr_ns,
                documents=yr_docs,
                batch_size=len(yr_docs),
                show_progress=False,
            )
            if res.has_errors:
                print(f"  ⚠ Upsert errors in namespace '{yr_ns}': {res.errors}")

        processed_count += len(docs)
        current_offset = offset + processed_count
        save_offset(offset_path, current_offset)

        elapsed = time.time() - t0
        rate = processed_count / elapsed if elapsed > 0 else 0
        rem_items = len(remaining_lines) - processed_count
        rem_sec = rem_items / rate if rate > 0 else 0

        print(
            f"  [{b_idx + 1:3d}/{num_batches:3d}] "
            f"Offset: {current_offset:,}/{total_records:,} "
            f"({(current_offset/total_records)*100:5.1f}%) | "
            f"Rate: {rate:.1f} docs/s | ETA: {rem_sec/60:.1f}m",
            flush=True,
        )

    print("\n" + "=" * 70)
    print("Ingestion complete!")
    print(f"Total chunks ingested this run : {processed_count:,}")
    print(f"Final offset checkpoint        : {load_offset(offset_path):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
