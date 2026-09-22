# Examples Guide

This directory contains standalone, runnable Python examples demonstrating how to use `loc-chronicling-america` for historical research, batch processing, OCR layout analysis, and vector search indexing.

---

## Example Walkthroughs

| Script | Purpose | Description |
| :--- | :--- | :--- |
| [`01_resolve_loc_record.py`](01_resolve_loc_record.py) | **Record & Batch Resolution** | Resolves an LoC URL (item, resource, gallery) or LCCN + date to its bibliographic metadata, page links, and exact source NDNP batch archive. |
| [`02_download_issue_pages.py`](02_download_issue_pages.py) | **Single Page & Issue Download** | Downloads individual single-page PDFs, high-res JP2 master scans, ALTO XML, or plain text OCR without downloading gigabytes of bulk archives. |
| [`03_search_newspapers_and_batches.py`](03_search_newspapers_and_batches.py) | **Offline SQLite Catalog Search** | Searches 15,000+ digitized titles and 3,000+ NDNP batches offline by keyword, state, or awardee institution with sub-millisecond latency. |
| [`04_inspect_alto_ocr.py`](04_inspect_alto_ocr.py) | **ALTO OCR Layout & Bounding Boxes** | Parses ALTO XML (v2-v4) to extract word tokens, normalized $[0, 1000]$ bounding boxes, line structures, and OCR confidence scores (`WC`). |
| [`05_export_pinecone_jsonl.py`](05_export_pinecone_jsonl.py) | **Pinecone Document Schema Export** | Downloads newspaper pages and produces validated Pinecone Document Schema JSONL (`<import_dir>/<namespace>/0.jsonl`) ready for object storage import. |
| [`06_run_streaming_pipeline.py`](06_run_streaming_pipeline.py) | **Streaming Batch Ingestion** | Transmutes raw `.tar.bz2` NDNP archives directly into compressed issue-level Apache Parquet files with zero local disk accumulation. |
| [`07_convert_parquet_to_pinecone.py`](07_convert_parquet_to_pinecone.py) | **Parquet to Pinecone Converter** | Converts generated Apache Parquet datasets into chunked Pinecone Document Schema JSONL files for vector search ingestion. |

---

## Running the Examples

Ensure dependencies are installed:
```bash
pip install -e ".[pipeline]"
```

Run any example directly from the repository root:

```bash
# Example 1: Resolve a record and find its source batch
python examples/01_resolve_loc_record.py

# Example 2: Download PDF, JP2, and OCR for an issue
python examples/02_download_issue_pages.py

# Example 3: Search offline catalog
python examples/03_search_newspapers_and_batches.py

# Example 4: Inspect ALTO XML OCR bounding boxes
python examples/04_inspect_alto_ocr.py

# Example 5: Export Pinecone JSONL documents
python examples/05_export_pinecone_jsonl.py --limit 50

# Example 6: Run streaming Parquet pipeline on a sample batch
python examples/06_run_streaming_pipeline.py --batch nbu_indescribablebeast_ver01

# Example 7: Convert Parquet files to Pinecone JSONL shards
python examples/07_convert_parquet_to_pinecone.py --input ./export_data/newspapers/nebraska/ --output ./pinecone_output
```
