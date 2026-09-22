# loc-chronicling-america

[![PyPI version](https://img.shields.io/badge/pypi-loc--chronicling--america-blue.svg)](https://pypi.org/project/loc-chronicling-america/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LOC--Chronicling--America-yellow)](https://huggingface.co/datasets/Tim-Pinecone/LOC-Chronicling-America)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()

A Python library, CLI tool, and high-throughput streaming ingestion pipeline built for researchers, digital humanists, and machine learning engineers to discover, resolve, and download historical digitized newspapers from the **Library of Congress [Chronicling America](https://chroniclingamerica.loc.gov/) / National Digital Newspaper Program (NDNP)** collection.

---

## The Problems This Solves

The US Library of Congress hosts over **3,000 bulk archives** comprising tens of millions of scanned, OCR-transcribed newspaper pages published between 1770 and 1963. However, working with these bulk assets presents significant engineering challenges:

1. **The Batch Disconnect**: Batches are cataloged under internal institutional awardee codes (e.g. `nbu_indescribablebeast_ver01`). When viewing a newspaper online (such as [*The Monitor*, July 3, 1915](https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery)), there is no direct link indicating which batch contains the master files.
2. **Download Scale Barrier**: Uncompressed batch directories often exceed 60 GB, and compressed bulk OCR archives (`.tar.bz2`) range from hundreds of megabytes to gigabytes. Researchers who need specific issues or articles are forced to either download massive archives or scrape fragile web endpoints.
3. **Complex XML & OCR Schemas**: Newspaper OCR text and coordinates are stored in heterogeneous METS/MODS and ALTO XML schemas across diverse historical OCR software versions.
4. **Layout & Multi-Column Geometry**: Historical newspapers feature complex multi-column layouts where naive text extraction destroys reading order, headlines, and column structure.

`loc-chronicling-america` solves these problems:
* **Instant Record & Batch Resolution**: Feed in any LoC URL (item, resource, gallery) or LCCN + date, and immediately obtain publication metadata, page links, and exact source batch archives.
* **Granular Asset Downloads**: Download individual single-page PDFs, high-resolution master scans (JP2), ALTO XML files, or plain text OCR directly without bulk downloads.
* **Streaming Transmutation Pipeline**: Stream bulk `.tar.bz2` archives on-the-fly and convert them into issue-level **Apache Parquet** files with zero local disk bloat.
* **Normalized OCR Layout**: Extracts word-level bounding boxes normalized to $[0, 1000]$ integer coordinates along with structured text lines, article blocks, and OCR confidence scores (`WC`).
* **Embedded SQLite Catalog**: Instant offline search across 3,000+ batches and 15,000+ titles with zero external database dependencies.
* **Vector Search Ready**: Direct export to Pinecone Document Schema JSONL format for seamless RAG and hybrid semantic/lexical search.

---

## Live Dataset on Hugging Face

We host a nationwide, partitioned Apache Parquet dataset on Hugging Face:
👉 **[https://huggingface.co/datasets/Tim-Pinecone/LOC-Chronicling-America](https://huggingface.co/datasets/Tim-Pinecone/LOC-Chronicling-America)**

### Dataset Layout
```text
newspapers/
└── {state}/
    └── {newspaper_slug}/
        └── {state}_{newspaper_slug}_{year}_{month}_{day}.parquet
```

* **Issue-Level Shards**: Each `.parquet` file contains all pages of a specific publication date, compressed with Snappy.
* **Master Catalog**: `catalog.parquet` provides an instant master index of all titles, publication date ranges, LCCNs, and page counts.

### Parquet Schema Reference

| Field | Type | Description |
| :--- | :--- | :--- |
| `_id` | `string` | Unique page record ID: `{lccn}_{date}_ed-{edition}_seq-{sequence}` |
| `text` | `string` | Complete plain text OCR for the page |
| `title` | `string` | Human-readable page title (e.g. *The Monitor, 1915-07-03 - Page 1*) |
| `newspaper_title` | `string` | Clean publication name (e.g. *The Monitor*) |
| `newspaper_slug` | `string` | URL-safe slug (e.g. `the_monitor`) |
| `lccn` | `string` | Library of Congress Control Number (e.g. `00225879`) |
| `date` | `string` | Publication date (`YYYY-MM-DD`) |
| `year`, `month`, `day` | `int32` | Integer date components for fast numeric partition pruning |
| `edition` | `string` | Edition identifier (e.g. `1`) |
| `sequence` | `int32` | Page sequence number within the issue (1-indexed) |
| `city`, `state` | `string` | Geographic publication location |
| `ethnicity` | `string` | Subject ethnicity if cataloged (e.g. *African American*) |
| `char_count`, `word_count` | `int32` | Character and word statistics |
| `loc_page_url` | `string` | Live Library of Congress interactive page viewer URL |
| `loc_item_url` | `string` | Canonical persistent LoC item URL |
| `pdf_url` | `string` | Direct download link for single-page PDF on `tile.loc.gov` |
| `image_url` | `string` | Direct download link for master JP2 scan on `tile.loc.gov` |
| `source_batch` | `string` | NDNP provenance batch name (e.g. `nbu_indescribablebeast_ver01`) |
| `awardee` | `string` | Digitizing institution (e.g. *University of Nebraska-Lincoln*) |
| `awardee_code` | `string` | Institution code prefix (e.g. `nbu`, `vi`, `iune`) |
| `reel_id` | `string` | Microfilm container / reel ID |
| `ocr_engine` | `string` | OCR software and version recorded in METS |
| `page_width`, `page_height`| `int32` | Physical scan pixel dimensions |
| `page_unit` | `string` | Scanner measurement unit |
| `words` | `list<string>` | OCR word tokens in logical reading order |
| `boxes` | `list<list<int16>>` | Word bounding boxes `[x0, y0, x1, y1]` normalized to $[0, 1000]$ |
| `word_confidences` | `list<float32>` | Word OCR confidence scores (0.0 to 1.0) |
| `lines` | `list<struct>` | Text line structures with `box: [x0, y0, x1, y1]` and `text` |
| `blocks` | `list<struct>` | Column/Article blocks with `block_id`, `box`, and `text` |

---

## Installation

### Standard Installation
```bash
pip install loc-chronicling-america
```

### With Parquet & Hugging Face Pipeline Support
```bash
pip install "loc-chronicling-america[pipeline]"
```

### Development Installation
```bash
git clone https://github.com/tcondello/loc-chronicling-america.git
cd loc-chronicling-america
pip install -e ".[dev]"
```

---

## Quickstart: Python SDK

### 1. Resolve a Record from an LoC URL
```python
from loc_chronicling_america import ChroniclingAmerica

client = ChroniclingAmerica()

# Resolve directly from an LoC item or resource URL
record = client.resolve("https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery")

print(f"Publication : {record.newspaper_title}")
print(f"Date        : {record.date}")
print(f"Source Batch: {record.batch_name}")
print(f"Bulk Archive: {record.bulk_ocr_url}")
print(f"Total Pages : {record.page_count}")
```

### 2. Download Specific Page Assets Without Bulk Archives
```python
# Select page 1
page = record.get_page(1)

# Stream plain text OCR
text = page.get_text()

# Download single-page PDF (stored on tile.loc.gov)
pdf_path = page.download_pdf("./downloads")

# Download high-resolution master scan (JP2)
jp2_path = page.download_jp2("./downloads")

# Or download all pages for an issue:
record.download_all("./downloads", asset_types=["pdf", "txt"])
```

### 3. Extract ALTO OCR Bounding Boxes & Confidence
```python
alto = page.get_alto()

print(f"Dimensions: {alto.page_width} x {alto.page_height} ({alto.measurement_unit})")

for word in alto.words[:5]:
    # Box coordinates are normalized [x0, y0, x1, y1] on a 1000x1000 grid
    print(f"Word: {word.content:<15} Box: {word.box}  Confidence: {word.confidence:.2f}")
```

### 4. Search Offline SQLite Catalog
```python
# Instant local search across 15,000+ digitized titles
titles = client.search_titles("Monitor", state="Nebraska")
for t in titles:
    print(f"{t.name} ({t.lccn}) - {t.city}, {t.state} [{t.formatted_years}]")

# Search batches by state or awardee
batches = client.search_batches(state="NE")
for b in batches[:5]:
    print(f"{b.name} ({b.size_mb} MB) - Ingested: {b.ingested_date}")
```

### 5. Stream from Hugging Face Using `datasets`
```python
from datasets import load_dataset

# Stream newspaper pages for Nebraska
ds = load_dataset(
    "Tim-Pinecone/LOC-Chronicling-America",
    "nebraska",
    streaming=True,
    split="train",
)

for row in ds.take(3):
    print(f"Title: {row['title']}")
    print(f"Words: {len(row['words'])} tokens | First line: {row['lines'][0]['text']}")
```

---

## Command-Line Interface (CLI)

`loc-chronicling-america` provides the `loc-chronam` (and `chronam`) command-line tool:

### Resolve a Record
```bash
loc-chronam resolve "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/"
```

### Download Pages
```bash
# Download issue as PDFs
loc-chronam download "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/" --format pdf --dest ./papers

# Download page 1 plain text OCR
loc-chronam download "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/" --page 1 --format txt
```

### Inspect or Download Bulk Batches
```bash
# Inspect batch metadata
loc-chronam batch info nbu_indescribablebeast_ver01

# Download bulk archive
loc-chronam batch download nbu_indescribablebeast_ver01 --dest ./bulk
```

### Search Titles and Batches
```bash
loc-chronam search titles "Monitor" --state Nebraska
loc-chronam search batches --state NE
```

### Run Streaming Parquet Pipeline
```bash
# Process a specific state and purge local files after upload to Hugging Face
loc-chronam export-pipeline --state NE --hf-repo "Tim-Pinecone/LOC-Chronicling-America" --purge-after-upload
```

---

## Vector Search & Pinecone Integration

The library supports direct conversion into the [Pinecone Document Schema format](https://docs.pinecone.io/guides/index-data/import-data#prepare-document-schema-files-jsonl) for hybrid lexical/vector search:

```bash
python examples/07_convert_parquet_to_pinecone.py \
    --input ./export_data/newspapers/nebraska/ \
    --output ./pinecone_import \
    --chunk-lines 100000
```

Upload the resulting `pinecone_import/` directory to Amazon S3, Google Cloud Storage, or Azure Blob Storage and call Pinecone's bulk import API:

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.Index(host="YOUR_INDEX_HOST")
index.start_import(uri="s3://my-bucket/pinecone_import")
```

---

## AWS Cloud Ingestion Architecture (CDK)

For nationwide, non-stop ingestion across all 50 states, an AWS CDK deployment stack is provided in [`infra/`](infra/):

* **Self-Healing**: Runs as an automated `systemd` service (`loc-pipeline.service`) with automatic restarts on network blips.
* **$0 NAT Gateway**: Single-AZ public subnet design with IAM-managed keyless AWS Systems Manager (SSM) terminal access.
* **Polite Rate Limiting**: Built-in 2.0-second request pacing and exponential backoff respecting HTTP 429 `Retry-After` headers to prevent CDN IP blacklisting.
* **Zero Disk Bloat**: Purges intermediate archives immediately after processing to run continuously on a 100 GB GP3 EBS volume.

To deploy to AWS:
```bash
cd infra
cdk deploy -c hf_token=<YOUR_HF_TOKEN>
```
See the [`infra/README.md`](infra/README.md) for full deployment instructions and CloudWatch log monitoring commands.

---

## Repository Structure

```text
loc-chronicling-america/
├── src/
│   ├── loc_chronicling_america/
│   │   ├── client.py         # Main ChroniclingAmerica research client
│   │   ├── resolver.py       # LoC URL & LCCN resolver mapping to batches
│   │   ├── downloader.py     # Resumable, rate-limited HTTP asset downloader
│   │   ├── batch.py          # Streaming .tar.bz2 NDNP archive inspector
│   │   ├── db.py             # Embedded SQLite catalog (titles, batches, queue)
│   │   ├── pipeline.py       # Streaming Parquet transmutation engine
│   │   ├── hf.py             # Hugging Face Hub dataset sync & metadata
│   │   ├── models.py         # Pydantic data models
│   │   ├── cli.py            # Rich CLI implementation
│   │   └── parsers/
│   │       ├── alto.py       # ALTO XML parser (v2-v4, bounding boxes, WC)
│   │       └── mets.py       # METS structure parser
├── examples/                 # Standalone runnable scripts (01-07)
│   ├── README.md             # Guide to all examples
├── scripts/                  # Production automation & maintenance scripts
│   ├── run_multi_state_pipeline.py
│   ├── cleanup_hf_dataset.py
│   └── README.md
├── infra/                    # Production AWS CDK deployment
│   ├── pipeline_stack.py     # EC2 worker, IAM, CloudWatch, VPC stack
│   ├── app.py                # CDK application entrypoint
│   └── README.md             # CDK deployment guide
├── tests/                    # Pytest test suite (26 unit/integration tests)
├── pyproject.toml            # Package configuration & dependencies
└── LICENSE                   # MIT License
```

---

## Examples

Check the [`examples/`](examples/README.md) folder for ready-to-run tutorials:
1. `01_resolve_loc_record.py`: Resolve LoC URLs to source batches.
2. `02_download_issue_pages.py`: Download single-page PDFs, JP2s, and OCR text.
3. `03_search_newspapers_and_batches.py`: Instant offline title and batch discovery.
4. `04_inspect_alto_ocr.py`: Extract OCR coordinates and confidence scores.
5. `05_export_pinecone_jsonl.py`: Generate Pinecone Document Schema JSONL.
6. `06_run_streaming_pipeline.py`: Stream batches directly to Apache Parquet.
7. `07_convert_parquet_to_pinecone.py`: Convert Parquet datasets into Pinecone shards.

---

## Testing

Run the test suite with `pytest`:
```bash
pytest
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
