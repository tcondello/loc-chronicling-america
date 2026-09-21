# loc-chronicling-america

[![PyPI version](https://img.shields.io/badge/pypi-loc--chronicling--america-blue.svg)](https://pypi.org/project/loc-chronicling-america/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python library and CLI tool built specifically for researchers, historians, and data scientists to discover, understand, and download historical digitized newspaper records and bulk batches from the **Library of Congress [Chronicling America](https://chroniclingamerica.loc.gov/)** collection.

---

## The Problem `loc-chronicling-america` Solves

The US Library of Congress hosts over **3,000 batches** comprising tens of millions of scanned and OCRed newspaper pages published between 1770 and 1963. However, researchers encounter major obstacles when trying to use these bulk assets:

1. **The Batch Disconnect**: Batches are cataloged under internal institutional awardee codes (e.g. `nbu_indescribablebeast_ver01`). When viewing a record online (such as [The Monitor, July 3, 1915](https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery)), there is no straightforward indicator showing which batch holds the files.
2. **Download Scale Barrier**: Uncompressed batch directories often exceed 60 GB, and compressed bulk OCR files (`.tar.bz2`) are hundreds of megabytes or gigabytes. Researchers who only need specific issues or articles are forced to either download massive archives or scrape web pages.
3. **Complex XML Schemas**: Newspaper OCR text and bounding boxes are stored in METS/MODS and ALTO XML schemas with diverse versions and namespaces.

`loc-chronicling-america` bridges this gap:
- **Instant Record Resolution**: Feed in any LoC URL (resource, item, gallery) or LCCN + date, and immediately get the newspaper metadata, page links, and exact source batch name.
- **Granular Asset Downloads**: Download individual single-page PDFs, master JP2 scans, ALTO XML files, or plain text OCR directly.
- **Embedded SQLite Catalog**: Instant offline search across batches and 15,000+ digitized titles with zero extra database dependencies.
- **ALTO/METS Parser**: High-performance parser extracting text, word coordinates, reading order, and OCR confidence scores (`WC`).

---

## Installation

```bash
pip install loc-chronicling-america
```

---

## Quickstart

### 1. Resolve a Record from an LoC URL

Given any Chronicling America link (like the inaugural issue of *The Monitor* in Omaha, NE):

```python
from loc_chronicling_america import ChroniclingAmerica

loc = ChroniclingAmerica()

# Resolve directly from URL
record = loc.resolve("https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery")

print(f"Title: {record.title}")
# Output: The monitor (Omaha, Neb.), July 3, 1915

print(f"Source Batch: {record.batch_name}")
# Output: nbu_indescribablebeast_ver01

print(f"Bulk Archive: {record.bulk_ocr_url}")
# Output: https://chroniclingamerica.loc.gov/data/ocr/nbu_indescribablebeast_ver01.tar.bz2

print(f"Total Pages: {record.page_count}")
# Output: 8
```

### 2. Inspect and Download Page Assets

You can inspect individual pages and download single-page PDFs, master JP2 scans, or plain text OCR without downloading the 895 MB batch archive:

```python
page1 = record.get_page(1)

# Get plain text OCR directly
print(page1.get_text()[:300])

# Download single-page PDF
pdf_path = page1.download_pdf("./downloads")
print(f"Saved PDF to: {pdf_path}")

# Download high-res JP2 master scan
jp2_path = page1.download_jp2("./downloads")

# Or download all pages for the issue at once:
record.download_all("./downloads", asset_types=["pdf", "txt"])
```

### 3. Extract Word Bounding Boxes and OCR Confidence

Researchers analyzing OCR quality or building visual search highlighting can parse the ALTO XML layout:

```python
alto = page1.get_alto()

print(f"Page dimensions: {alto.page_width} x {alto.page_height} {alto.measurement_unit}")

for word in alto.words[:10]:
    print(f"Word: '{word.content}' | Box: {word.box} | Confidence: {word.confidence}")
```

### 4. Search the Offline SQLite Catalog

The embedded SQLite catalog allows instant offline cross-referencing:

```python
# Search newspaper titles
titles = loc.search_titles("Monitor", state="Nebraska")
for t in titles:
    print(f"{t.name} ({t.lccn}) - {t.city}, {t.state} [{t.formatted_years}]")

# Search batches by state or keyword
batches = loc.search_batches(state="NE")
for b in batches[:5]:
    print(f"{b.name} ({b.size_mb} MB) - Ingested: {b.ingested_date}")
```

---

## Command Line Interface (CLI)

`loc-chronicling-america` includes both `loc-chronam` and `chronam` CLI tools:

### Resolve a Record
```bash
loc-chronam resolve "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery"
```

### Download an Issue or Specific Page
```bash
# Download all pages as PDFs
loc-chronam download "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/" --format pdf --dest ./papers

# Download page 1 as plain text OCR
loc-chronam download "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/" --page 1 --format txt
```

### Inspect or Download a Batch
```bash
# Get batch metadata (size, LCCNs, page count)
loc-chronam batch info nbu_indescribablebeast_ver01

# Download entire .tar.bz2 bulk archive
loc-chronam batch download nbu_indescribablebeast_ver01 --dest ./bulk
```

### Search Titles and Batches
```bash
loc-chronam search titles "Monitor" --state Nebraska
loc-chronam search batches --state NE
```

### Sync Catalog
```bash
loc-chronam catalog --sync
```

---

## Architecture

- **`loc_chronicling_america.resolver`**: Translates LoC newspaper item and resource URLs to issue records and maps them to their source batches.
- **`loc_chronicling_america.downloader`**: Resumable HTTP client with progress bars, rate-limiting, and SHA-256 integrity checks.
- **`loc_chronicling_america.parsers.alto`**: Fast, robust ALTO XML parser (v2, v3, v4) extracting text, bounding boxes, and word confidence scores.
- **`loc_chronicling_america.parsers.mets`**: METS issue and batch manifest parser.
- **`loc_chronicling_america.batch`**: Bulk archive inspector with streaming `.tar.bz2` iteration without disk expansion.
- **`loc_chronicling_america.db`**: Embedded SQLite database for fast offline queries and local download tracking.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
