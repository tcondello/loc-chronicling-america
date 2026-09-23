"""Hugging Face Hub dataset helper for Chronicling America Parquet exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


DATASET_CARD_TEMPLATE = """---
license: cc0-1.0
pretty_name: Chronicling America (US Library of Congress) Historical Newspapers
language:
- en
annotations_creators:
- machine-generated
language_creators:
- found
multilinguality:
- multilingual
size_categories:
- 10M<n<100M
source_datasets:
- extended|other
task_categories:
- text-retrieval
- question-answering
- token-classification
- fill-mask
tags:
- history
- historical-newspapers
- chronicling-america
- library-of-congress
- national-digital-newspaper-program
- ndnp
- ocr
- layoutlm
- document-ai
- pinecone
- vector-search
- rag
- digital-humanities
- us-history
- parquet
configs:
  - config_name: default
    data_files: "newspapers/**/*.parquet"
{state_configs}
---

# Chronicling America (US Library of Congress) - Parquet Dataset

[![License: Public Domain](https://img.shields.io/badge/License-Public_Domain-blue.svg)](https://chroniclingamerica.loc.gov/about/)
[![Source: Library of Congress](https://img.shields.io/badge/Source-Library%20of%20Congress%20%2F%20NEH-red.svg)](https://chroniclingamerica.loc.gov/)
[![Format: Apache Parquet](https://img.shields.io/badge/Format-Apache%20Parquet%20(Snappy)-orange.svg)](https://parquet.apache.org/)
[![Document AI Ready](https://img.shields.io/badge/Document%20AI-LayoutLMv3%20Compatible-green.svg)](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)

A high-performance, columnar **Apache Parquet** dataset containing digitized, OCR-extracted historical American newspapers from the **US Library of Congress Chronicling America / National Digital Newspaper Program (NDNP)**.

Produced by streaming and transmuting massive Library of Congress preservation archives (`.tar.bz2`, METS/MODS, and ALTO XML) into compact, issue-level Parquet shards partitioned by **state**, **newspaper title**, and **publication year**.

---

## 🌟 Key Features

* **Issue-Level Granularity**: Each `.parquet` file represents one complete newspaper issue, preserving page sequence, front-page primacy, and full issue integrity.
* **Normalized Spatial Coordinates ($[0, 1000]$)**: Word-level bounding boxes are pre-normalized to the standard $[0, 1000]$ coordinate space, ready for out-of-the-box use with **LayoutLMv3**, **LiLT**, and Vision-Language models.
* **Reading-Order Geometry**: Preserves physical layout structures—words, lines, and article column blocks—resolving the historical "text scramble" caused by multi-column printing.
* **Zero-Copy Remote Querying**: Query petabyte-scale historical archives directly in seconds using **DuckDB**, **Polars**, or **Hugging Face `datasets`** without downloading entire collections.
* **Master Discovery Catalog**: Includes `catalog.parquet` (~5 MB), an instant index of every newspaper title, city, state, date range, page count, and demographic classification.
* **Vector Search Ready**: Direct conversion pipelines available for the **Pinecone Document Schema** to enable hybrid semantic/lexical RAG over 200+ years of American history.

---

## 📂 Dataset Architecture

The repository uses a hierarchical, year-partitioned layout designed to scale effortlessly across millions of issues while respecting Git tree limits:

```text
newspapers/
└── {state}/
    └── {newspaper_slug}/
        └── {year}/
            └── {state}_{newspaper_slug}_{year}_{month}_{day}.parquet
```

* **`catalog.parquet`**: Master index table at the root directory containing metadata for all indexed newspapers and issues.
* **`newspapers/{state}/{newspaper_slug}/{year}/*.parquet`**: Columnar Snappy-compressed Parquet files where each file contains all pages of a specific publication date.

---

## 📋 Data Schema

Every page row conforms to the following schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `_id` | `string` | Canonical record ID: `{lccn}_{date}_ed-{edition}_seq-{sequence}` |
| `text` | `string` | Full plain text OCR for the page in reading order |
| `title` | `string` | Human-readable page title (e.g. *The Monitor, 1915-07-03 - Page 1*) |
| `newspaper_title` | `string` | Clean publication name (e.g. *The Monitor*) |
| `newspaper_slug` | `string` | URL-safe directory slug (e.g. *the_monitor*) |
| `lccn` | `string` | Library of Congress Control Number (e.g. *00225879*) |
| `date` | `string` | Publication date (`YYYY-MM-DD`) |
| `year`, `month`, `day` | `int32` | Integer date components for fast numeric partition pruning |
| `edition` | `string` | Issue edition identifier (e.g. `1`) |
| `sequence` | `int32` | 1-indexed page sequence within the issue |
| `city`, `state` | `string` | Geographic origin of publication |
| `ethnicity` | `string` | Subject ethnicity if cataloged (e.g. *African American*, *German*) |
| `char_count`, `word_count` | `int32` | Text volume statistics |
| `loc_page_url` | `string` | Live Library of Congress interactive page viewer URL (`loc.gov/resource/.../?sp=...`) |
| `loc_item_url` | `string` | Canonical persistent Library of Congress catalog item record URL (`loc.gov/item/.../`) |
| `pdf_url` | `string` | Direct page PDF download URL on `chroniclingamerica.loc.gov/data/batches/...` |
| `image_url` | `string` | Direct master JP2 image download URL on `chroniclingamerica.loc.gov/data/batches/...` |
| `source_batch` | `string` | Exact NDNP source batch identifier for provenance (e.g. `nbu_indescribablebeast_ver01`) |
| `awardee` | `string` | Digitizing institution (e.g. *University of Nebraska-Lincoln*) |
| `awardee_code` | `string` | Awardee prefix code (e.g. `nbu`, `vi`, `iune`) |
| `reel_id` | `string` | Microfilm container / reel ID |
| `ocr_engine` | `string` | OCR software and version recorded in METS (e.g. *ABBYY FineReader / apex-alto 2.0*) |
| `page_width`, `page_height`| `int32` | Physical scan dimensions in original pixels |
| `page_unit` | `string` | Scanner measurement unit (e.g. *inch1200*) |
| `words` | `list<string>` | OCR word tokens in logical reading order |
| `boxes` | `list<list<int16>>` | Word bounding boxes `[x0, y0, x1, y1]` normalized to $[0, 1000]$ integer space |
| `word_confidences` | `list<float32>` | Word OCR confidence scores ($0.0$ to $1.0$) |
| `lines` | `list<struct>` | Text line structures with `box: [x0, y0, x1, y1]` and `text` |
| `blocks` | `list<struct>` | Column / article blocks with `block_id`, `box`, and `text` |

---

## 💻 Code Recipes

### 1. Streaming with Hugging Face `datasets`

Stream issues on-demand without downloading the entire multi-gigabyte dataset:

```python
from datasets import load_dataset

# Stream newspaper pages for a specific state
dataset = load_dataset(
    "{repo_id}",
    "nebraska",
    streaming=True,
    split="train",
)

for sample in dataset.take(3):
    print(f"Publication : {sample['newspaper_title']} ({sample['date']})")
    print(f"Page        : {sample['sequence']} of issue")
    print(f"Headline    : {sample['lines'][0]['text'] if sample['lines'] else 'N/A'}")
    print(f"Sample Text : {sample['text'][:200]}...")
    print("-" * 60)
```

### 2. Zero-Copy Remote SQL with DuckDB

Query across millions of historical pages directly over HTTP/S3 with full predicate pushdown:

```python
import duckdb

con = duckdb.connect()

# Query across all California issues for 1906 San Francisco earthquake coverage
query = \"\"\"
SELECT 
    newspaper_title,
    date,
    city,
    substring(text, 1, 200) AS excerpt
FROM 'hf://datasets/{repo_id}/newspapers/california/**/*.parquet'
WHERE year = 1906 
  AND text ILIKE '%earthquake%'
ORDER BY date ASC
LIMIT 10;
\"\"\"

df = con.execute(query).df()
print(df)
```

### 3. High-Speed Columnar Analysis with Polars

Scan partitioned Parquet files directly with Polars for sub-second timeline aggregations:

```python
import polars as pl

# Scan dataset lazily using glob expressions
q = (
    pl.scan_parquet("hf://datasets/{repo_id}/newspapers/nebraska/**/*.parquet")
    .filter(pl.col("year") >= 1900)
    .group_by("year")
    .agg([
        pl.count().alias("page_count"),
        pl.col("word_count").sum().alias("total_words"),
    ])
    .sort("year")
)

df = q.collect()
print(df)
```

### 4. Document AI: LayoutLMv3 & Vision-Language Modeling

Every record includes pre-normalized $[0, 1000]$ integer bounding boxes, ready for direct ingestion by LayoutLMv3:

```python
from transformers import LayoutLMv3Processor

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Assuming 'sample' is a row from the dataset:
words = sample["words"][:512]
boxes = sample["boxes"][:512]

# Format inputs for model forward pass
encoding = processor(
    images=None,  # Or pass PIL image downloaded from sample['image_url']
    text=words,
    boxes=boxes,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

print("Input IDs shape:", encoding["input_ids"].shape)
print("BBoxes shape   :", encoding["bbox"].shape)
```

### 5. Vector Search with Pinecone Bulk Import

Convert Parquet files into standard Pinecone Document Schema JSONL files using the included conversion script:

```bash
python examples/07_convert_parquet_to_pinecone.py \\
    --input ./export_data/newspapers/nebraska/ \\
    --output ./pinecone_import/ \\
    --chunk-lines 100000
```

Upload `pinecone_import/` to cloud storage (S3/GCS/Azure) and trigger Pinecone Bulk Import:

```python
from pinecone import Pinecone, ImportErrorMode

pc = Pinecone()
index = pc.Index(host="YOUR_INDEX_HOST")

# Start asynchronous serverless bulk import
import_job = index.start_import(
    uri="s3://my-bucket/pinecone_import/",
    error_mode=ImportErrorMode.CONTINUE,
)
print(f"Bulk Import Job Started: {import_job.id}")
```

---

## 🏛️ Source & Attribution

Data digitized and curated by the **Library of Congress** and the **National Endowment for the Humanities (NEH)** under the **National Digital Newspaper Program (NDNP)**.

* **Collection Homepage**: [https://chroniclingamerica.loc.gov/](https://chroniclingamerica.loc.gov/)
* **Copyright & Rights**: Historical newspapers digitized under NDNP were published prior to 1963 and are in the **Public Domain**. United States government contributions are not subject to copyright protection under Title 17 U.S.C. § 105.
* **Pipeline Source Code**: [https://github.com/tcondello/loc-chronicling-america](https://github.com/tcondello/loc-chronicling-america) (MIT License).

---

## 📖 Citation

If you use this dataset or tooling in your research, blog posts, or applications, please cite:

```bibtex
@dataset{chronicling_america_parquet_2026,
  author       = {Condello, Tim},
  title        = {Chronicling America: Nationwide Partitioned Apache Parquet Newspaper Dataset},
  year         = {2026},
  publisher    = {Hugging Face},
  version      = {1.0.0},
  url          = {https://huggingface.co/datasets/Tim-Pinecone/LOC-Chronicling-America}
}
```
"""


class HuggingFaceDatasetManager:
    """Manages creation, dataset cards, and uploading to the Hugging Face Hub."""

    def __init__(self, repo_id: str, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self._fs = None

    @property
    def fs(self):
        """Lazy-loaded HfFileSystem instance."""
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def file_exists(self, path_in_repo: str) -> bool:
        """Check if a file exists in the Hugging Face repository."""
        full_path = f"datasets/{self.repo_id}/{path_in_repo.lstrip('/')}"
        try:
            return bool(self.fs.exists(full_path))
        except Exception:
            return False


    def ensure_repo_exists(self, private: bool = False) -> None:
        """Create the Hugging Face dataset repository if it does not already exist."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or access Hugging Face repo '{self.repo_id}': {e}") from e

    def generate_readme(
        self,
        output_path: Path | str,
        states: Optional[List[str]] = None,
    ) -> Path:
        """Generate and save the README.md dataset card with Data Studio configs."""
        state_config_lines = []
        if states:
            for st in sorted(states):
                clean_st = st.lower().replace(" ", "-")
                state_config_lines.append(f"  - config_name: {clean_st}\n    data_files: \"newspapers/{clean_st}/**/*.parquet\"")

        state_configs_str = "\n".join(state_config_lines)
        content = DATASET_CARD_TEMPLATE.replace("{repo_id}", self.repo_id)
        content = content.replace("{state_configs}", state_configs_str)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def upload_file(
        self,
        local_path: Path | str,
        path_in_repo: str,
        commit_message: Optional[str] = None,
    ) -> None:
        """Upload a single file to the Hugging Face dataset repository."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        msg = commit_message or f"Upload {path_in_repo}"
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=msg,
        )

    def upload_folder(
        self,
        folder_path: Path | str,
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> None:
        """Upload an entire directory to the Hugging Face dataset repository."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        msg = commit_message or "Upload dataset shards"
        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=msg,
        )

    def upload_files_atomic(
        self,
        local_files: List[Tuple[Path, str]],
        commit_message: Optional[str] = None,
    ) -> None:
        """Upload multiple files in a single atomic Git commit on Hugging Face."""
        if not local_files:
            return
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=self.token)
        operations = [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path))
            for local_path, path_in_repo in local_files
        ]
        msg = commit_message or f"Upload {len(operations)} dataset shards"
        api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=msg,
        )
