"""Hugging Face Hub dataset helper for Chronicling America Parquet exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


DATASET_CARD_TEMPLATE = """---
license: mit
task_categories:
- text-retrieval
- fill-mask
tags:
- history
- newspapers
- ocr
- chronicling-america
- library-of-congress
- pinecone
- digital-humanities
- us-history
size_categories:
- 10M<n<100M
configs:
  - config_name: default
    data_files: "newspapers/*/*/*.parquet"
{state_configs}
---

# Chronicling America (US Library of Congress) - Parquet Dataset

This dataset contains digitized, OCR-extracted historical American newspapers from the **US Library of Congress Chronicling America / National Digital Newspaper Program (NDNP)**, partitioned into high-performance, columnar **Apache Parquet** files.

## Interactive Dataset Studio / Viewer

This dataset is fully viewable directly within the **Hugging Face Dataset Viewer / Data Studio**. Use the subset dropdown above to filter by state or explore the entire collection.

## Dataset Structure

The dataset is partitioned by media type, state, and publication:

```text
newspapers/
└── {state}/
    └── {newspaper_slug}/
        └── {state}_{newspaper_slug}_{city_slug}_{year}.parquet
```

* **`catalog.parquet`**: A lightweight master table (~5 MB) indexing every newspaper title, city, state, publication years, page counts, and demographics for instant discovery.
* **`newspapers/{state}/{newspaper_slug}/{state}_{newspaper_slug}_{year}_{month}_{day}.parquet`**: Columnar Snappy-compressed Parquet files where each file contains all pages of one issue, with complete plain text OCR and rich bibliographic metadata.

## Document Schema

Each Parquet record contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `_id` | `string` | Unique record ID: `{lccn}_{date}_ed-{edition}_seq-{sequence}` |
| `text` | `string` | Full plain text OCR for the page |
| `title` | `string` | Human-readable page title (e.g. *The Monitor, 1915-07-03 - Page 1*) |
| `newspaper_title`| `string` | Clean publication name (e.g. *The Monitor*) |
| `newspaper_slug` | `string` | URL-safe underscore slug (e.g. *the_monitor*) |
| `lccn` | `string` | Library of Congress Control Number |
| `date` | `string` | Publication date (`YYYY-MM-DD`) |
| `year`, `month`, `day` | `int32` | Date components for numeric filtering |
| `edition` | `string` | Edition identifier |
| `sequence` | `int32` | Page number within the issue |
| `city`, `state` | `string` | Geographic origin |
| `ethnicity` | `string` | Subject ethnicity if recorded (e.g. African American) |
| `char_count`, `word_count` | `int32` | Text length statistics |
| `loc_page_url` | `string` | Live Library of Congress interactive page viewer URL |
| `loc_item_url` | `string` | Alias to canonical page viewer URL |
| `pdf_url` | `string` | Direct Library of Congress page PDF download URL |
| `image_url` | `string` | Direct high-resolution scan (JP2) URL |
| `source_batch` | `string` | Exact NDNP source batch identifier for provenance |
| `awardee` | `string` | Grantee/institution (e.g. *Library of Virginia*) |
| `awardee_code` | `string` | Awardee prefix code (e.g. *vi*, *nbu*, *iune*) |
| `reel_id` | `string` | Microfilm container / reel ID |
| `ocr_engine` | `string` | OCR software name and version (e.g. *Tesseract 5.4.1*) |
| `page_width`, `page_height` | `int32` | Physical scan dimensions |
| `page_unit` | `string` | Scanner measurement unit (e.g. *inch1200*) |
| `words` | `list<string>` | OCR word tokens in logical reading order |
| `boxes` | `list<list<int16>>` | Word bounding boxes `[x0, y0, x1, y1]` normalized to `[0, 1000]` |
| `word_confidences` | `list<float32>` | Word OCR confidence scores (0.0 to 1.0) |
| `lines` | `list<struct>` | Text lines with normalized `box` and `text` |
| `blocks` | `list<struct>` | Column/Article blocks with `block_id`, `box`, and `text` |

## Quickstart: Loading in Python

```python
from datasets import load_dataset

# 1. Stream all newspapers from a state (e.g. Nebraska)
dataset = load_dataset(
    "{repo_id}",
    "nebraska",
    streaming=True
)

for doc in dataset["train"].take(5):
    print(doc["_id"], doc["newspaper_title"], doc["date"])
    print(doc["text"][:150])
```

### Document AI & LayoutLM Usage

Every page contains normalized `[0, 1000]` word bounding boxes and column blocks, directly compatible with Hugging Face `LayoutLMv3`, `LiLT`, and Vision-Language models:

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", "nebraska", streaming=True)
sample = next(iter(dataset["train"]))

print("Words:", sample["words"][:5])
print("Boxes (0-1000 scale):", sample["boxes"][:5])
print("Newspaper Columns (Blocks):", len(sample["blocks"]))
```

### Direct Parquet Querying with DuckDB

```python
import duckdb

# Query across all California newspapers without downloading the full dataset
con = duckdb.connect()
df = con.execute(\"\"\"
    SELECT newspaper_title, date, text
    FROM 'hf://datasets/{repo_id}/newspapers/california/*/*.parquet'
    WHERE year = 1906 AND text ILIKE '%earthquake%'
    LIMIT 10
\"\"\").df()
print(df)
```

## Importing into Pinecone

To import into Pinecone, use the included conversion utility `examples/parquet_to_pinecone.py` to produce standard Pinecone Document Schema JSONL files:

```bash
python examples/parquet_to_pinecone.py --input newspapers/nebraska/ --output pinecone_import/
```

Sync `pinecone_import/` to cloud storage (S3/GCS/Azure), then run Pinecone Bulk Import:

```python
from pinecone import Pinecone, ImportErrorMode

pc = Pinecone()
index = pc.Index(host="YOUR_INDEX_HOST")

index.start_import(
    uri="s3://my-bucket/chronicling-america/pinecone_import",
    error_mode=ImportErrorMode.CONTINUE
)
```

## Source & Attribution
Data digitized and curated by the **Library of Congress** and the **National Endowment for the Humanities** under the National Digital Newspaper Program (NDNP). Distributed in the public domain.
"""


class HuggingFaceDatasetManager:
    """Manages creation, dataset cards, and uploading to the Hugging Face Hub."""

    def __init__(self, repo_id: str, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")

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
                state_config_lines.append(f"  - config_name: {clean_st}\n    data_files: \"newspapers/{clean_st}/*/*.parquet\"")

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
