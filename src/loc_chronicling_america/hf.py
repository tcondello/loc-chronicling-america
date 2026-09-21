"""Hugging Face Hub dataset helper for Chronicling America Pinecone exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


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
---

# Chronicling America (US Library of Congress) - Pinecone Document Dataset

This dataset contains digitized, OCR-extracted historical American newspapers from the **US Library of Congress Chronicling America / National Digital Newspaper Program (NDNP)**, pre-formatted according to the **Pinecone Document Schema JSONL specification**.

## Dataset Structure

The dataset is partitioned hierarchically by **State**, **Newspaper Title**, and **Year**:

```text
data/
├── {state}/
│   └── {newspaper_slug}_{city_slug}_{lccn}/
│       ├── {year}.jsonl.gz
```

* **`catalog.parquet`**: A lightweight master table (~5 MB) indexing every newspaper title, city, state, publication years, page counts, and demographics for instant discovery.
* **`data/{state}/.../{year}.jsonl.gz`**: Gzip-compressed JSON Lines files where each line is one newspaper page ready for bulk import into Pinecone or streaming NLP pipelines.

## Document Schema (Pinecone Import Format)

Each JSONL document contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `_id` | `string` | Unique record ID: `{lccn}_{date}_ed-{edition}_seq-{sequence}` |
| `text` | `string` | Full plain text OCR for the page |
| `title` | `string` | Human-readable page title |
| `newspaper_title`| `string` | Base publication name (e.g. *The Monitor*) |
| `newspaper_slug` | `string` | URL-safe slug |
| `lccn` | `string` | Library of Congress Control Number |
| `date` | `string` | Publication date (`YYYY-MM-DD`) |
| `year`, `month`, `day` | `int` | Date components for numeric filtering |
| `city`, `state` | `string` | Geographic origin |
| `sequence` | `int` | Page number within the issue |
| `page_count` | `int` | Total pages in the issue |
| `char_count`, `word_count` | `int` | Text length statistics |
| `loc_item_url` | `string` | Canonical Library of Congress permalink |
| `image_url` | `string` | Full master scan IIIF URL |
| `pdf_url` | `string` | Single-page PDF URL |
| `source_batch` | `string` | Exact NDNP source batch identifier for provenance |

## Quickstart: Loading in Python

```python
from datasets import load_dataset

# 1. Stream all newspapers from a state (e.g. Nebraska)
dataset = load_dataset(
    "{repo_id}",
    data_files="data/nebraska/**/*.jsonl.gz",
    streaming=True
)

for doc in dataset["train"].take(5):
    print(doc["_id"], doc["newspaper_title"], doc["date"])
    print(doc["text"][:150])
```

## Importing into Pinecone

Sync your desired state or folder to cloud storage (S3/GCS/Azure), then run Pinecone Bulk Import:

```python
from pinecone import Pinecone, ImportErrorMode

pc = Pinecone()
index = pc.Index(host="YOUR_INDEX_HOST")

index.start_import(
    uri="s3://my-bucket/chronicling-america/data/nebraska",
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

    def generate_readme(self, output_path: Path | str) -> Path:
        """Generate and save the README.md dataset card."""
        content = DATASET_CARD_TEMPLATE.replace("{repo_id}", self.repo_id)
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
