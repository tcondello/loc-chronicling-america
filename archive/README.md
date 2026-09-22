# Archive

This directory contains historical one-off migration and maintenance scripts that have served their primary operational purpose and are preserved here for reference.

---

## Archived Utilities

* **[`migrate_hf_to_year_partitions.py`](migrate_hf_to_year_partitions.py)**:
  Zero-download server-side migration script that reorganized 17,719 flat Parquet files (`newspapers/{state}/{newspaper}/*.parquet`) on Hugging Face into year-partitioned hierarchies (`newspapers/{state}/{newspaper}/{year}/*.parquet`) using `CommitOperationCopy` and `CommitOperationDelete`. Used to resolve the platform 10,000 files/directory ceiling on high-volume titles like *Omaha Daily Bee* and *The Beatrice Daily Express*.

* **[`cleanup_hf_dataset.py`](cleanup_hf_dataset.py)**:
  Administrative housekeeping script used during early development to prune test artifacts and reset SQLite task queue states.
