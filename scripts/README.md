# Operational & Automation Scripts

This directory contains production automation, deployment helpers, and administrative maintenance utilities for the Chronicling America dataset pipeline.

---

## Available Scripts

### 1. `run_multi_state_pipeline.py`
The production driver script executed on dedicated worker instances (e.g. AWS EC2 via `systemd`). It orchestrates continuous, unattended batch transmutation and streaming uploads to Hugging Face across states.

**Key Features:**
* Self-healing retry logic for transient network/CDN drops.
* Zero-bloat scratch disk cleanup (immediately unlinks extracted raw `.tar.bz2` archives).
* Optional local Parquet purging (`--purge-local-after-upload`) to run endlessly on small disks (e.g. 50–100 GB EBS).
* Auto-syncs catalog metadata if run on a fresh machine.

**Usage:**
```bash
# Ingest specific states
python scripts/run_multi_state_pipeline.py --states NE NJ NY --purge-local-after-upload

# Ingest nationwide (all 50 states + territories)
python scripts/run_multi_state_pipeline.py --all-states --purge-local-after-upload
```

---

### 2. `cleanup_hf_dataset.py`
Administrative dataset utility for Hugging Face repository owners. Safely cleans legacy chunks, prunes obsolete directories, and resets local SQLite pipeline queue states.

**Usage:**
```bash
python scripts/cleanup_hf_dataset.py --repo "Tim-Pinecone/LOC-Chronicling-America"
```
