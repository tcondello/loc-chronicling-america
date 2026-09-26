#!/usr/bin/env python3
"""Batch ingest all prepared chunks files into Pinecone herald-hybrid-fts index."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT.parent.parent / ".venv" / "bin" / "python"
DATA_DIR = ROOT / "data"

newspapers = [
    "the_sun",
    "the_beatrice_daily_express",
    "chicago_eagle",
    "the_san_francisco_call",
]

for np_slug in newspapers:
    chunks_file = DATA_DIR / f"chunks_{np_slug}.jsonl"
    if not chunks_file.exists():
        print(f"Chunks file not found for {np_slug}: {chunks_file}, skipping.", flush=True)
        continue

    print(f"\n{'='*70}\nStarting ingestion for: {np_slug} ({chunks_file})\n{'='*70}", flush=True)
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "ingest.py"),
        "--chunks-file", str(chunks_file),
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Error ingesting {np_slug}! Exit code: {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

print("\nAll newspapers successfully ingested into Pinecone!")
