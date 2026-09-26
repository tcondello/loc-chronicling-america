#!/usr/bin/env python3
"""Batch prepare the remaining 3 newspapers into Pinecone-ready JSONL files."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT.parent.parent / ".venv" / "bin" / "python"

newspapers = [
    "the_beatrice_daily_express",
    "chicago_eagle",
    "the_san_francisco_call",
]

for np_slug in newspapers:
    print(f"\n{'='*70}\nStarting preparation for: {np_slug}\n{'='*70}", flush=True)
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "prepare.py"),
        "--newspaper", np_slug,
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Error preparing {np_slug}! Exit code: {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

print("\nAll 3 newspapers successfully prepared!")
