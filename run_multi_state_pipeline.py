#!/usr/bin/env python3
"""Entrypoint wrapper forwarding to scripts/run_multi_state_pipeline.py."""

import sys
from pathlib import Path

scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from scripts.run_multi_state_pipeline import main

if __name__ == "__main__":
    main()
