#!/usr/bin/env python3
"""Convenience alias forwarding to 07_convert_parquet_to_pinecone.py."""

from pathlib import Path
import sys

examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

from importlib import import_module
convert_mod = import_module("07_convert_parquet_to_pinecone")

if __name__ == "__main__":
    convert_mod.main()
