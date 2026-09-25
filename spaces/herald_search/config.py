"""Central configuration for the cross-year hybrid search pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHUNKS_NY_PATH = DATA_DIR / "chunks_ny.jsonl"
PREPARE_DONE_PATH = DATA_DIR / ".prepare_done"
PREPARE_NY_DONE_PATH = DATA_DIR / ".prepare_ny_done"
INGEST_OFFSET_PATH = DATA_DIR / ".ingest_offset"
INGEST_NY_OFFSET_PATH = DATA_DIR / ".ingest_ny_offset"

# ── Pinecone Configuration ──────────────────────────────────────────────────
INDEX_NAME = os.environ.get("PINECONE_INDEX", "herald-hybrid-fts")

# ── OpenAI Embeddings ───────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ── Ingest Batching ─────────────────────────────────────────────────────────
BATCH_SIZE = 50

# ── Chonkie Recursive Chunking ──────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 500
TOKENIZER = "cl100k_base"

# ── Source Dataset Scope (Hugging Face) ──────────────────────────────────────
HF_REPO = os.environ.get("HF_REPO", "Tim-Pinecone/LOC-Chronicling-America")

NEWSPAPERS = {
    "los_angeles_herald": {
        "title": "Los Angeles Herald",
        "slug": "los_angeles_herald",
        "state": "california",
        "state_name": "California",
        "city": "Los Angeles",
        "region": "West Coast",
        "icon": "🌴",
        "lccn": "sn85042462",
        "prefix": "herald",
        "path_prefix": "newspapers/california/los_angeles_herald",
    },
    "the_evening_world": {
        "title": "The Evening World",
        "slug": "the_evening_world",
        "state": "new_york",
        "state_name": "New York",
        "city": "New York",
        "region": "East Coast",
        "icon": "🗽",
        "lccn": "sn83030193",
        "prefix": "evening_world",
        "path_prefix": "newspapers/new_york/the_evening_world",
    },
}

# 9 years covering the turn-of-the-century (1890-1910)
TARGET_YEARS = [
    "1890",
    "1891",
    "1892",
    "1905",
    "1906",
    "1907",
    "1908",
    "1909",
    "1910",
]

# Dual season comparison window: Winter (January, December) + Summer (July & August)
TARGET_MONTHS = ["01", "07", "08", "12"]

SEASON_MAP = {
    "01": "Winter",
    1: "Winter",
    "07": "Summer",
    7: "Summer",
    "08": "Summer",
    8: "Summer",
    "12": "Winter",
    12: "Winter",
}
