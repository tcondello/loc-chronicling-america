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
        "region": "West Coast (Southern California)",
        "icon": "🌴",
        "lccn": "sn85042462",
        "prefix": "herald",
        "path_prefix": "newspapers/california/los_angeles_herald",
        "accent": "#d97706",
        "bg": "#fffbeb",
        "border": "#fde68a",
        "desc": "West Coast pioneer broadsheet • Southern California boom & transcontinental rail terminus",
    },
    "the_san_francisco_call": {
        "title": "The San Francisco Call",
        "slug": "the_san_francisco_call",
        "state": "california",
        "state_name": "California",
        "city": "San Francisco",
        "region": "West Coast (Bay Area)",
        "icon": "🌁",
        "lccn": "sn85066387",
        "prefix": "sfcall",
        "path_prefix": "newspapers/california/the_san_francisco_call",
        "accent": "#0d9488",
        "bg": "#f0fdfa",
        "border": "#99f6e4",
        "chunks": 9939,
        "pages": 382,
        "issues": 18,
        "desc": "Bay Area metropolis • Pacific maritime commerce, unionism & 1906 catastrophe aftermath",
    },
    "the_evening_world": {
        "title": "The Evening World",
        "slug": "the_evening_world",
        "state": "new_york",
        "state_name": "New York",
        "city": "New York",
        "region": "East Coast (New York)",
        "icon": "🗽",
        "lccn": "sn83030193",
        "prefix": "evening_world",
        "path_prefix": "newspapers/new_york/the_evening_world",
        "accent": "#0284c7",
        "bg": "#f0f9ff",
        "border": "#bae6fd",
        "desc": "Joseph Pulitzer's high-circulation NYC evening paper • Progressive crusades & vivid street reporting",
    },
    "the_sun": {
        "title": "The Sun",
        "slug": "the_sun",
        "state": "new_york",
        "state_name": "New York",
        "city": "New York",
        "region": "East Coast (New York)",
        "icon": "☀️",
        "lccn": "sn83030272",
        "prefix": "sun",
        "path_prefix": "newspapers/new_york/the_sun",
        "accent": "#ca8a04",
        "bg": "#fefce8",
        "border": "#fef08a",
        "chunks": 16205,
        "pages": 414,
        "issues": 27,
        "desc": "Legendary NYC broadsheet • Masterful editorial prose, Wall Street finance & national political balance",
    },
    "chicago_eagle": {
        "title": "Chicago Eagle",
        "slug": "chicago_eagle",
        "state": "illinois",
        "state_name": "Illinois",
        "city": "Chicago",
        "region": "Midwest (Great Lakes)",
        "icon": "🦅",
        "lccn": "sn84025828",
        "prefix": "eagle",
        "path_prefix": "newspapers/illinois/chicago_eagle",
        "accent": "#7c3aed",
        "bg": "#faf5ff",
        "border": "#ddd6fe",
        "chunks": 2439,
        "issues": 22,
        "desc": "Midwestern industrial hub • Railroad unionism, Pullman labor disputes & Chicago machine politics",
    },
    "the_beatrice_daily_express": {
        "title": "The Beatrice Daily Express",
        "slug": "the_beatrice_daily_express",
        "state": "nebraska",
        "state_name": "Nebraska",
        "city": "Beatrice",
        "region": "Midwest & Heartland (Great Plains)",
        "icon": "🌾",
        "lccn": "sn84020107",
        "prefix": "beatrice",
        "path_prefix": "newspapers/nebraska/the_beatrice_daily_express",
        "accent": "#16a34a",
        "bg": "#f0fdf4",
        "border": "#bbf7d0",
        "chunks": 1800,
        "desc": "Great Plains agricultural daily • The Western Populist revolt, prairie farm economies & Free Silver",
    },
}

# Regional Groups
REGIONS = {
    "all": ["los_angeles_herald", "the_san_francisco_call", "the_evening_world", "the_sun", "chicago_eagle", "the_beatrice_daily_express"],
    "west_coast": ["los_angeles_herald", "the_san_francisco_call"],
    "east_coast": ["the_evening_world", "the_sun"],
    "midwest": ["chicago_eagle", "the_beatrice_daily_express"],
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

# Exact Vector Counts Indexed in Pinecone (herald-hybrid-fts) across 9 Namespaces
NAMESPACE_STATS = {
    "1890": 3652,
    "1891": 3253,
    "1892": 4696,
    "1905": 7858,
    "1906": 4540,
    "1907": 4629,
    "1908": 4603,
    "1909": 5659,
    "1910": 7956,
}
TOTAL_INDEX_VECTORS = 46846

# All 12 calendar months supported
TARGET_MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

SEASON_MAP = {
    "01": "Winter", 1: "Winter",
    "02": "Winter", 2: "Winter",
    "03": "Spring", 3: "Spring",
    "04": "Spring", 4: "Spring",
    "05": "Spring", 5: "Spring",
    "06": "Summer", 6: "Summer",
    "07": "Summer", 7: "Summer",
    "08": "Summer", 8: "Summer",
    "09": "Fall", 9: "Fall",
    "10": "Fall", 10: "Fall",
    "11": "Fall", 11: "Fall",
    "12": "Winter", 12: "Winter",
}

