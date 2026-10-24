"""Search Chronicling America Parquet on Hugging Face."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .explore_query import CompiledQuery, ExploreQueryError, compile_query, normalize_slug

REPO_ID = "Tim-Pinecone/LOC-Chronicling-America"
HF_ROOT = f"hf://datasets/{REPO_ID}"
_SLUG = re.compile(r"^[a-z0-9_]+$")
_LOCK = threading.Lock()
_CON = None


@dataclass
class Hit:
    newspaper_title: str
    city: str
    state: str
    date: str
    year: int
    month: int
    page: int
    url: str
    word_count: int
    readable: float
    excerpt: str


@dataclass
class PaperLayout:
    years: List[int]
    flat: bool


def connect():
    """Open a process-wide DuckDB connection with the Hugging Face filesystem."""
    global _CON
    if _CON is not None:
        return _CON
    try:
        import duckdb
    except ImportError as exc:
        raise ExploreQueryError(
            "DuckDB is not installed. From the repo root run: uv sync --extra dev"
        ) from exc
    _CON = duckdb.connect()
    _CON.execute("INSTALL httpfs; LOAD httpfs;")
    return _CON


def list_papers(state: str) -> List[str]:
    state_slug = _require_slug(normalize_slug(state), "state")
    from huggingface_hub import HfApi

    api = HfApi()
    kids = api.list_repo_tree(
        REPO_ID,
        repo_type="dataset",
        path_in_repo=f"newspapers/{state_slug}",
        recursive=False,
    )
    names = [item.path.rstrip("/").split("/")[-1] for item in kids]
    return sorted(name for name in names if name and not name.endswith(".parquet"))


def paper_layout(state: str, paper: str, *, refresh: bool = False) -> PaperLayout:
    state_slug = _require_slug(normalize_slug(state), "state")
    paper_slug = _require_slug(normalize_slug(paper), "paper")
    cache = _load_cache()
    key = f"{state_slug}/{paper_slug}"
    if not refresh and key in cache:
        entry = cache[key]
        return PaperLayout(list(entry["years"]), bool(entry["flat"]))

    from huggingface_hub import HfApi

    api = HfApi()
    kids = list(
        api.list_repo_tree(
            REPO_ID,
            repo_type="dataset",
            path_in_repo=f"newspapers/{state_slug}/{paper_slug}",
            recursive=False,
        )
    )
    years: List[int] = []
    flat = False
    for item in kids:
        name = item.path.rstrip("/").split("/")[-1]
        if name.endswith(".parquet"):
            flat = True
        elif name.isdigit():
            years.append(int(name))
    if not years and not flat:
        raise ExploreQueryError(f"No issues found at newspapers/{state_slug}/{paper_slug}.")
    layout = PaperLayout(sorted(set(years)), flat)
    cache[key] = {"years": layout.years, "flat": layout.flat}
    _save_cache(cache)
    return layout


def search_pages(
    query: str,
    *,
    state: str,
    paper: str,
    year_from: int,
    year_to: int,
    readable: bool = True,
    limit: int = 30,
    progress: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> tuple[List[Hit], str]:
    """Run a Lucene-like query across the requested years.

    Returns the hits and a one-line explanation of the compiled predicate.
    """
    compiled = compile_query(query)
    state_slug = _require_slug(compiled.state or normalize_slug(state), "state")
    paper_slug = _require_slug(compiled.paper or normalize_slug(paper), "paper")
    start, end = _intersect_years(compiled, year_from, year_to)
    layout = paper_layout(state_slug, paper_slug)
    if layout.flat:
        if progress:
            progress(f"Scanning {paper_slug} {start}-{end}")
        hits = _search_glob(
            compiled,
            path=f"{HF_ROOT}/newspapers/{state_slug}/{paper_slug}/*.parquet",
            year_from=start,
            year_to=end,
            readable=readable,
            limit=limit,
        )
        return hits, compiled.explanation

    chosen = [year for year in layout.years if start <= year <= end]
    if not chosen:
        raise ExploreQueryError(
            f"{paper_slug} has no issues from {start} to {end}. On disk: {layout.years[0]}-{layout.years[-1]}."
            if layout.years
            else f"{paper_slug} has no year folders in that range."
        )
    hits: List[Hit] = []
    for index, year in enumerate(chosen, start=1):
        if cancel is not None and cancel.is_set():
            break
        if progress:
            progress(f"Scanning {year} ({index}/{len(chosen)})")
        remaining = limit - len(hits)
        if remaining <= 0:
            break
        try:
            batch = _search_glob(
                compiled,
                path=f"{HF_ROOT}/newspapers/{state_slug}/{paper_slug}/{year}/*.parquet",
                year_from=year,
                year_to=year,
                readable=readable,
                limit=remaining,
            )
        except ExploreQueryError as exc:
            if "No files found" in str(exc) or "IO Error" in str(exc):
                continue
            raise
        hits.extend(batch)
    if progress:
        progress(f"{len(hits)} hit{'s' if len(hits) != 1 else ''} in {chosen[0]}-{chosen[-1]}")
    return hits, compiled.explanation


def _intersect_years(compiled: CompiledQuery, year_from: int, year_to: int) -> tuple[int, int]:
    if year_from > year_to:
        raise ExploreQueryError("From-year is after to-year.")
    start = year_from
    end = year_to
    if compiled.year_from is not None:
        start = max(start, compiled.year_from)
    if compiled.year_to is not None:
        end = min(end, compiled.year_to)
    if start > end:
        raise ExploreQueryError("The year filter does not overlap the years selected in the form.")
    return start, end


def _search_glob(
    compiled: CompiledQuery,
    *,
    path: str,
    year_from: int,
    year_to: int,
    readable: bool,
    limit: int,
) -> List[Hit]:
    extra_sql: List[str] = ["year BETWEEN ? AND ?"]
    extra_params: List[object] = [year_from, year_to]
    if compiled.month is not None:
        extra_sql.append("month = ?")
        extra_params.append(compiled.month)
    if compiled.day is not None:
        extra_sql.append("day = ?")
        extra_params.append(compiled.day)
    if compiled.page is not None:
        extra_sql.append("sequence = ?")
        extra_params.append(compiled.page)
    if compiled.city:
        extra_sql.append("lower(city) = lower(?)")
        extra_params.append(compiled.city)

    sql = f"""
    WITH matched AS (
      SELECT newspaper_title, city, state, CAST(date AS VARCHAR) AS date,
             year, month, day, sequence, loc_page_url, word_count, text
      FROM read_parquet('{path}')
      WHERE word_count >= 40
        AND ({compiled.where_sql})
        AND {" AND ".join(extra_sql)}
    ),
    scored AS (
      SELECT *,
        coalesce(
          nullif(regexp_extract(replace(text, chr(10), ' '), ?, 0, 'i'), ''),
          left(replace(text, chr(10), ' '), 240)
        ) AS excerpt
      FROM matched
    ),
    rated AS (
      SELECT *,
        length(regexp_extract_all(lower(excerpt), '\\b(the|and|of|to|was|that|in|for|with|his|her|she|he)\\b'))::DOUBLE
          / greatest(length(regexp_extract_all(lower(excerpt), '\\b[a-z]{{2,}}\\b')), 1) AS readable
      FROM scored
    )
    SELECT newspaper_title, city, state, date, year, month, sequence, loc_page_url,
           word_count, round(readable, 3), excerpt
    FROM rated
    WHERE (? = 0 OR readable >= 0.16)
    ORDER BY date, sequence
    LIMIT ?
    """
    params = [
        *compiled.params,
        *extra_params,
        compiled.excerpt_regex,
        0 if not readable else 1,
        int(limit),
    ]
    try:
        with _LOCK:
            rows = connect().execute(sql, params).fetchall()
    except Exception as exc:
        message = str(exc)
        if "No files found" in message or "HTTP" in message and "404" in message:
            return []
        raise ExploreQueryError(message) from exc
    hits: List[Hit] = []
    for row in rows:
        excerpt = re.sub(r"\s+", " ", row[10] or "").strip()
        hits.append(
            Hit(
                newspaper_title=row[0] or "",
                city=row[1] or "",
                state=row[2] or "",
                date=row[3] or "",
                year=int(row[4] or 0),
                month=int(row[5] or 0),
                page=int(row[6] or 0),
                url=row[7] or "",
                word_count=int(row[8] or 0),
                readable=float(row[9] or 0),
                excerpt=excerpt,
            )
        )
    return hits


def _require_slug(value: str, label: str) -> str:
    if not _SLUG.match(value or ""):
        raise ExploreQueryError(f"{label} must be a folder slug such as new_york or the_evening_world.")
    return value


def _cache_path() -> Path:
    path = Path.home() / ".cache" / "chronam"
    path.mkdir(parents=True, exist_ok=True)
    return path / "hf_layout.json"


def _load_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict) -> None:
    _cache_path().write_text(json.dumps(cache))
