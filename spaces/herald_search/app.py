"""Modern, High-Performance Backend Server for Library of Congress Newspaper Search.

Modeled after Daniel van Strien's clean gr.Server + standalone index.html architecture.
Serves a dedicated HTML5/CSS3 frontend and exposes high-throughput JSON API endpoints
for Pinecone Documents API, Lucene full-text search, dense embeddings, and IIIF imagery.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure directory of app.py is always first in sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gradio as gr
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

import config
import highlight as hl
from iiif import normalize_iiif_url
import query as q
import story_data as sd

load_dotenv()
os.environ["GRADIO_SSR_MODE"] = "False"

# Hugging Face ZeroGPU validation no-op
try:
    import spaces

    @spaces.GPU
    def zero_gpu_noop():
        return True
except Exception:
    def zero_gpu_noop():
        return False

# Initialize backend clients (Pinecone & OpenAI)
index_client = None
openai_client = None
init_error = None

try:
    index_client, openai_client = q.get_clients()
except Exception as e:
    init_error = str(e)
    print(f"Backend client initialization error: {e}", file=sys.stderr)

# Initialize Gradio Server (FastAPI subclass with HF Spaces compatibility)
app = gr.Server(title="Coast-to-Coast & Nationwide Historical Newspaper Search (1890–1910)")
demo = app

MODES = ["Query string (Lucene)", "Hybrid", "Full-text (BM25)", "Semantic"]

MATCH_LABELS = {
    "None (off)": None,
    "All words — $match_all": "all",
    "Exact phrase — $match_phrase": "phrase",
    "Any word — $match_any": "any",
}

SHOWCASE_QUERIES = [
    {
        "category": "⚡ Nationwide Archive Comparison",
        "title": "1906 SF Earthquake: Eyewitness Reports vs. Nationwide Panic",
        "desc": "Compare how all 6 broadsheets across 4 regions reacted to the catastrophic April 1906 San Francisco earthquake. SF Call's front-line survival dispatches, LA's emergency relief trains, Chicago's telegraph bulletins, and NYC's financial shocks.",
        "query": '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
        "state": "All States",
        "newspaper_scope": "All 6 Newspapers (Nationwide Archive)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "🌲 Progressive Era Governance",
        "title": "Theodore Roosevelt: Western Forestry vs. NYC Machine Politics",
        "desc": "Examine divergent press focus on Roosevelt across 6 publications: Western broadsheets championed conservation, irrigation, and public lands, while New York tracked Tammany Hall battles and trust-busting.",
        "query": '+"Theodore Roosevelt" +(policy OR conservation OR trust OR campaign OR speech)',
        "state": "All States",
        "newspaper_scope": "All 6 Newspapers (Nationwide Archive)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "✈️ Technology & Urban Modernity",
        "title": "Aviation Dawn (1910): Dominguez Air Meet & Coastal Spectacles",
        "desc": "Track the birth of American aviation across all 6 broadsheets: Los Angeles's world-famous 1910 Dominguez Field Air Meet vs. NYC's Hudson-Fulton demonstration flights.",
        "query": '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude)',
        "state": "All States",
        "newspaper_scope": "All 6 Newspapers (Nationwide Archive)",
        "mode": "Query string (Lucene)",
        "years": ["1910"],
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "🦅 Midwestern Industrial Hubs",
        "title": "Chicago Labor Movement, Railroads & Great Strikes",
        "desc": "The Chicago Eagle covering Midwestern rail networks, union organizing, wage strikes, and municipal corruption during rapid industrial expansion.",
        "query": '+(labor OR strike OR railroad OR union OR wages) -advertisement',
        "state": "Illinois",
        "newspaper_scope": "Heartland & Midwest (Chicago Eagle & Beatrice Express)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "☀️ Metropolitan Press Ideology",
        "title": "NYC Broadsheet Feud: The Sun vs. Pulitzer's Evening World",
        "desc": "Compare Dana's orthodox, literary broadsheet (The Sun) against Pulitzer's crusading, tabloid-format Evening World on municipal scandals and Wall Street finance.",
        "query": '+(Tammany OR corruption OR "Wall Street" OR police OR trial)',
        "state": "New York",
        "newspaper_scope": "East Coast Comparison (Evening World & The Sun)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "🌁 California Regional Trade",
        "title": "Pacific Maritime Commerce: Bay Area vs. Los Angeles Harbor",
        "desc": "Contrast San Francisco Call's maritime dominance in Pacific trade routes with Los Angeles Herald's coverage of the San Pedro Free Harbor fight.",
        "query": '+(harbor OR shipping OR Pacific OR "San Francisco" OR trade)',
        "state": "California",
        "newspaper_scope": "West Coast Comparison (LA Herald & SF Call)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "🗳️ Social Movements & Suffrage",
        "title": "Women's Suffrage Across 4 States: Coast-to-Coast Rallies",
        "desc": "Follow suffragist campaigns across California, Illinois, Nebraska, and New York as western states led legislative victories preceding national enfranchisement.",
        "query": '+("woman suffrage" OR "equal suffrage") +(convention OR ballot OR vote OR parade)',
        "state": "All States",
        "newspaper_scope": "All 6 Newspapers (Nationwide Archive)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "🌾 Agrarian Frontier Economics",
        "title": "Great Plains Harvest, Wheat Prices & Nebraska Homesteads",
        "desc": "The Beatrice Daily Express tracking the agricultural economy, populist farmer politics, crop yields, and rural midwestern prairie life.",
        "query": '+(crops OR wheat OR corn OR farmer OR harvest OR populist)',
        "state": "Nebraska",
        "newspaper_scope": "The Beatrice Daily Express (Nebraska, Heartland)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Roosevelt Microfilm Typo (~1) Across All 6 Publications",
        "desc": "Microfilm ink bleed corrupts 'Roosevelt' to 'Roosevclt' (c for e). Fuzzy matching within 1 edit distance rescues corrupted historical records nationwide.",
        "query": "Roosevclt~1",
        "state": "All States",
        "newspaper_scope": "All 6 Newspapers (Nationwide Archive)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
]


def format_hit(hit: Any, query_str: str = "") -> Dict[str, Any]:
    """Normalize Pinecone Hit object or dictionary into clean JSON-serializable card record."""
    doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
    fields = getattr(hit, "fields", None) or doc
    score = float(getattr(hit, "score", getattr(hit, "_score", doc.get("_score", 0.0))))
    ns = getattr(hit, "namespace", str(int(fields.get("year", 1910))))

    date_str = str(fields.get("date", "Unknown Date"))
    ed_val = int(fields.get("edition", 1))
    seq_val = int(fields.get("sequence", 1))
    season_val = str(fields.get("season", "Unknown"))
    chunk_idx = int(fields.get("chunk_index", 0)) + 1
    total_chunks = int(fields.get("chunk_count", 1))
    raw_text = str(fields.get("text", ""))

    highlighted_html = (
        hl.highlight_text(raw_text, query_str)
        if query_str
        else html.escape(raw_text[:450]) + ("..." if len(raw_text) > 450 else "")
    )

    np_slug = str(fields.get("newspaper_slug", "los_angeles_herald"))
    meta = config.NEWSPAPERS.get(np_slug, {
        "title": "Historical Broadsheet",
        "city": "Unknown",
        "state_name": "Unknown",
        "lccn": "",
        "accent": "#475569",
        "bg": "#f8fafc",
        "border": "#cbd5e1",
        "icon": "📰",
    })

    file_path = fields.get("file_path") or fields.get("image_url", "")
    raw_thumb = fields.get("iiif_thumb_url") or ""
    thumb_url = normalize_iiif_url(raw_thumb or file_path, size="600,")
    high_res_url = normalize_iiif_url(file_path or raw_thumb, size="1600,")
    loc_page = str(fields.get("loc_page_url", ""))
    pdf_url = str(fields.get("pdf_url", ""))

    return {
        "id": getattr(hit, "id", doc.get("id", f"{meta.get('lccn')}_{date_str}_{seq_val}_{chunk_idx}")),
        "score": score,
        "namespace": ns,
        "date": date_str,
        "edition": ed_val,
        "sequence": seq_val,
        "season": season_val,
        "chunk_index": chunk_idx,
        "chunk_count": total_chunks,
        "char_count": len(raw_text),
        "raw_text": raw_text,
        "highlighted_html": highlighted_html,
        "newspaper_slug": np_slug,
        "newspaper_title": meta.get("title", np_slug),
        "newspaper_icon": meta.get("icon", "📰"),
        "newspaper_city": meta.get("city", ""),
        "newspaper_state": meta.get("state_name", ""),
        "newspaper_lccn": meta.get("lccn", ""),
        "accent": meta.get("accent", "#0284c7"),
        "bg": meta.get("bg", "#f0f9ff"),
        "border": meta.get("border", "#bae6fd"),
        "thumb_url": thumb_url,
        "high_res_url": high_res_url,
        "loc_page_url": loc_page,
        "pdf_url": pdf_url,
    }


def analyze_query_syntax(query_str: str, mode: str) -> List[Dict[str, str]]:
    """Parse Lucene token syntax and return structured breakdown pills."""
    chips: List[Dict[str, str]] = []
    if not query_str or mode != "Query string (Lucene)":
        return chips

    phrases = re.findall(r'"([^"]+)"(?!\~)', query_str)
    for p in phrases:
        chips.append({"type": "phrase", "label": f'Exact Phrase: "{p}"'})

    slop = re.findall(r'"([^"]+)"~(\d+)', query_str)
    for p, s in slop:
        chips.append({"type": "slop", "label": f'Proximity (~{s}): "{p}" within {s} words'})

    fuzzy = re.findall(r"([A-Za-z0-9]+)~(?:(\d+))?", query_str)
    for term, d in fuzzy:
        dist = d if d else "auto"
        chips.append({"type": "fuzzy", "label": f"Fuzzy (~{dist}): {term}"})

    boost = re.findall(r'([A-Za-z0-9_"]+)\^([0-9\.]+)', query_str)
    for term, b in boost:
        chips.append({"type": "boost", "label": f"Boosted (x{b}): {term}"})

    regex = re.findall(r"/([^/]+)/", query_str)
    for r in regex:
        chips.append({"type": "regex", "label": f"Regex: /{r}/"})

    return chips


# ── REST API Endpoints ───────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    """Serve the modern standalone HTML5 frontend."""
    index_file = HERE / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


def load_calendar_manifest() -> Dict[str, Any]:
    """Scan dataset chunks to map MM-DD to available years, newspapers, and issue counts."""
    from collections import defaultdict
    manifest = defaultdict(lambda: {"total_issues": 0, "years": set(), "newspapers": set()})
    data_dir = HERE / "data"
    if data_dir.exists():
        for f in data_dir.glob("*.jsonl"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        fields = rec.get("fields", rec)
                        d = fields.get("date")
                        if not d:
                            continue
                        parts = d.split("-")
                        if len(parts) != 3:
                            continue
                        yr, mo, da = parts[0], parts[1], parts[2]
                        key = f"{int(mo):02d}-{int(da):02d}"
                        np_slug = fields.get("newspaper_slug", "")
                        chunk_idx = fields.get("chunk_index", 0)
                        if chunk_idx == 0:
                            manifest[key]["total_issues"] += 1
                        manifest[key]["years"].add(yr)
                        if np_slug:
                            manifest[key]["newspapers"].add(np_slug)
            except Exception as e:
                print(f"Error reading {f} for calendar manifest: {e}")
    out = {}
    for k, v in manifest.items():
        out[k] = {
            "total_issues": v["total_issues"],
            "years": sorted(list(v["years"])),
            "newspapers": sorted(list(v["newspapers"])),
        }
    return out


CALENDAR_MANIFEST = load_calendar_manifest()


@app.get("/api/config")
def get_config():
    """Return archive configuration, newspaper metadata, years, and curated presets."""
    return JSONResponse({
        "status": "ok",
        "init_error": init_error,
        "total_articles": 46846,
        "newspapers": config.NEWSPAPERS,
        "target_years": config.TARGET_YEARS,
        "target_months": config.TARGET_MONTHS,
        "month_names": getattr(config, "MONTH_NAMES", {}),
        "calendar_manifest": CALENDAR_MANIFEST,
        "regions": config.REGIONS,
        "modes": MODES,
        "presets": SHOWCASE_QUERIES,
        "chapters": sd.CHAPTERS,
    })


@app.post("/api/search")
async def api_search(request: Request):
    """Execute search across up to 3 broadsheets, multi-year namespaces, and ranking modes."""
    if init_error or not index_client or not openai_client:
        return JSONResponse({
            "error": f"Search engine offline: {init_error or 'Missing PINECONE_API_KEY or OPENAI_API_KEY'}"
        }, status_code=503)

    data = await request.json()
    query_str = str(data.get("query", "")).strip()
    newspapers = data.get("newspapers") or []
    mode = str(data.get("mode", "Query string (Lucene)")).strip()
    years_val = data.get("years", "All")
    season = str(data.get("season", "(any)")).strip()
    top_k = int(data.get("top_k", 4))
    month = str(data.get("month", "(any)")).strip()

    if not query_str:
        return JSONResponse({"hits": [], "total_hits": 0, "metrics": {}, "newspaper_counts": {}, "year_np_counts": {}})

    # Resolve target broadsheets (capped at 3)
    target_slugs: List[str] = []
    if isinstance(newspapers, list):
        for s in newspapers:
            s_clean = str(s).strip()
            if s_clean in config.NEWSPAPERS and s_clean not in target_slugs:
                target_slugs.append(s_clean)
            elif not s_clean:
                continue
            else:
                for k, meta in config.NEWSPAPERS.items():
                    if meta["title"].lower() in s_clean.lower() or k in s_clean.lower():
                        if k not in target_slugs:
                            target_slugs.append(k)
                        break
    elif isinstance(newspapers, str) and newspapers.strip():
        s_clean = newspapers.strip()
        for k, meta in config.NEWSPAPERS.items():
            if meta["title"].lower() in s_clean.lower() or k in s_clean.lower():
                target_slugs.append(k)
                break

    target_slugs = target_slugs[:3]
    if not target_slugs:
        target_slugs = ["los_angeles_herald", "the_evening_world"]

    # Resolve namespaces
    if not years_val or years_val == "All" or years_val == ["All"]:
        target_namespaces = config.TARGET_YEARS
    elif isinstance(years_val, list):
        target_namespaces = [str(y).strip() for y in years_val if str(y).strip() in config.TARGET_YEARS]
    elif isinstance(years_val, str):
        if years_val.strip() in config.TARGET_YEARS:
            target_namespaces = [years_val.strip()]
        elif "," in years_val:
            target_namespaces = [y.strip() for y in years_val.split(",") if y.strip() in config.TARGET_YEARS]
        else:
            target_namespaces = config.TARGET_YEARS
    else:
        target_namespaces = config.TARGET_YEARS

    if not target_namespaces:
        target_namespaces = config.TARGET_YEARS

    query_chips = analyze_query_syntax(query_str, mode)

    base_filter = q.build_filter(
        month=month if month != "(any)" else None,
        season=season if season != "(any)" else None,
        front_page_only=False,
        match_op=None,
        match_terms=query_str,
    )

    if len(target_slugs) > 1:
        res = q.search_side_by_side(
            index=index_client,
            oai=openai_client,
            query=query_str,
            namespaces=target_namespaces,
            mode=mode,
            match_op=None,
            match_terms=query_str,
            top_k_per_year=top_k,
            base_filter_dict=base_filter,
            slugs=target_slugs,
        )

        by_newspaper: Dict[str, Any] = {}
        all_hits = []
        for slug in target_slugs:
            raw_hits = res.get("by_newspaper", {}).get(slug, {}).get("merged", [])
            formatted = [format_hit(h, query_str) for h in raw_hits]
            all_hits.extend(formatted)
            by_newspaper[slug] = {
                "slug": slug,
                "meta": config.NEWSPAPERS.get(slug, {}),
                "hits": formatted,
                "total_hits": len(formatted),
            }

        return JSONResponse({
            "is_comparative": True,
            "target_slugs": target_slugs,
            "by_newspaper": by_newspaper,
            "hits": all_hits,
            "total_hits": len(all_hits),
            "metrics": res.get("metrics", {}),
            "newspaper_counts": res.get("newspaper_counts", {}),
            "year_np_counts": res.get("year_np_counts", {}),
            "query_chips": query_chips,
        })

    # Single newspaper search
    slug = target_slugs[0]
    filt = q.build_filter(
        newspaper_slug=slug,
        month=month if month != "(any)" else None,
        season=season if season != "(any)" else None,
        front_page_only=False,
        match_op=None,
        match_terms=query_str,
    )

    search_data = q.search_cross_years(
        index=index_client,
        oai=openai_client,
        query=query_str,
        namespaces=target_namespaces,
        mode=mode,
        match_op=None,
        match_terms=query_str,
        top_k_per_year=top_k,
        filter_dict=filt,
    )

    raw_hits = search_data.get("merged", [])
    formatted_hits = [format_hit(h, query_str) for h in raw_hits]

    by_newspaper = {
        slug: {
            "slug": slug,
            "meta": config.NEWSPAPERS.get(slug, {}),
            "hits": formatted_hits,
            "total_hits": len(formatted_hits),
        }
    }

    return JSONResponse({
        "is_comparative": False,
        "target_slugs": target_slugs,
        "by_newspaper": by_newspaper,
        "hits": formatted_hits,
        "total_hits": search_data.get("total_hits", len(formatted_hits)),
        "metrics": search_data.get("metrics", {}),
        "newspaper_counts": search_data.get("newspaper_counts", {}),
        "year_np_counts": search_data.get("year_np_counts", {}),
        "query_chips": query_chips,
    })


@app.post("/api/story")
async def api_story(request: Request):
    """Execute dual-path hybrid story query across up to 3 selected broadsheets."""
    if init_error or not index_client or not openai_client:
        return JSONResponse({"error": f"Search engine offline: {init_error}"}, status_code=503)

    data = await request.json()
    chapter_idx = int(data.get("chapter_idx", 0))
    if chapter_idx < 0 or chapter_idx >= len(sd.CHAPTERS):
        chapter_idx = 0

    chapter = sd.CHAPTERS[chapter_idx]
    semantic_query = str(data.get("semantic_query", "")).strip() or chapter.get("semantic_query", "")
    fulltext_query = str(data.get("fulltext_query", "")).strip() or chapter.get("query", "")

    newspapers = data.get("newspapers") or []
    target_slugs: List[str] = []
    if isinstance(newspapers, list):
        for s in newspapers:
            s_clean = str(s).strip()
            if s_clean in config.NEWSPAPERS and s_clean not in target_slugs:
                target_slugs.append(s_clean)
            elif not s_clean:
                continue
            else:
                for k, meta in config.NEWSPAPERS.items():
                    if meta["title"].lower() in s_clean.lower() or k in s_clean.lower():
                        if k not in target_slugs:
                            target_slugs.append(k)
                        break

    target_slugs = target_slugs[:3]
    if not target_slugs:
        target_slugs = chapter.get("default_newspapers", ["los_angeles_herald", "the_evening_world"])

    top_k = int(data.get("top_k", 4))
    year_filter = str(data.get("year_filter", "All Years"))
    namespaces = [year_filter.strip()] if year_filter in config.TARGET_YEARS else config.TARGET_YEARS

    # Execute dual-path hybrid search with RRF fusion
    res = q.search_side_by_side(
        index=index_client,
        oai=openai_client,
        query=fulltext_query or semantic_query,
        namespaces=namespaces,
        mode="Hybrid (Dual-Path RRF)",
        match_op=None,
        match_terms=None,
        top_k_per_year=top_k,
        slugs=target_slugs,
        semantic_query=semantic_query,
        fulltext_query=fulltext_query,
    )

    by_newspaper: Dict[str, Any] = {}
    all_hits = []
    for slug in target_slugs:
        raw_hits = res.get("by_newspaper", {}).get(slug, {}).get("merged", [])
        formatted = [format_hit(h, fulltext_query) for h in raw_hits]
        all_hits.extend(formatted)
        by_newspaper[slug] = {
            "slug": slug,
            "meta": config.NEWSPAPERS.get(slug, {}),
            "hits": formatted,
            "total_hits": len(formatted),
        }

    return JSONResponse({
        "chapter": chapter,
        "chapter_idx": chapter_idx,
        "semantic_query": semantic_query,
        "fulltext_query": fulltext_query,
        "target_slugs": target_slugs,
        "by_newspaper": by_newspaper,
        "total_hits": len(all_hits),
        "metrics": res.get("metrics", {}),
        "newspaper_counts": res.get("newspaper_counts", {}),
        "year_np_counts": res.get("year_np_counts", {}),
    })


@app.post("/api/calendar")
async def api_calendar(request: Request):
    """Execute 'Explore by Date' calendar date lookup across turn-of-the-century years."""
    if init_error or not index_client:
        return JSONResponse({"error": f"Search engine offline: {init_error}"}, status_code=503)

    data = await request.json()
    month = int(data.get("month", 1))
    day = int(data.get("day", 1))
    year = str(data.get("year", "All")).strip()
    state = data.get("state")
    if state in ("All States", "All", "(all)", None):
        state = None
    newspaper_slug = data.get("newspaper_slug")
    if newspaper_slug in ("(all)", "all", "All Broadsheets", "", None):
        newspaper_slug = None
    front_page_only = bool(data.get("front_page_only", False))
    top_k = int(data.get("top_k", 8))

    namespaces = [year] if year in config.TARGET_YEARS else config.TARGET_YEARS

    res = q.search_calendar_date(
        index=index_client,
        month=month,
        day=day,
        namespaces=namespaces,
        front_page_only=front_page_only,
        state=state,
        newspaper_slug=newspaper_slug,
        top_k_per_year=top_k,
    )

    by_year_formatted: Dict[str, List[Dict[str, Any]]] = {}
    for yr, hits in res.get("by_year", {}).items():
        if hits:
            by_year_formatted[yr] = [format_hit(h, "") for h in hits]

    month_name = config.MONTH_NAMES.get(month, f"Month {month}")
    return JSONResponse({
        "by_year": by_year_formatted,
        "metrics": res.get("metrics", {}),
        "total_hits": res.get("total_hits", 0),
        "date_label": f"{month_name} {day:02d}",
        "month": month,
        "day": day,
        "year": year,
        "newspaper_counts": res.get("newspaper_counts", {}),
    })


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
