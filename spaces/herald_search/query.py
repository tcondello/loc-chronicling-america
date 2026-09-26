"""Stage 3: Query the hybrid FTS index across year namespaces.

Exposes the full Pinecone Document Schema search capabilities:
  - BM25 full-text ranking (score_by type 'text')
  - Lucene query_string ranking (score_by type 'query_string')
  - Semantic dense-vector ranking (score_by type 'dense_vector' via OpenAI)
  - Hybrid search: Semantic dense ranking + lexical hard filter ($match_*)
  - Multi-namespace parallel querying across year namespaces via ThreadPoolExecutor.
"""

import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

import config

load_dotenv()

FIELDS_TO_INCLUDE = [
    "text",
    "parent_page_id",
    "newspaper_title",
    "newspaper_slug",
    "region",
    "city",
    "state",
    "lccn",
    "date",
    "year",
    "month",
    "day",
    "season",
    "edition",
    "sequence",
    "chunk_index",
    "chunk_count",
    "token_count",
    "loc_page_url",
    "pdf_url",
    "image_url",
    "iiif_thumb_url",
]

TEXT_MATCH_OPS = {
    "phrase": "$match_phrase",
    "all": "$match_all",
    "any": "$match_any",
}


def get_clients() -> Tuple[Any, OpenAI]:
    api_key = os.environ.get("PINECONE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is missing.")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(config.INDEX_NAME)
    oai = OpenAI(api_key=openai_key)
    return index, oai


import json
import time

_EMBED_CACHE: Dict[str, List[float]] = {}
_SEARCH_CACHE: Dict[str, Any] = {}


def embed_query(oai: OpenAI, text: str) -> List[float]:
    """Generate dense embeddings with thread-safe in-memory LRU caching."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        return [0.0] * config.EMBED_DIM
    if clean_text in _EMBED_CACHE:
        return _EMBED_CACHE[clean_text]

    resp = oai.embeddings.create(model=config.EMBED_MODEL, input=[clean_text])
    emb = resp.data[0].embedding
    if len(_EMBED_CACHE) >= 512:
        # Evict oldest 128 items
        for k in list(_EMBED_CACHE.keys())[:128]:
            _EMBED_CACHE.pop(k, None)
    _EMBED_CACHE[clean_text] = emb
    return emb


def build_filter(
    state: Optional[str] = None,
    newspaper_slug: Optional[Union[str, List[str]]] = None,
    region: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[Any] = None,
    day: Optional[Any] = None,
    season: Optional[str] = None,
    front_page_only: bool = False,
    match_op: Optional[str] = None,
    match_terms: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble a Pinecone filter dict combining metadata and optional lexical filters."""
    filt: Dict[str, Any] = {}

    if region and region.lower() in config.REGIONS and region.lower() != "all":
        filt["newspaper_slug"] = {"$in": config.REGIONS[region.lower()]}
    elif state and state.strip().lower() not in ("all states", "all", "(all)", ""):
        s = state.strip().lower()
        if "california" in s:
            filt["newspaper_slug"] = {"$in": ["los_angeles_herald", "the_san_francisco_call"]}
        elif "new york" in s:
            filt["newspaper_slug"] = {"$in": ["the_evening_world", "the_sun"]}
        elif "nebraska" in s:
            filt["newspaper_slug"] = {"$eq": "the_beatrice_daily_express"}
        elif "illinois" in s:
            filt["newspaper_slug"] = {"$eq": "chicago_eagle"}

    if newspaper_slug:
        if isinstance(newspaper_slug, list):
            filt["newspaper_slug"] = {"$in": newspaper_slug}
        elif isinstance(newspaper_slug, str):
            ns_clean = newspaper_slug.strip()
            if ns_clean.lower() in config.REGIONS and ns_clean.lower() != "all":
                filt["newspaper_slug"] = {"$in": config.REGIONS[ns_clean.lower()]}
            elif ns_clean in config.NEWSPAPERS:
                filt["newspaper_slug"] = {"$eq": ns_clean}
            elif ns_clean not in ("(all)", "all", "Both Newspapers (Side-by-Side Comparison)", "All Newspapers (Nationwide)", "Nationwide Archive (All 6 Publications)", ""):
                # Substring match against registered titles or slugs
                matched_slug = None
                for slug, meta in config.NEWSPAPERS.items():
                    if meta["title"].lower() in ns_clean.lower() or slug in ns_clean.lower():
                        matched_slug = slug
                        break
                if matched_slug:
                    filt["newspaper_slug"] = {"$eq": matched_slug}

    if year:
        filt["year"] = {"$eq": int(year)}
    if month and str(month) not in ("(any)", "all", "(all)", ""):
        m_str = str(month).split()[0].lstrip("0") or "0"
        if m_str.isdigit() and int(m_str) > 0:
            filt["month"] = {"$eq": int(m_str)}
    if day and str(day) not in ("(any)", "all", "(all)", ""):
        d_str = str(day).split()[0].lstrip("0") or "0"
        if d_str.isdigit() and int(d_str) > 0:
            filt["day"] = {"$eq": int(d_str)}
    if season and season != "(any)":
        filt["season"] = {"$eq": season}
    if front_page_only:
        filt["sequence"] = {"$eq": 1}

    # Lexical hard filter on the text FTS field
    if match_op and match_terms and match_terms.strip():
        op = TEXT_MATCH_OPS.get(match_op, match_op)
        if op and op.startswith("$"):
            filt["text"] = {op: match_terms.strip()}

    return filt if filt else None


def search_text(
    index,
    query: str,
    namespace: str,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
):
    """BM25 token-OR full text ranking over the 'text' field."""
    score_by = [{"type": "text", "field": "text", "query": query}]
    return index.documents.search(
        namespace=namespace,
        score_by=score_by,
        top_k=top_k,
        filter=filter_dict,
        include_fields=FIELDS_TO_INCLUDE,
    )


def search_query_string(
    index,
    query: str,
    namespace: str,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
):
    """Raw Lucene query_string ranking (boolean / phrase / + / - / slop / boost)."""
    score_by = [{"type": "query_string", "query": query}]
    return index.documents.search(
        namespace=namespace,
        score_by=score_by,
        top_k=top_k,
        filter=filter_dict,
        include_fields=FIELDS_TO_INCLUDE,
    )


def search_semantic(
    index,
    oai: OpenAI,
    query: str,
    namespace: str,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
):
    """Dense vector cosine ranking using OpenAI embeddings."""
    emb = embedding if embedding is not None else embed_query(oai, query)
    score_by = [{"type": "dense_vector", "field": "embedding", "values": emb}]
    return index.documents.search(
        namespace=namespace,
        score_by=score_by,
        top_k=top_k,
        filter=filter_dict,
        include_fields=FIELDS_TO_INCLUDE,
    )


def search_hybrid(
    index,
    oai: OpenAI,
    query: str,
    namespace: str,
    match_op: Optional[str] = None,
    match_terms: Optional[str] = None,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
):
    """Hybrid search: OpenAI dense cosine ranking + optional lexical hard filter ($match_*)."""
    emb = embedding if embedding is not None else embed_query(oai, query)
    score_by = [{"type": "dense_vector", "field": "embedding", "values": emb}]

    combined_filter = dict(filter_dict or {})
    if match_op and match_op not in ("None (off)", "none", "None"):
        op = TEXT_MATCH_OPS.get(match_op, match_op)
        terms = match_terms.strip() if match_terms and match_terms.strip() else ""
        if op and op.startswith("$") and terms:
            combined_filter["text"] = {op: terms}

    return index.documents.search(
        namespace=namespace,
        score_by=score_by,
        top_k=top_k,
        filter=combined_filter if combined_filter else None,
        include_fields=FIELDS_TO_INCLUDE,
    )


def search_cross_years(
    index,
    oai: OpenAI,
    query: str,
    namespaces: List[str],
    mode: str = "Hybrid",
    match_op: Optional[str] = None,
    match_terms: Optional[str] = None,
    top_k_per_year: int = 5,
    filter_dict: Optional[Dict[str, Any]] = None,
    precomputed_embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Execute concurrent queries across multiple year namespaces and group results.

    Returns:
        Dict with:
        - "by_year": { "1890": [hits...], "1910": [hits...] }
        - "merged": list of all hits sorted by date
        - "total_hits": int
        - "year_counts": { "1890": count, ... }
        - "metrics": { "embed_ms": float, "pinecone_ms": float, "total_ms": float }
    """
    cache_key = f"cross_{query}_{mode}_{','.join(sorted(namespaces))}_{match_op}_{match_terms}_{top_k_per_year}_{json.dumps(filter_dict, sort_keys=True) if filter_dict else ''}"
    if cache_key in _SEARCH_CACHE:
        cached = dict(_SEARCH_CACHE[cache_key])
        cached["metrics"] = dict(cached["metrics"])
        cached["metrics"]["cached"] = True
        return cached

    t0 = time.time()
    embed_ms = 0.0
    emb = precomputed_embedding

    if mode in ("Semantic", "Hybrid") and emb is None:
        t_emb_0 = time.time()
        emb = embed_query(oai, query)
        embed_ms = (time.time() - t_emb_0) * 1000.0

    t_pc_0 = time.time()
    results_by_year: Dict[str, List[Any]] = {}

    def query_single_namespace(ns: str):
        try:
            if mode == "Full-text (BM25)":
                res = search_text(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict)
            elif mode == "Query string (Lucene)":
                res = search_query_string(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict)
            elif mode == "Semantic":
                res = search_semantic(index, oai, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict, embedding=emb)
            else:  # Hybrid
                res = search_hybrid(
                    index,
                    oai,
                    query,
                    namespace=ns,
                    match_op=match_op,
                    match_terms=match_terms,
                    top_k=top_k_per_year,
                    filter_dict=filter_dict,
                    embedding=emb,
                )
            matches = list(getattr(res, "matches", getattr(res, "hits", [])) or [])
            hits = []
            for m in matches:
                d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
                d["namespace"] = ns
                d["_score"] = getattr(m, "score", d.get("_score", 0.0))
                d["_id"] = getattr(m, "id", d.get("_id", ""))
                hits.append(d)
            return ns, hits
        except Exception as e:
            print(f"Error querying namespace {ns}: {e}")
            return ns, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(namespaces), 12)) as executor:
        future_map = {executor.submit(query_single_namespace, ns): ns for ns in namespaces}
        for future in concurrent.futures.as_completed(future_map):
            ns, hits = future.result()
            results_by_year[ns] = hits

    pinecone_ms = (time.time() - t_pc_0) * 1000.0
    total_ms = (time.time() - t0) * 1000.0

    # Merge all hits chronologically and group by publication
    merged = []
    by_newspaper: Dict[str, List[Any]] = {slug: [] for slug in config.NEWSPAPERS.keys()}
    newspaper_counts: Dict[str, int] = {slug: 0 for slug in config.NEWSPAPERS.keys()}
    year_np_counts: Dict[str, Dict[str, int]] = {yr: {slug: 0 for slug in config.NEWSPAPERS.keys()} for yr in namespaces}

    for yr in sorted(results_by_year.keys()):
        for h in results_by_year[yr]:
            merged.append(h)
            doc = h.to_dict() if hasattr(h, "to_dict") else (h if isinstance(h, dict) else {})
            fields = getattr(h, "fields", None) or doc
            s = fields.get("newspaper_slug") or "unknown"
            if s in by_newspaper:
                by_newspaper[s].append(h)
                newspaper_counts[s] += 1
            else:
                by_newspaper[s] = [h]
                newspaper_counts[s] = 1

            if yr in year_np_counts:
                if s in year_np_counts[yr]:
                    year_np_counts[yr][s] += 1
                else:
                    year_np_counts[yr][s] = 1

    def sort_key(hit):
        doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
        fields = getattr(hit, "fields", None) or doc
        return (str(fields.get("date", "")), int(fields.get("sequence", 1)), int(fields.get("chunk_index", 0)))

    merged.sort(key=sort_key)

    year_counts = {yr: len(results_by_year.get(yr, [])) for yr in namespaces}

    result = {
        "by_year": results_by_year,
        "by_newspaper": by_newspaper,
        "merged": merged,
        "total_hits": len(merged),
        "year_counts": year_counts,
        "newspaper_counts": newspaper_counts,
        "year_np_counts": year_np_counts,
        "metrics": {
            "embed_ms": round(embed_ms, 1),
            "pinecone_ms": round(pinecone_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }

    if len(_SEARCH_CACHE) >= 256:
        for k in list(_SEARCH_CACHE.keys())[:64]:
            _SEARCH_CACHE.pop(k, None)
    _SEARCH_CACHE[cache_key] = result
    return result


def search_side_by_side(
    index,
    oai: OpenAI,
    query: str,
    namespaces: List[str],
    mode: str = "Hybrid",
    match_op: Optional[str] = None,
    match_terms: Optional[str] = None,
    top_k_per_year: int = 5,
    base_filter_dict: Optional[Dict[str, Any]] = None,
    slugs: Optional[List[str]] = None,
    semantic_query: Optional[str] = None,
    fulltext_query: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute concurrent queries across multiple specified publications via a flat parallel pool.
    
    Supports dual-path hybrid search (dense semantic embedding + lexical query_string with RRF fusion)
    when both semantic_query and fulltext_query are provided.
    
    If slugs is omitted, defaults to comparing the two coastal anchors:
    - West Coast: Los Angeles Herald
    - East Coast: The Evening World (New York)

    Returns:
        {
            "la": search results for los_angeles_herald,
            "ny": search results for the_evening_world,
            "by_newspaper": { slug: { "by_year", "merged", "total_hits", "year_counts" } },
            "newspaper_counts": { slug: count },
            "all_merged": list of all hits across all queried publications,
            "total_hits": int,
            "metrics": { "embed_ms": float, "pinecone_ms": float, "total_ms": float }
        }
    """
    target_slugs = slugs if slugs else ["los_angeles_herald", "the_evening_world"]
    sem_q = semantic_query.strip() if semantic_query and semantic_query.strip() else query.strip()
    fts_q = fulltext_query.strip() if fulltext_query and fulltext_query.strip() else query.strip()
    is_dual_path = bool(semantic_query and fulltext_query) or (mode == "Hybrid (Dual-Path RRF)")

    cache_key = f"sbs_{query}_{sem_q}_{fts_q}_{mode}_{','.join(sorted(namespaces))}_{','.join(sorted(target_slugs))}_{match_op}_{match_terms}_{top_k_per_year}_{json.dumps(base_filter_dict, sort_keys=True) if base_filter_dict else ''}"
    if cache_key in _SEARCH_CACHE:
        cached = dict(_SEARCH_CACHE[cache_key])
        cached["metrics"] = dict(cached["metrics"])
        cached["metrics"]["cached"] = True
        return cached

    t0 = time.time()
    embed_ms = 0.0
    emb = None

    if is_dual_path or mode in ("Semantic", "Hybrid"):
        t_emb_0 = time.time()
        emb = embed_query(oai, sem_q)
        embed_ms = (time.time() - t_emb_0) * 1000.0

    t_pc_0 = time.time()

    # Flatten all tasks across all publications and year namespaces into a single pool
    tasks = []
    for slug in target_slugs:
        slug_filter = dict(base_filter_dict or {})
        slug_filter["newspaper_slug"] = {"$eq": slug}
        for ns in namespaces:
            tasks.append((slug, ns, slug_filter))

    def _query_task(target_slug: str, ns: str, filt: Dict[str, Any]):
        try:
            if is_dual_path:
                # Dual-Path Hybrid: Semantic Dense Search + Lucene FTS combined via RRF
                res_sem = search_semantic(index, oai, sem_q, namespace=ns, top_k=top_k_per_year * 2, filter_dict=filt, embedding=emb)
                res_fts = search_query_string(index, fts_q, namespace=ns, top_k=top_k_per_year * 2, filter_dict=filt)

                matches_sem = getattr(res_sem, "matches", []) or []
                matches_fts = getattr(res_fts, "matches", []) or []

                rrf_scores: Dict[str, float] = {}
                doc_map: Dict[str, Any] = {}

                for rank, m in enumerate(matches_sem):
                    mid = getattr(m, "id", getattr(m, "_id", ""))
                    rrf_scores[mid] = rrf_scores.get(mid, 0.0) + (1.0 / (60.0 + rank + 1))
                    doc_map[mid] = m

                for rank, m in enumerate(matches_fts):
                    mid = getattr(m, "id", getattr(m, "_id", ""))
                    rrf_scores[mid] = rrf_scores.get(mid, 0.0) + (1.0 / (60.0 + rank + 1))
                    if mid not in doc_map:
                        doc_map[mid] = m

                sorted_mids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k_per_year]
                hits = []
                for mid in sorted_mids:
                    m = doc_map[mid]
                    d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
                    d["namespace"] = ns
                    d["_score"] = rrf_scores[mid]
                    d["_id"] = mid
                    hits.append(d)
                return target_slug, ns, hits

            elif mode == "Full-text (BM25)":
                res = search_text(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filt)
            elif mode == "Query string (Lucene)":
                res = search_query_string(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filt)
            elif mode == "Semantic":
                res = search_semantic(index, oai, query, namespace=ns, top_k=top_k_per_year, filter_dict=filt, embedding=emb)
            else:  # Hybrid
                res = search_hybrid(
                    index,
                    oai,
                    query,
                    namespace=ns,
                    match_op=match_op,
                    match_terms=match_terms,
                    top_k=top_k_per_year,
                    filter_dict=filt,
                    embedding=emb,
                )
            matches = list(getattr(res, "matches", getattr(res, "hits", [])) or [])
            hits = []
            for m in matches:
                d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
                d["namespace"] = ns
                d["_score"] = getattr(m, "score", d.get("_score", 0.0))
                d["_id"] = getattr(m, "id", d.get("_id", ""))
                hits.append(d)
            return target_slug, ns, hits
        except Exception as e:
            print(f"Error querying {target_slug} in namespace {ns}: {e}")
            return target_slug, ns, []

    results_by_slug_by_year: Dict[str, Dict[str, List[Any]]] = {slug: {yr: [] for yr in namespaces} for slug in target_slugs}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(24, max(1, len(tasks)))) as executor:
        fut_map = {executor.submit(_query_task, target_slug, ns, filt): (target_slug, ns) for target_slug, ns, filt in tasks}
        for fut in concurrent.futures.as_completed(fut_map):
            target_slug, ns, hits = fut.result()
            results_by_slug_by_year[target_slug][ns] = hits

    pinecone_ms = (time.time() - t_pc_0) * 1000.0
    total_ms = (time.time() - t0) * 1000.0

    def sort_key(hit):
        doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
        fields = getattr(hit, "fields", None) or doc
        return (str(fields.get("date", "")), int(fields.get("sequence", 1)), int(fields.get("chunk_index", 0)))

    by_newspaper: Dict[str, Dict[str, Any]] = {}
    newspaper_counts: Dict[str, int] = {}
    all_merged = []

    for slug in target_slugs:
        slug_years = results_by_slug_by_year[slug]
        slug_merged = []
        for yr in sorted(slug_years.keys()):
            slug_merged.extend(slug_years[yr])
        slug_merged.sort(key=sort_key)
        all_merged.extend(slug_merged)
        counts = {yr: len(slug_years.get(yr, [])) for yr in namespaces}
        res_slug = {
            "by_year": slug_years,
            "merged": slug_merged,
            "total_hits": len(slug_merged),
            "year_counts": counts,
            "metrics": {"embed_ms": round(embed_ms, 1), "pinecone_ms": round(pinecone_ms, 1), "total_ms": round(total_ms, 1)},
        }
        by_newspaper[slug] = res_slug
        newspaper_counts[slug] = len(slug_merged)

    all_merged.sort(key=sort_key)

    empty_res = {"by_year": {yr: [] for yr in namespaces}, "merged": [], "total_hits": 0, "year_counts": {yr: 0 for yr in namespaces}}

    out = {
        "la": by_newspaper.get("los_angeles_herald", empty_res),
        "ny": by_newspaper.get("the_evening_world", empty_res),
        "by_newspaper": by_newspaper,
        "newspaper_counts": newspaper_counts,
        "all_merged": all_merged,
        "total_hits": len(all_merged),
        "metrics": {
            "embed_ms": round(embed_ms, 1),
            "pinecone_ms": round(pinecone_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }

    if len(_SEARCH_CACHE) >= 256:
        for k in list(_SEARCH_CACHE.keys())[:64]:
            _SEARCH_CACHE.pop(k, None)
    _SEARCH_CACHE[cache_key] = out
    return out


def search_calendar_date(
    index,
    month: int,
    day: int,
    namespaces: Optional[List[str]] = None,
    front_page_only: bool = False,
    state: Optional[str] = None,
    newspaper_slug: Optional[str] = None,
    top_k_per_year: int = 8,
) -> Dict[str, Any]:
    """Query year namespaces for newspaper issues published on a specific calendar month and day.

    Returns:
        {
            "by_year": { "1890": [hits...], ... },
            "by_newspaper": { "la": [hits...], "ny": [hits...] },
            "merged": [hits sorted chronologically],
            "total_hits": int,
            "metrics": { "pinecone_ms": float, "total_ms": float },
            "date_label": "MM-DD",
        }
    """
    import time
    t0 = time.time()
    target_namespaces = namespaces or config.TARGET_YEARS

    base_filter: Dict[str, Any] = {
        "month": {"$eq": int(month)},
        "day": {"$eq": int(day)},
    }
    if front_page_only:
        base_filter["sequence"] = {"$eq": 1}

    if state and state.strip().lower() not in ("all states", "all", "(all)", ""):
        s = state.strip().lower()
        if "california" in s:
            base_filter["newspaper_slug"] = {"$in": ["los_angeles_herald", "the_san_francisco_call"]}
        elif "new york" in s:
            base_filter["newspaper_slug"] = {"$in": ["the_evening_world", "the_sun"]}
        elif "nebraska" in s:
            base_filter["newspaper_slug"] = {"$eq": "the_beatrice_daily_express"}
        elif "illinois" in s:
            base_filter["newspaper_slug"] = {"$eq": "chicago_eagle"}
    elif newspaper_slug and newspaper_slug not in ("(all)", "all", ""):
        base_filter["newspaper_slug"] = {"$eq": newspaper_slug}

    results_by_year: Dict[str, List[Any]] = {}

    def query_ns(ns: str):
        try:
            res = index.documents.search(
                namespace=ns,
                score_by=[{"type": "query_string", "query": "*"}],
                top_k=top_k_per_year,
                filter=base_filter,
                include_fields=FIELDS_TO_INCLUDE,
            )
            matches = list(getattr(res, "matches", getattr(res, "hits", [])) or [])
            hits = []
            for m in matches:
                d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
                d["namespace"] = ns
                d["_score"] = getattr(m, "score", d.get("_score", 0.0))
                d["_id"] = getattr(m, "id", d.get("_id", ""))
                hits.append(d)
            return ns, hits
        except Exception as e:
            print(f"Error querying namespace {ns} for calendar date: {e}")
            return ns, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(target_namespaces), 9)) as executor:
        future_map = {executor.submit(query_ns, ns): ns for ns in target_namespaces}
        for future in concurrent.futures.as_completed(future_map):
            ns, hits = future.result()
            results_by_year[ns] = hits

    from collections import defaultdict
    merged = []
    hits_by_newspaper = defaultdict(list)

    for yr in sorted(results_by_year.keys()):
        for hit in results_by_year[yr]:
            merged.append(hit)
            slug = hit.get("newspaper_slug") or hit.get("fields", {}).get("newspaper_slug") or "unknown"
            hits_by_newspaper[slug].append(hit)

    def sort_key(hit):
        fields = getattr(hit, "fields", None) or hit
        return (str(fields.get("date", "")), int(fields.get("sequence", 1)), int(fields.get("chunk_index", 0)))

    merged.sort(key=sort_key)
    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "by_year": results_by_year,
        "by_newspaper": dict(hits_by_newspaper),
        "merged": merged,
        "total_hits": len(merged),
        "date_label": f"{int(month):02d}-{int(day):02d}",
        "metrics": {
            "pinecone_ms": round(elapsed_ms, 1),
            "total_ms": round(elapsed_ms, 1),
        },
    }


