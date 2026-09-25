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
from typing import Any, Dict, List, Optional, Tuple

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


def embed_query(oai: OpenAI, text: str) -> List[float]:
    resp = oai.embeddings.create(model=config.EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def build_filter(
    newspaper_slug: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    season: Optional[str] = None,
    front_page_only: bool = False,
    match_op: Optional[str] = None,
    match_terms: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble a Pinecone filter dict combining metadata and optional lexical filters."""
    filt: Dict[str, Any] = {}

    if newspaper_slug and newspaper_slug not in ("(all)", "all", ""):
        filt["newspaper_slug"] = {"$eq": newspaper_slug}
    if year:
        filt["year"] = {"$eq": int(year)}
    if month:
        filt["month"] = {"$eq": int(month)}
    if season and season != "(any)":
        filt["season"] = {"$eq": season}
    if front_page_only:
        filt["sequence"] = {"$eq": 1}

    # Lexical hard filter on the text FTS field
    if match_op and match_terms and match_terms.strip():
        op = TEXT_MATCH_OPS.get(match_op, match_op)
        if op.startswith("$"):
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
):
    """Dense vector cosine ranking using OpenAI embeddings."""
    emb = embed_query(oai, query)
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
    match_op: str = "all",
    match_terms: Optional[str] = None,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
):
    """Hybrid search: OpenAI dense cosine ranking + lexical hard filter ($match_*)."""
    emb = embed_query(oai, query)
    score_by = [{"type": "dense_vector", "field": "embedding", "values": emb}]

    combined_filter = dict(filter_dict or {})
    terms = match_terms if match_terms else query
    op = TEXT_MATCH_OPS.get(match_op, match_op)
    if op.startswith("$") and terms.strip():
        combined_filter["text"] = {op: terms.strip()}

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
) -> Dict[str, Any]:
    """Execute concurrent queries across multiple year namespaces and group results.

    Returns:
        Dict with:
        - "by_year": { "1890": [hits...], "1910": [hits...] }
        - "merged": list of all hits sorted by date
        - "total_hits": int
    """
    results_by_year: Dict[str, List[Any]] = {}

    def query_single_namespace(ns: str):
        try:
            if mode == "Full-text (BM25)":
                res = search_text(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict)
            elif mode == "Query string (Lucene)":
                res = search_query_string(index, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict)
            elif mode == "Semantic":
                res = search_semantic(index, oai, query, namespace=ns, top_k=top_k_per_year, filter_dict=filter_dict)
            else:  # Hybrid
                res = search_hybrid(
                    index,
                    oai,
                    query,
                    namespace=ns,
                    match_op=match_op or "all",
                    match_terms=match_terms or query,
                    top_k=top_k_per_year,
                    filter_dict=filter_dict,
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(namespaces), 8)) as executor:
        future_map = {executor.submit(query_single_namespace, ns): ns for ns in namespaces}
        for future in concurrent.futures.as_completed(future_map):
            ns, hits = future.result()
            results_by_year[ns] = hits

    # Merge all hits chronologically
    merged = []
    for yr in sorted(results_by_year.keys()):
        merged.extend(results_by_year[yr])

    def sort_key(hit):
        doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
        fields = getattr(hit, "fields", None) or doc
        return (str(fields.get("date", "")), int(fields.get("sequence", 1)), int(fields.get("chunk_index", 0)))

    merged.sort(key=sort_key)

    return {
        "by_year": results_by_year,
        "merged": merged,
        "total_hits": len(merged),
    }


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
) -> Dict[str, Any]:
    """Execute paired searches across both newspapers:
    - West Coast: Los Angeles Herald
    - East Coast: The Evening World (New York)

    Returns:
        {
            "la": search_cross_years results for los_angeles_herald,
            "ny": search_cross_years results for the_evening_world,
        }
    """
    filter_la = dict(base_filter_dict or {})
    filter_la["newspaper_slug"] = {"$eq": "los_angeles_herald"}

    filter_ny = dict(base_filter_dict or {})
    filter_ny["newspaper_slug"] = {"$eq": "the_evening_world"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exec_both:
        fut_la = exec_both.submit(
            search_cross_years,
            index=index,
            oai=oai,
            query=query,
            namespaces=namespaces,
            mode=mode,
            match_op=match_op,
            match_terms=match_terms,
            top_k_per_year=top_k_per_year,
            filter_dict=filter_la,
        )
        fut_ny = exec_both.submit(
            search_cross_years,
            index=index,
            oai=oai,
            query=query,
            namespaces=namespaces,
            mode=mode,
            match_op=match_op,
            match_terms=match_terms,
            top_k_per_year=top_k_per_year,
            filter_dict=filter_ny,
        )
        res_la = fut_la.result()
        res_ny = fut_ny.result()

    return {
        "la": res_la,
        "ny": res_ny,
    }

