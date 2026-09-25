"""Gradio Web Application for Coast-to-Coast Historical Newspaper Search (1890–1910).

Exposes Pinecone Documents API full-text search, semantic search, and hybrid search
with dynamic visual page rendering via Library of Congress IIIF Image API,
featuring side-by-side comparative analysis between:
- Los Angeles Herald (California, West Coast)
- The Evening World (New York, East Coast)
"""

import html
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
from dotenv import load_dotenv

import config
import highlight as hl
from iiif import jp2_to_iiif_url
import query as q

load_dotenv()

try:
    import spaces

    @spaces.GPU
    def zero_gpu_noop():
        """Satisfies Hugging Face ZeroGPU startup validator."""
        return True
except Exception:
    def zero_gpu_noop():
        return False

# ── Backend Client Initialization ───────────────────────────────────────────
index_client = None
openai_client = None
init_error = None

try:
    index_client, openai_client = q.get_clients()
except Exception as e:
    init_error = str(e)


MATCH_LABELS = {
    "None (off)": None,
    "All words — $match_all": "all",
    "Exact phrase — $match_phrase": "phrase",
    "Any word — $match_any": "any",
}

MODES = ["Query string (Lucene)", "Hybrid", "Full-text (BM25)", "Semantic"]

NEWSPAPER_OPTIONS = [
    "Both Newspapers (Side-by-Side Comparison)",
    "Los Angeles Herald (California, West Coast)",
    "The Evening World (New York, East Coast)",
]

YEAR_OPTIONS = ["All Years (Cross-Year Comparison)"] + config.TARGET_YEARS

# ── Curated Library of Congress Digital Strategy Showcase ────────────────────
SHOWCASE_QUERIES = [
    # ── East Coast vs. West Coast Historical Comparisons ──
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "1906 SF Earthquake: West Coast Panic vs. Wall Street Impact",
        "desc": "Compare how the two coasts responded to the catastrophic April 1906 earthquake. LA mobilized emergency supply trains, while New York covered Wall Street financial shocks and severed telegraph cables.",
        "query": '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Theodore Roosevelt: Western Conservation vs. NYC Politics",
        "desc": "Examine divergent press focus on Roosevelt: LA Herald championed western forestry and reclamation, while NYC Evening World tracked Tammany Hall political machines and Wall Street trusts.",
        "query": '+"Theodore Roosevelt" +(policy OR conservation OR trust OR campaign OR speech)',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Aviation Dawn (1908–1910): Dominguez Meet vs. Hudson Flights",
        "desc": "Captures the birth of American aviation across coasts: LA Herald's coverage of the landmark January 1910 Dominguez Field Air Meet vs. NYC Evening World's reports on Curtiss and Wright demonstration flights.",
        "query": '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude)',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Progressive Era Suffrage: California Enfranchisement vs. NYC Rallies",
        "desc": "Traces women's voting rights mobilization: Western states moving toward California's 1911 equal suffrage victory vs. New York City's growing mass parades on Fifth Avenue.",
        "query": '+("woman suffrage" OR "equal suffrage") +(convention OR ballot OR vote OR parade)',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "1890s Monetary Battles: Western Free Silver vs. Wall Street Gold Standard",
        "desc": "Contrast western populism favoring silver coinage in California against New York banking house orthodoxy and gold reserve defense during the 1890s financial crises.",
        "query": '+(currency OR silver OR "Wall Street") +(gold OR panic OR treasury)',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Halley\'s Comet May 1910: Mount Wilson Telescopes vs. Manhattan Rooftops",
        "desc": "Contrast Southern California's scientific observations atop Mount Wilson with Manhattan's rooftop vantage points and cyanogen tail anxieties.",
        "query": '("Halley\'s comet"~3 OR "tail of the comet"~2) +(astronomer OR observatory OR sky)',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "1910",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    # ── Lucene Grammar & Digital Strategy Archival Showcase ──
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Roosevelt Microfilm Typo (~1)",
        "desc": "Microfilm ink bleed corrupts 'Roosevelt' to 'Roosevclt' (c for e). Fuzzy matching within 1 edit distance rescues corrupted historical records across both newspapers.",
        "query": "Roosevclt~1",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Aviation Variant & Dropout (aeroplan~1)",
        "desc": "Recovers archaic spelling 'aeroplane', early American 'airplane', and OCR dropouts ('aeroplan') in early aviation news.",
        "query": "aeroplan~1",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Morphological Regex (/aeronaut.*/)",
        "desc": "Regular expression wildcard pattern matching across morphological variants: aeronaut, aeronautic, aeronautical, aeronauts.",
        "query": "text:/aeronaut.*/",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "📍 Historical Proximity & Column Co-Occurrence",
        "title": "1910 Dominguez Air Meet (Phrase Slop ~4)",
        "desc": "Solves multi-column page layout false positives by requiring 'aviation' and 'meet' to appear within 4 words of each other.",
        "query": '"aviation meet"~4',
        "newspaper_scope": "Los Angeles Herald (California, West Coast)",
        "mode": "Query string (Lucene)",
        "year": "1910",
        "season": "Winter",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "📍 Historical Proximity & Column Co-Occurrence",
        "title": "Wright Brothers Collaborative Coverage (~5)",
        "desc": "Pinpoints reporting treating Orville and Wilbur Wright as an inventive team within a tight 5-word phrasal proximity window.",
        "query": '"Wright brothers"~5',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "🏛️ Archival De-Noising & Commercial Exclusion",
        "title": "San Francisco Relief (Excluding Patent Ads)",
        "desc": "Isolates genuine civic relief efforts for San Francisco, explicitly stripping ubiquitous late-Victorian patent medicine ads.",
        "query": '+"San Francisco" +(earthquake OR catastrophe OR relief) -remedy -medicine -patent',
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "year": "All Years (Cross-Year Comparison)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
]

CUSTOM_CSS = """
mark.hl {
    background-color: #fde047;
    color: #000000;
    padding: 0.1em 0.3em;
    border-radius: 0.25em;
    font-weight: 600;
    border-bottom: 1.5px dotted #ca8a04;
    cursor: help;
}
.stat-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    background: #f1f5f9;
    color: #334155;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    border: 1px solid #e2e8f0;
}
.stat-pill.active {
    background: #e0f2fe;
    color: #0369a1;
    font-weight: 600;
    border-color: #bae6fd;
}
.syntax-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    border: 1px solid;
}
.quick-chip {
    font-family: monospace;
    font-size: 0.8rem;
    font-weight: 600;
}
.newspaper-split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 1rem;
}
@media (max-width: 960px) {
    .newspaper-split {
        grid-template-columns: 1fr;
    }
}
"""


def analyze_query_syntax(query_str: str, mode: str) -> str:
    """Produce a real-time visual breakdown of active Lucene operators in the query."""
    if not query_str or not query_str.strip():
        return ""
    if mode != "Query string (Lucene)":
        return f'<div style="font-size: 0.85rem; color: #64748b;">⚡ Ranking Mode: <strong>{mode}</strong></div>'

    chips = []
    # Check Required terms (+)
    req_matches = re.findall(r'\+(?:\"([^\"]+)\"|([A-Za-z0-9_]+))', query_str)
    req = [m[0] or m[1] for m in req_matches]
    if req:
        chips.append(f'<span class="syntax-badge" style="background:#dcfce7; color:#15803d; border-color:#86efac;">Required (+): {html.escape(", ".join(req))}</span>')

    # Check Excluded terms (-)
    exc_matches = re.findall(r'\-(?:\"([^\"]+)\"|([A-Za-z0-9_]+))', query_str)
    exc = [m[0] or m[1] for m in exc_matches]
    if exc:
        chips.append(f'<span class="syntax-badge" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">Excluded (-): {html.escape(", ".join(exc))}</span>')

    # Check Proximity slop ("..."~N)
    slop = re.findall(r'"([^"]+)"~(\d+)', query_str)
    if slop:
        for p, s in slop:
            chips.append(f'<span class="syntax-badge" style="background:#fef3c7; color:#b45309; border-color:#fde68a;">Proximity (~{s}): "{html.escape(p)}" within {s} words</span>')

    # Check Fuzzy terms (term~N)
    fuzzy = re.findall(r"([A-Za-z0-9]+)~(?:(\d+))?", query_str)
    if fuzzy:
        for term, d in fuzzy:
            dist = d if d else "auto"
            chips.append(f'<span class="syntax-badge" style="background:#ede9fe; color:#6d28d9; border-color:#ddd6fe;">Fuzzy (~{dist}): {html.escape(term)}</span>')

    # Check Boosted terms (term^N)
    boost = re.findall(r"([A-Za-z0-9_\"]+)\^([0-9\.]+)", query_str)
    if boost:
        for term, b in boost:
            chips.append(f'<span class="syntax-badge" style="background:#e0f2fe; color:#0369a1; border-color:#bae6fd;">Boosted (x{b}): {html.escape(term)}</span>')

    # Check Regex (/pattern.*/)
    regex = re.findall(r"/([^/]+)/", query_str)
    if regex:
        for r in regex:
            chips.append(f'<span class="syntax-badge" style="background:#fce7f3; color:#be185d; border-color:#fbcfe8;">Regex: /{html.escape(r)}/</span>')

    if not chips:
        chips.append('<span class="syntax-badge" style="background:#f1f5f9; color:#475569; border-color:#cbd5e1;">Standard Boolean / Term Disjunction</span>')

    return (
        '<div style="margin-top:0.4rem; padding:0.4rem 0.6rem; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; display:flex; flex-wrap:wrap; gap:0.25rem; align-items:center;">'
        '<span style="font-weight:700; font-size:0.8rem; color:#334155; margin-right:0.3rem;">⚡ Lucene Query Breakdown:</span> '
        + " ".join(chips)
        + "</div>"
    )


def hit_to_card_html(hit: Any, query_str: str) -> str:
    """Format an individual search match into an interactive visual HTML card."""
    doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
    fields = getattr(hit, "fields", None) or doc
    score = getattr(hit, "score", getattr(hit, "_score", doc.get("_score", 0.0)))
    ns = getattr(hit, "namespace", str(int(fields.get("year", 1910))))

    date_str = str(fields.get("date", "Unknown Date"))
    ed_val = int(fields.get("edition", 1))
    seq_val = int(fields.get("sequence", 1))
    season_val = fields.get("season", "Unknown")
    tok_count = int(fields.get("token_count", 0))
    chunk_idx = int(fields.get("chunk_index", 0)) + 1
    total_chunks = int(fields.get("chunk_count", 1))
    raw_text = fields.get("text", "")
    highlighted_html = hl.highlight_text(raw_text, query_str)

    np_slug = fields.get("newspaper_slug", "los_angeles_herald")
    if np_slug == "the_evening_world":
        badge_html = '<span style="background: #e0f2fe; color: #0369a1; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #bae6fd;">🗽 The Evening World (NY)</span>'
        accent_color = "#0284c7"
    else:
        badge_html = '<span style="background: #fef3c7; color: #92400e; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #fde68a;">🌴 LA Herald (CA)</span>'
        accent_color = "#d97706"

    file_path = fields.get("file_path") or fields.get("image_url", "")
    thumb_url = fields.get("iiif_thumb_url") or jp2_to_iiif_url(file_path, size="600,")
    high_res_url = jp2_to_iiif_url(file_path, size="1600,")
    loc_page = fields.get("loc_page_url", "")
    pdf_url = fields.get("pdf_url", "")

    return f"""
    <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.1rem; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-family: system-ui, -apple-system, sans-serif;">
      <!-- Card Header Badges -->
      <div style="margin-bottom: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;">
        {badge_html}
        <span style="font-weight: 700; font-size: 1rem; color: #0f172a;">📅 {date_str}</span>
        <span style="background: #f1f5f9; color: #334155; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">Page {seq_val} (Ed. {ed_val})</span>
        <span style="background: #f8fafc; color: #64748b; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem;">Year {ns}</span>
        <span style="background: #f3e8ff; color: #6b21a8; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">Chunk {chunk_idx}/{total_chunks}</span>
        <span style="margin-left: auto; background: #dbeafe; color: #1e40af; padding: 0.15rem 0.6rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">Score: {score:.4f}</span>
      </div>

      <!-- Split Layout: OCR Text + LoC IIIF Scan -->
      <div style="display: flex; flex-direction: row; gap: 1rem; align-items: flex-start;">
        <div style="flex: 3; min-width: 0;">
          <div style="background: #fafafa; border-left: 4px solid {accent_color}; padding: 0.85rem; border-radius: 4px; font-family: Georgia, serif; font-size: 0.92rem; line-height: 1.55; color: #1e293b; max-height: 250px; overflow-y: auto;">
            {highlighted_html}
          </div>
          <div style="margin-top: 0.6rem; font-size: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.85rem;">
            {f'<a href="{loc_page}" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">🏛️ View on LoC.gov ↗</a>' if loc_page else ''}
            {f'<a href="{pdf_url}" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">📄 Page PDF ↗</a>' if pdf_url else ''}
            {f'<a href="{high_res_url}" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">🔍 1600px High-Res ↗</a>' if high_res_url else ''}
          </div>
        </div>

        <div style="flex: 1; min-width: 140px; max-width: 190px; text-align: center;">
          <a href="{high_res_url}" target="_blank" title="Click to view high-resolution scan in new tab">
            <img src="{thumb_url}" alt="Scan for {date_str} Page {seq_val}" style="width: 100%; height: auto; border-radius: 4px; border: 1px solid #cbd5e1; box-shadow: 0 1px 2px rgba(0,0,0,0.1); cursor: zoom-in;" />
          </a>
          <span style="display: block; font-size: 0.72rem; color: #64748b; margin-top: 0.3rem;">Page {seq_val} (click to zoom)</span>
        </div>
      </div>
    </div>
    """


def run_search(
    query_str: str,
    newspaper_scope: str,
    mode: str,
    match_label: str,
    year_option: str,
    season: str,
    front_page: bool,
    top_k: int,
) -> Tuple[str, str]:
    """Execute search query and return summary HTML and cards HTML."""
    if init_error or not index_client or not openai_client:
        err_msg = f"""
        <div style="padding: 1rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; color: #991b1b;">
          <strong>Configuration Notice:</strong> Pinecone or OpenAI credentials are not configured.
          <code>{init_error or 'Missing API keys'}</code>.
          <br>Please set <code>PINECONE_API_KEY</code> and <code>OPENAI_API_KEY</code> in Space secrets.
        </div>
        """
        return err_msg, ""

    if not query_str or not query_str.strip():
        return (
            '<div style="color: #64748b; padding: 0.5rem 0;">Enter a search term above to query the archive.</div>',
            "",
        )

    t0 = time.time()
    match_op = MATCH_LABELS.get(match_label)
    breakdown_html = analyze_query_syntax(query_str, mode)

    is_all_years = year_option.startswith("All Years")
    target_namespaces = config.TARGET_YEARS if is_all_years else [year_option.strip()]

    # ── CASE 1: Both Newspapers (Side-by-Side Comparison) ─────────────────────
    if newspaper_scope.startswith("Both Newspapers"):
        base_filter = q.build_filter(
            season=season if season != "(any)" else None,
            front_page_only=front_page,
            match_op=match_op,
            match_terms=query_str,
        )

        res = q.search_side_by_side(
            index=index_client,
            oai=openai_client,
            query=query_str,
            namespaces=target_namespaces,
            mode=mode,
            match_op=match_op,
            match_terms=query_str,
            top_k_per_year=int(top_k),
            base_filter_dict=base_filter,
        )

        elapsed = time.time() - t0
        la_hits = res["la"]["merged"]
        ny_hits = res["ny"]["merged"]
        total_hits = len(la_hits) + len(ny_hits)

        # Summary Header
        summary = f"""
        <div style="margin-bottom: 1rem; padding: 0.85rem 1.1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
          <div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.8rem; align-items: center;">
            <span>🎯 Cross-Country Comparison: <strong>{total_hits}</strong> chunk(s) across {len(target_namespaces)} year(s) in {elapsed:.2f}s</span>
            <span style="background: #fef3c7; color: #92400e; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; border: 1px solid #fde68a;">🌴 LA Herald: {len(la_hits)} hits</span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; border: 1px solid #bae6fd;">🗽 NY Evening World: {len(ny_hits)} hits</span>
          </div>
          {breakdown_html}
        </div>
        """

        # Build Side-by-Side Two Column Layout
        la_cards = [hit_to_card_html(h, query_str) for h in la_hits]
        ny_cards = [hit_to_card_html(h, query_str) for h in ny_hits]

        la_content = "\n".join(la_cards) if la_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in the Los Angeles Herald.</div>'
        ny_content = "\n".join(ny_cards) if ny_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in The Evening World (New York).</div>'

        results_html = f"""
        <div class="newspaper-split">
          <!-- Left: West Coast (LA Herald) -->
          <div>
            <div style="padding: 0.65rem 0.9rem; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <strong style="color: #92400e; font-size: 1.05rem;">🌴 Los Angeles Herald</strong>
                <div style="font-size: 0.8rem; color: #78350f;">West Coast • Los Angeles, California • LCCN sn85042462</div>
              </div>
              <span style="background: #d97706; color: #ffffff; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">{len(la_hits)} hits</span>
            </div>
            {la_content}
          </div>

          <!-- Right: East Coast (The Evening World) -->
          <div>
            <div style="padding: 0.65rem 0.9rem; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <strong style="color: #0369a1; font-size: 1.05rem;">🗽 The Evening World</strong>
                <div style="font-size: 0.8rem; color: #075985;">East Coast • New York, New York • LCCN sn83030193</div>
              </div>
              <span style="background: #0284c7; color: #ffffff; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">{len(ny_hits)} hits</span>
            </div>
            {ny_content}
          </div>
        </div>
        """
        return summary, results_html

    # ── CASE 2: Single Newspaper Scope ────────────────────────────────────────
    np_slug = "the_evening_world" if "Evening World" in newspaper_scope else "los_angeles_herald"
    np_title = "The Evening World (New York)" if np_slug == "the_evening_world" else "Los Angeles Herald"

    filt = q.build_filter(
        newspaper_slug=np_slug,
        season=season if season != "(any)" else None,
        front_page_only=front_page,
        match_op=match_op,
        match_terms=query_str,
    )

    if is_all_years:
        search_data = q.search_cross_years(
            index=index_client,
            oai=openai_client,
            query=query_str,
            namespaces=target_namespaces,
            mode=mode,
            match_op=match_op,
            match_terms=query_str,
            top_k_per_year=int(top_k),
            filter_dict=filt,
        )
        elapsed = time.time() - t0
        merged_hits = search_data["merged"]
        total_count = search_data["total_hits"]
        by_year = search_data["by_year"]

        pills = []
        for yr in target_namespaces:
            count = len(by_year.get(yr, []))
            cls = "stat-pill active" if count > 0 else "stat-pill"
            pills.append(f'<span class="{cls}">{yr}: <strong>{count}</strong></span>')
        pills_html = " ".join(pills)

        summary = f"""
        <div style="margin-bottom: 1rem; padding: 0.75rem 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
          <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.4rem;">
            🎯 Retrieved <strong>{total_count}</strong> matching chunk(s) in <em>{np_title}</em> across {len(target_namespaces)} years in {elapsed:.2f}s
          </div>
          <div>{pills_html}</div>
          {breakdown_html}
        </div>
        """
        if not merged_hits:
            return summary, f'<div style="padding: 1rem; color: #64748b; font-style: italic;">No documents matched your query in {np_title}.</div>'

        cards = [hit_to_card_html(hit, query_str) for hit in merged_hits]
        return summary, "\n".join(cards)

    else:
        yr_ns = year_option.strip()
        try:
            if mode == "Full-text (BM25)":
                res = q.search_text(index_client, query_str, namespace=yr_ns, top_k=int(top_k), filter_dict=filt)
            elif mode == "Query string (Lucene)":
                res = q.search_query_string(index_client, query_str, namespace=yr_ns, top_k=int(top_k), filter_dict=filt)
            elif mode == "Semantic":
                res = q.search_semantic(index_client, openai_client, query_str, namespace=yr_ns, top_k=int(top_k), filter_dict=filt)
            else:  # Hybrid
                res = q.search_hybrid(
                    index_client,
                    openai_client,
                    query_str,
                    namespace=yr_ns,
                    match_op=match_op or "all",
                    match_terms=query_str,
                    top_k=int(top_k),
                    filter_dict=filt,
                )
            matches = list(getattr(res, "matches", getattr(res, "hits", [])) or [])
            hits = []
            for m in matches:
                d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
                d["namespace"] = yr_ns
                d["_score"] = getattr(m, "score", d.get("_score", 0.0))
                d["_id"] = getattr(m, "id", d.get("_id", ""))
                hits.append(d)
        except Exception as e:
            return f'<div style="color: #dc2626;">Search error: {e}</div>', ""

        elapsed = time.time() - t0
        summary = f"""
        <div style="margin-bottom: 1rem; padding: 0.75rem 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
          <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.2rem;">
            🎯 Retrieved <strong>{len(hits)}</strong> matching chunk(s) in <em>{np_title}</em> (namespace {yr_ns}) in {elapsed:.2f}s
          </div>
          {breakdown_html}
        </div>
        """
        if not hits:
            return summary, f'<div style="padding: 1rem; color: #64748b; font-style: italic;">No documents found in {np_title} (namespace "{yr_ns}") matching "{query_str}".</div>'

        cards = [hit_to_card_html(hit, query_str) for hit in hits]
        return summary, "\n".join(cards)


# Helper function to append Lucene tokens to the query box
def insert_token(current_query: str, token: str) -> Tuple[str, str]:
    new_query = f"{current_query.strip()} {token}".strip() if current_query.strip() else token
    return new_query, "Query string (Lucene)"


# ── Gradio Blocks Interface ─────────────────────────────────────────────────
with gr.Blocks(title="Coast-to-Coast Historical Newspaper Search (1890–1910)") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.Markdown(
        """
        # 📰 Coast-to-Coast Historical Newspaper Search (1890–1910)
        ### Cross-Country Comparative Archive • Library of Congress Chronicling America & Pinecone Documents API
        Compare **🌴 Los Angeles Herald** (West Coast) and **🗽 The Evening World** (New York, East Coast) across 20 pivotal years of American history using **Pinecone Lucene Query Syntax**, **BM25 Lexical Ranking**, **OpenAI Semantic Vectors**, and **Library of Congress IIIF Dynamic Image CDN**.
        """
    )

    # ── Lucene Query Assistant Quick-Insert Toolbar ───────────────────────────
    with gr.Row():
        gr.Markdown("**🛠️ Lucene Query Assistant (Quick-Insert Operators):**")
    with gr.Row():
        btn_req = gr.Button("+ Must Include", size="sm", elem_classes=["quick-chip"])
        btn_exc = gr.Button("- Exclude Term", size="sm", elem_classes=["quick-chip"])
        btn_phr = gr.Button('" " Exact Phrase', size="sm", elem_classes=["quick-chip"])
        btn_slp = gr.Button('~4 Proximity Slop', size="sm", elem_classes=["quick-chip"])
        btn_fuz = gr.Button('~1 Fuzzy / OCR', size="sm", elem_classes=["quick-chip"])
        btn_bst = gr.Button('^2 Term Boost', size="sm", elem_classes=["quick-chip"])
        btn_rgx = gr.Button('/ / Token Regex', size="sm", elem_classes=["quick-chip"])
        btn_and = gr.Button('AND', size="sm", elem_classes=["quick-chip"])
        btn_or  = gr.Button('OR', size="sm", elem_classes=["quick-chip"])

    with gr.Row():
        with gr.Column(scale=5):
            query_input = gr.Textbox(
                value='+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
                label="Search Query Expression",
                placeholder="e.g. '+\"Theodore Roosevelt\" +(speech OR visit) -advertisement' or '\"aviation meet\"~4'",
                lines=1,
            )
        with gr.Column(scale=1, min_width=120):
            search_btn = gr.Button("Search Archive", variant="primary", size="lg")

    with gr.Row():
        newspaper_dropdown = gr.Dropdown(
            choices=NEWSPAPER_OPTIONS,
            value=NEWSPAPER_OPTIONS[0],
            label="Newspaper Scope",
        )
        mode_radio = gr.Radio(
            choices=MODES,
            value="Query string (Lucene)",
            label="Ranking Engine Mode",
        )
        match_dropdown = gr.Dropdown(
            choices=list(MATCH_LABELS.keys()),
            value="None (off)",
            label="Lexical Hard Filter ($match_*) — Used in Hybrid Mode",
        )

    with gr.Row():
        year_dropdown = gr.Dropdown(
            choices=YEAR_OPTIONS,
            value=YEAR_OPTIONS[0],
            label="Historical Scope (Pinecone Namespace)",
        )
        season_dropdown = gr.Dropdown(
            choices=["(any)", "Winter", "Summer"],
            value="(any)",
            label="Season Window",
        )
        front_page_cb = gr.Checkbox(
            value=False,
            label="Front Pages Only (Page 1)",
        )
        top_k_slider = gr.Slider(
            minimum=1,
            maximum=20,
            value=4,
            step=1,
            label="Results per Year / Paper",
        )

    # ── Curated Library of Congress Digital Strategy Showcase Accordion ───────
    with gr.Accordion("🏛️ Curated Inquiries: Library of Congress Digital Strategy Showcase (Click to Load & Run)", open=True):
        gr.Markdown(
            """
            *Select any curated historical inquiry below to test Pinecone's Lucene syntax on real archival challenges across both **Los Angeles Herald** and **The Evening World (New York)**.*
            """
        )

        showcase_names = [f"[{item['category'].split()[0]}] {item['title']}: {item['query']}" for item in SHOWCASE_QUERIES]
        showcase_dropdown = gr.Dropdown(
            choices=showcase_names,
            label="Select a Curated Research Inquiry",
            value=showcase_names[0],  # 1906 SF Earthquake
        )
        showcase_desc = gr.Markdown(
            value=f"**Rationale:** {SHOWCASE_QUERIES[0]['desc']}\n\n**Query:** `{SHOWCASE_QUERIES[0]['query']}`"
        )
        load_run_btn = gr.Button("🚀 Load & Execute Inquiry", variant="secondary")

    summary_output = gr.HTML(label="Search Summary")
    results_output = gr.HTML(label="Search Results")

    # Wire Quick-Insert Toolbar buttons to append tokens and set Lucene mode
    btn_req.click(fn=lambda q: insert_token(q, "+term"), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_exc.click(fn=lambda q: insert_token(q, "-advertisement"), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_phr.click(fn=lambda q: insert_token(q, '"exact phrase"'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_slp.click(fn=lambda q: insert_token(q, '"phrase words"~4'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_fuz.click(fn=lambda q: insert_token(q, 'term~1'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_bst.click(fn=lambda q: insert_token(q, 'term^2'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_rgx.click(fn=lambda q: insert_token(q, 'text:/pattern.*/'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_and.click(fn=lambda q: insert_token(q, 'AND'), inputs=[query_input], outputs=[query_input, mode_radio])
    btn_or.click(fn=lambda q: insert_token(q, 'OR'), inputs=[query_input], outputs=[query_input, mode_radio])

    # Update description when showcase selection changes
    def update_showcase_desc(selected_label: str) -> str:
        idx = showcase_names.index(selected_label) if selected_label in showcase_names else 0
        item = SHOWCASE_QUERIES[idx]
        return f"**Archival Challenge & Rationale:** {item['desc']}\n\n**Active Expression:** `{item['query']}`"

    showcase_dropdown.change(fn=update_showcase_desc, inputs=[showcase_dropdown], outputs=[showcase_desc])

    # Load and execute showcase item
    def load_showcase_query(selected_label: str):
        idx = showcase_names.index(selected_label) if selected_label in showcase_names else 0
        item = SHOWCASE_QUERIES[idx]
        q_str = item["query"]
        scope_val = item.get("newspaper_scope", NEWSPAPER_OPTIONS[0])
        mode_val = item["mode"]
        yr_val = item["year"]
        season_val = item["season"]
        fp_val = item["front_page"]
        top_k_val = item["top_k"]

        summary, results = run_search(
            query_str=q_str,
            newspaper_scope=scope_val,
            mode=mode_val,
            match_label="None (off)",
            year_option=yr_val,
            season=season_val,
            front_page=fp_val,
            top_k=top_k_val,
        )
        return q_str, scope_val, mode_val, "None (off)", yr_val, season_val, fp_val, top_k_val, summary, results

    load_run_btn.click(
        fn=load_showcase_query,
        inputs=[showcase_dropdown],
        outputs=[
            query_input,
            newspaper_dropdown,
            mode_radio,
            match_dropdown,
            year_dropdown,
            season_dropdown,
            front_page_cb,
            top_k_slider,
            summary_output,
            results_output,
        ],
    )

    # Standard Search triggers
    search_inputs = [
        query_input,
        newspaper_dropdown,
        mode_radio,
        match_dropdown,
        year_dropdown,
        season_dropdown,
        front_page_cb,
        top_k_slider,
    ]

    search_btn.click(
        fn=run_search,
        inputs=search_inputs,
        outputs=[summary_output, results_output],
    )

    query_input.submit(
        fn=run_search,
        inputs=search_inputs,
        outputs=[summary_output, results_output],
    )

    # Reference Accordions
    with gr.Accordion("📖 Comprehensive Pinecone Lucene Query Syntax Reference", open=False):
        gr.Markdown(
            r"""
            ### Pinecone `query_string` Grammar & Operator Guide
            
            | Capability | Syntax Example | Archival Use Case / Historical Description |
            | :--- | :--- | :--- |
            | **Required Term** | `+aviation +pilot` | Ensures both terms MUST be present in the chunk. |
            | **Excluded Term** | `-patent -advertisement` | Strips ubiquitous late-Victorian commercial ads and patent medicines. |
            | **Exact Phrase** | `"San Francisco"` | Matches exact adjacent sequence of tokens in order. |
            | **Phrase Slop** | `"aviation meet"~4` | Allows up to 4 words between terms; solves multi-column layout false positives. |
            | **Term Boosting** | `aeroplane^3 "flying machine"^2` | Multiplies term contribution to score (3x for aeroplane, 2x for flying machine). |
            | **Fuzzy / Typo** | `Roosevclt~1`, `aeroplan~1` | Typo & OCR noise tolerance: Levenshtein distance $\le 1$ or $\le 2$. |
            | **Regex Match** | `text:/aeronaut.*/` | Matches regular expression patterns on indexed tokens. |
            | **Phrase Prefix** | `"san fran"*` | Matches phrases with autocomplete-style prefix expansion on final term. |
            | **Boolean Logic** | `((a OR b) AND c NOT d)` | Explicit precedence grouping with parentheses and uppercase operators. |
            """
        )

    with gr.Accordion("🏛️ About the Dataset & Architecture", open=False):
        gr.Markdown(
            """
            - **Source Newspapers**:
              - *Los Angeles Herald* (1890–1910), LCCN `sn85042462` (West Coast, California)
              - *The Evening World* (1890–1910), LCCN `sn83030193` (East Coast, New York)
            - **Digitization**: Library of Congress National Digital Newspaper Program (NDNP)
            - **Vector & Full-Text Search**: Pinecone Document Schema (`herald-hybrid-fts`) partitioned across 9 year namespaces
            - **Dense Vectors**: OpenAI `text-embedding-3-small` (1,536 dimensions)
            - **Chunking**: Chonkie recursive chunker (~500 tokens per chunk)
            - **Scan Display**: Dynamic Library of Congress IIIF Image API (`tile.loc.gov`) with zero local image storage
            """
        )

if __name__ == "__main__":
    demo.launch()
