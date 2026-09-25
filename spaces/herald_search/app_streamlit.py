"""Streamlit UI for Coast-to-Coast Historical Newspaper Search (1890–1910).

Exposes Pinecone Documents API full-text search, semantic search, and hybrid search
with dynamic visual page rendering via Library of Congress IIIF,
featuring side-by-side comparative analysis between:
- Los Angeles Herald (California, West Coast)
- The Evening World (New York, East Coast)
"""

import os
from typing import Any, Dict, List
import streamlit as st
from dotenv import load_dotenv

import config
import highlight as hl
from iiif import jp2_to_iiif_url, normalize_iiif_url
import query as q

load_dotenv()

st.set_page_config(
    page_title="Coast-to-Coast Newspaper Search (1890–1910)",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
      .chunk-box {
          background-color: #fcfcfc;
          border-left: 4px solid #d97706;
          padding: 1rem;
          border-radius: 4px;
          font-family: Georgia, serif;
          font-size: 0.95rem;
          line-height: 1.6;
          color: #1f2937;
          margin-bottom: 0.75rem;
      }
      .chunk-box-ny {
          background-color: #fcfcfc;
          border-left: 4px solid #0284c7;
          padding: 1rem;
          border-radius: 4px;
          font-family: Georgia, serif;
          font-size: 0.95rem;
          line-height: 1.6;
          color: #1f2937;
          margin-bottom: 0.75rem;
      }
      mark.hl {
          background: #fde047;
          color: #000;
          padding: 0.1em 0.25em;
          border-radius: 0.2em;
          font-weight: 600;
      }
      .meta-tag {
          display: inline-block;
          background: #e5e7eb;
          color: #374151;
          padding: 0.15rem 0.5rem;
          border-radius: 9999px;
          font-size: 0.75rem;
          font-weight: 500;
          margin-right: 0.4rem;
      }
      .score-tag {
          display: inline-block;
          background: #dbeafe;
          color: #1e40af;
          padding: 0.15rem 0.5rem;
          border-radius: 9999px;
          font-size: 0.75rem;
          font-weight: 600;
      }
      .card-container {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 1.1rem;
          margin-bottom: 1.25rem;
          background-color: #ffffff;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

MODES = ["Query string (Lucene)", "Hybrid", "Full-text (BM25)", "Semantic"]

MATCH_LABELS = {
    "None (off)": None,
    "All words — $match_all": "all",
    "Exact phrase — $match_phrase": "phrase",
    "Any word — $match_any": "any",
}

NEWSPAPER_OPTIONS = [
    "Both Newspapers (Side-by-Side Comparison)",
    "Los Angeles Herald (California, West Coast)",
    "The Evening World (New York, East Coast)",
]

LUCENE_GUIDE = """\
**Lucene Query Syntax Examples:**
- **Required / Excluded**: `+aviation -military`
- **Exact phrase**: `"Halley's comet"`
- **Phrase proximity / slop**: `"aviation meet"~4`
- **Term boost**: `aeroplane^3 "flying machine"^2`
- **Fuzzy typo tolerance**: `Roosevclt~1`, `aeroplan~1`
- **Regex pattern**: `text:/aeronaut.*/`
"""


@st.cache_resource
def get_backend():
    try:
        index, oai = q.get_clients()
        return index, oai, None
    except Exception as e:
        return None, None, str(e)


index, oai, init_error = get_backend()

# ── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Controls")
    newspaper_scope = st.selectbox("Newspaper Scope", NEWSPAPER_OPTIONS, index=0)
    mode = st.radio("Ranking Mode", MODES, index=0)

    match_label = st.selectbox(
        "Lexical Text-Match Filter ($match_*)",
        list(MATCH_LABELS.keys()),
        index=0,
    )
    match_op = MATCH_LABELS[match_label]

    st.divider()
    st.header("Historical Scope")

    year_options = ["All Years (Cross-Year Comparison)"] + config.TARGET_YEARS
    selected_year_option = st.selectbox("Year Namespace", year_options, index=0)

    season_filter = st.selectbox("Season", ["(any)", "Winter", "Summer"], index=0)
    front_page_cb = st.checkbox("Front Pages Only (Page 1)", value=False)
    top_k = st.slider("Results per Year / Paper", min_value=1, max_value=20, value=4, step=1)

    with st.expander("Lucene Syntax Reference"):
        st.markdown(LUCENE_GUIDE)

    with st.expander("About This Archive"):
        st.markdown(
            """
            **Newspapers:**
            - *Los Angeles Herald* (1890–1910, CA)
            - *The Evening World* (1890–1910, NY)
            - Digitized via the Library of Congress NDNP
            - Partitioned across 9 year namespaces in Pinecone
            - Chunked with Chonkie (~500 tokens)
            - Scans rendered dynamically via LoC IIIF CDN
            """
        )

# ── Main Header ─────────────────────────────────────────────────────────────
st.title("📰 Coast-to-Coast Historical Newspaper Search (1890–1910)")
st.caption(
    "Cross-Country Comparative Search: **Los Angeles Herald** (West Coast) vs. **The Evening World** (New York, East Coast) "
    "using Pinecone Documents API Full-Text Search (BM25 & Lucene), OpenAI Semantic Vectors, and Library of Congress IIIF."
)

if init_error:
    st.warning(
        f"⚠️ Pinecone or OpenAI credentials need configuration: `{init_error}`. "
        "Set `PINECONE_API_KEY` and `OPENAI_API_KEY` in environment or Space secrets."
    )

# ── Curated Library of Congress Digital Strategy Showcase ────────────────────
with st.expander("🏛️ Curated Inquiries: Library of Congress Digital Strategy Showcase", expanded=True):
    st.markdown(
        "*Select any curated historical inquiry below to test Pinecone's Lucene syntax on real archival challenges across both newspapers.*"
    )
    showcase_labels = [
        "Select a curated showcase inquiry...",
        "⚡ [Comparison] 1906 SF Earthquake: West Coast Panic vs. Wall Street Shock",
        "⚡ [Comparison] Theodore Roosevelt: Western Conservation vs. NYC Politics",
        "⚡ [Comparison] Aviation Dawn (1908–1910): Dominguez Meet vs. Hudson Flights",
        "⚡ [Comparison] Progressive Suffrage: California Enfranchisement vs. NYC Rallies",
        "⚡ [Comparison] 1890s Monetary Battles: Western Free Silver vs. Wall Street Gold",
        "⚡ [Comparison] Halley's Comet May 1910: Mount Wilson Telescopes vs. Manhattan Rooftops",
        "📜 [OCR Typo] Roosevelt Microfilm Typo: Roosevclt~1",
        "📜 [OCR Typo] Aviation Variant & Dropout: aeroplan~1",
        "📜 [Regex] Morphological Token Wildcard: text:/aeronaut.*/",
        "📍 [Proximity Slop] 1910 Dominguez Air Meet: \"aviation meet\"~4",
        "📍 [Proximity Slop] Wright Brothers Collaborative Coverage: \"Wright brothers\"~5",
        "🏛️ [De-Noising Ads] San Francisco Relief (Excluding Patent Ads)",
    ]
    chosen_showcase = st.selectbox("Showcase Inquiry", showcase_labels, index=0)

showcase_query_map = {
    showcase_labels[1]: '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
    showcase_labels[2]: '+"Theodore Roosevelt" +(policy OR conservation OR trust OR campaign OR speech)',
    showcase_labels[3]: '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude)',
    showcase_labels[4]: '+("woman suffrage" OR "equal suffrage") +(convention OR ballot OR vote OR parade)',
    showcase_labels[5]: '+(currency OR silver OR "Wall Street") +(gold OR panic OR treasury)',
    showcase_labels[6]: '("Halley\'s comet"~3 OR "tail of the comet"~2) +(astronomer OR observatory OR sky)',
    showcase_labels[7]: "Roosevclt~1",
    showcase_labels[8]: "aeroplan~1",
    showcase_labels[9]: "text:/aeronaut.*/",
    showcase_labels[10]: '"aviation meet"~4',
    showcase_labels[11]: '"Wright brothers"~5',
    showcase_labels[12]: '+"San Francisco" +(earthquake OR catastrophe OR relief) -remedy -medicine -patent',
}

# ── Search Input ────────────────────────────────────────────────────────────
default_val = showcase_query_map.get(chosen_showcase, '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy')
col_q, col_btn = st.columns([5, 1])
query_str = col_q.text_input(
    "Search Query Expression",
    value=default_val,
    placeholder="e.g. '+\"Theodore Roosevelt\" +(speech OR visit) -advertisement' or '\"aviation meet\"~4'",
    label_visibility="collapsed",
)
search_clicked = col_btn.button("Search Archive", type="primary", use_container_width=True)

# Quick toolbar
c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
if c1.button("+ Include"): query_str += " +term"
if c2.button("- Exclude"): query_str += " -advertisement"
if c3.button('" " Phrase'): query_str += ' "exact phrase"'
if c4.button("~4 Slop"): query_str += ' "phrase words"~4'
if c5.button("~1 Fuzzy"): query_str += " term~1"
if c6.button("^2 Boost"): query_str += " term^2"
if c7.button("/ / Regex"): query_str += " text:/pattern.*/"
if c8.button("AND"): query_str += " AND "
if c9.button("OR"): query_str += " OR "

st.divider()


def render_card(hit: Any, query: str):
    doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
    fields = getattr(hit, "fields", None) or doc
    score = getattr(hit, "score", getattr(hit, "_score", doc.get("_score", 0.0)))
    ns = getattr(hit, "namespace", str(fields.get("year", "1910")))

    date_str = str(fields.get("date", "Unknown"))
    seq_val = int(fields.get("sequence", 1))
    chunk_idx = int(fields.get("chunk_index", 0)) + 1
    chunk_tot = int(fields.get("chunk_count", 1))
    np_slug = fields.get("newspaper_slug", "los_angeles_herald")

    if np_slug == "the_evening_world":
        badge = '<span class="meta-tag" style="background:#e0f2fe; color:#0369a1; font-weight:700;">🗽 The Evening World (NY)</span>'
        box_class = "chunk-box-ny"
    else:
        badge = '<span class="meta-tag" style="background:#fef3c7; color:#92400e; font-weight:700;">🌴 LA Herald (CA)</span>'
        box_class = "chunk-box"

    loc_page = fields.get("loc_page_url", "")
    pdf_url = fields.get("pdf_url", "")
    jp2_url = fields.get("image_url", "")
    raw_thumb = fields.get("iiif_thumb_url", "")
    thumb_url = normalize_iiif_url(raw_thumb or jp2_url, size="600,")
    high_res_url = normalize_iiif_url(jp2_url or raw_thumb, size="1600,")

    text_body = fields.get("text", "")
    highlighted_html = hl.highlight_text(text_body, query)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="margin-bottom: 0.6rem; display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;">
          {badge}
          <span style="font-weight: 700; font-size: 1rem; color: #111827;">📅 {date_str}</span>
          <span class="meta-tag">Page {seq_val}</span>
          <span class="meta-tag">Year {ns}</span>
          <span class="meta-tag">Chunk {chunk_idx}/{chunk_tot}</span>
          <span class="score-tag" style="margin-left: auto;">Score: {score:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_text, col_img = st.columns([3, 1])
    with col_text:
        st.markdown(f'<div class="{box_class}">{highlighted_html}</div>', unsafe_allow_html=True)
        link_md = []
        if loc_page: link_md.append(f"[🏛️ View on LoC.gov]({loc_page})")
        if pdf_url: link_md.append(f"[📄 Download Page PDF]({pdf_url})")
        if high_res_url: link_md.append(f"[🔍 1600px High-Res]({high_res_url})")
        if link_md:
            st.markdown(" • ".join(link_md))

    with col_img:
        if thumb_url:
            st.image(thumb_url, caption=f"Page {seq_val}", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Search Execution ────────────────────────────────────────────────────────
if search_clicked or query_str:
    if not index or not oai:
        st.error("Pinecone or OpenAI clients could not be initialized.")
        st.stop()

    is_all_years = selected_year_option.startswith("All Years")
    target_namespaces = config.TARGET_YEARS if is_all_years else [selected_year_option.strip()]

    # Case 1: Both Newspapers (Side-by-Side)
    if newspaper_scope.startswith("Both Newspapers"):
        base_filter = q.build_filter(
            season=season_filter if season_filter != "(any)" else None,
            front_page_only=front_page_cb,
            match_op=match_op,
            match_terms=query_str,
        )

        with st.spinner("Searching both newspapers across coasts..."):
            res = q.search_side_by_side(
                index=index,
                oai=oai,
                query=query_str,
                namespaces=target_namespaces,
                mode=mode,
                match_op=match_op,
                match_terms=query_str,
                top_k_per_year=int(top_k),
                base_filter_dict=base_filter,
            )

        la_hits = res["la"]["merged"]
        ny_hits = res["ny"]["merged"]
        total_hits = len(la_hits) + len(ny_hits)

        st.success(
            f"🎯 Retrieved **{total_hits}** total chunks across {len(target_namespaces)} year(s) "
            f"(🌴 LA Herald: **{len(la_hits)}** | 🗽 NY Evening World: **{len(ny_hits)}**)"
        )

        col_la, col_ny = st.columns(2)
        with col_la:
            st.subheader(f"🌴 Los Angeles Herald ({len(la_hits)} hits)")
            if not la_hits:
                st.info("No matching records found in the Los Angeles Herald.")
            for hit in la_hits:
                render_card(hit, query_str)

        with col_ny:
            st.subheader(f"🗽 The Evening World ({len(ny_hits)} hits)")
            if not ny_hits:
                st.info("No matching records found in The Evening World (New York).")
            for hit in ny_hits:
                render_card(hit, query_str)

    # Case 2: Single Newspaper Scope
    else:
        np_slug = "the_evening_world" if "Evening World" in newspaper_scope else "los_angeles_herald"
        np_title = "The Evening World (New York)" if np_slug == "the_evening_world" else "Los Angeles Herald"

        filt = q.build_filter(
            newspaper_slug=np_slug,
            season=season_filter if season_filter != "(any)" else None,
            front_page_only=front_page_cb,
            match_op=match_op,
            match_terms=query_str,
        )

        with st.spinner(f"Searching {np_title}..."):
            search_data = q.search_cross_years(
                index=index,
                oai=oai,
                query=query_str,
                namespaces=target_namespaces,
                mode=mode,
                match_op=match_op,
                match_terms=query_str,
                top_k_per_year=int(top_k),
                filter_dict=filt,
            )

        hits = search_data["merged"]
        st.success(f"🎯 Retrieved **{len(hits)}** matching chunk(s) in {np_title} across {len(target_namespaces)} year(s).")
        if not hits:
            st.info(f"No documents matched your query in {np_title}.")
        for hit in hits:
            render_card(hit, query_str)
