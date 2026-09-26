"""Streamlit UI for Coast-to-Coast Historical Newspaper Search (1890–1910).

Exposes Pinecone Documents API full-text search, semantic search, and hybrid search
with dynamic visual page rendering via Library of Congress IIIF,
featuring:
1. Interactive Story Mode: 7 Curated narrative chapters with editable queries & live re-execution
2. Archival Research Workbench: State filter (CA/NY), multi-year multiselect, and month filter
3. Interactive Calendar View ("On This Day in History"): Select any calendar day to explore all 9 years
4. Real-Time Latency & Throughput HUD: Sub-second metrics across 9 parallel year namespaces
5. Time Walker & Timeline Distribution: Article density histogram across 1890–1910
"""

import os
import time
from typing import Any, Dict, List
import streamlit as st
from dotenv import load_dotenv

import config
import highlight as hl
from iiif import jp2_to_iiif_url, normalize_iiif_url
import query as q
import story_data as sd

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
          background-color: #fafafa;
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
          background-color: #fafafa;
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
    "Both Original Coast-to-Coast (Side-by-Side Comparison)",
    "All Newspapers (Nationwide)",
    "Los Angeles Herald (California, West Coast)",
    "The San Francisco Call (California, Bay Area)",
    "The Evening World (New York, East Coast)",
    "The Sun (New York, East Coast)",
    "The Beatrice Daily Express (Nebraska, Heartland)",
    "Chicago Eagle (Illinois, Midwest)",
]

STATE_OPTIONS = ["All States", "California", "New York", "Nebraska", "Illinois"]

MONTH_OPTIONS = [
    "(any)",
    "01 - January",
    "07 - July",
    "08 - August",
    "12 - December",
]

CALENDAR_MONTHS = ["01 - January", "07 - July", "08 - August", "12 - December"]
CALENDAR_DAYS = [f"{d:02d}" for d in range(1, 32)]


@st.cache_resource
def get_backend():
    try:
        index, oai = q.get_clients()
        return index, oai, None
    except Exception as e:
        return None, None, str(e)


index, oai, init_error = get_backend()

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


def render_latency_hud(metrics: Dict[str, float], total_hits: int, la_hits: int, ny_hits: int, mode: str, ns_count: int):
    total_ms = metrics.get("total_ms", 0.0)
    pinecone_ms = metrics.get("pinecone_ms", 0.0)
    embed_ms = metrics.get("embed_ms", 0.0)
    embed_str = f" | 🤖 Embedding: **{embed_ms:.0f}ms**" if embed_ms > 0 else ""

    st.markdown(
        f"""
        <div style="background: #0f172a; color: #f8fafc; padding: 0.65rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-family: monospace; font-size: 0.85rem; border-left: 4px solid #38bdf8;">
          <span style="color: #38bdf8; font-weight: 700;">⚡ PINECONE METRICS:</span>
          ⏱️ Latency: <strong style="color:#38bdf8;">{total_ms:.0f}ms</strong> ({total_ms/1000.0:.2f}s) |
          🌲 Pinecone Query: <strong style="color:#4ade80;">{pinecone_ms:.0f}ms</strong>{embed_str} |
          📚 Retrieved: <strong>{total_hits}</strong> (<span style="color:#fbbf24;">{la_hits} LA</span> / <span style="color:#60a5fa;">{ny_hits} NY</span>) |
          🎯 Mode: <strong>{mode}</strong> | 🗂️ Namespaces: <strong>{ns_count}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline_distribution(la_counts: Dict[str, int], ny_counts: Dict[str, int]):
    years = config.TARGET_YEARS
    max_count = max(max(la_counts.values(), default=1), max(ny_counts.values(), default=1), 1)

    columns_html = []
    total_la = sum(la_counts.values())
    total_ny = sum(ny_counts.values())

    for yr in years:
        c_la = la_counts.get(yr, 0)
        c_ny = ny_counts.get(yr, 0)
        h_la = max(int((c_la / max_count) * 45), 4) if c_la > 0 else 2
        h_ny = max(int((c_ny / max_count) * 45), 4) if c_ny > 0 else 2
        bg_la = "#f59e0b" if c_la > 0 else "#e2e8f0"
        bg_ny = "#0284c7" if c_ny > 0 else "#e2e8f0"

        col = f"""
        <div style="flex: 1; min-width: 32px; display: flex; flex-direction: column; align-items: center; gap: 2px;">
          <div style="font-size: 0.7rem; color: #64748b; font-weight: 600;">{c_la + c_ny}</div>
          <div style="display: flex; gap: 2px; align-items: flex-end; height: 50px; width: 100%; justify-content: center;">
            <div title="LA Herald ({yr}): {c_la} hit(s)" style="width: 42%; height: {h_la}px; background: {bg_la}; border-radius: 2px 2px 0 0;"></div>
            <div title="The Evening World ({yr}): {c_ny} hit(s)" style="width: 42%; height: {h_ny}px; background: {bg_ny}; border-radius: 2px 2px 0 0;"></div>
          </div>
          <div style="font-size: 0.72rem; color: #334155; font-weight: 700; border-top: 1px solid #cbd5e1; width: 100%; text-align: center; padding-top: 2px;">{yr}</div>
        </div>
        """
        columns_html.append(col)

    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 700; font-size: 0.88rem; color: #1e293b;">📊 Article Density Distribution Across 9 Year Namespaces</span>
            <div style="font-size: 0.78rem; display: flex; gap: 0.75rem; align-items: center;">
              <span style="display: inline-flex; align-items: center; gap: 0.25rem;"><span style="display:inline-block; width:10px; height:10px; background:#f59e0b; border-radius:2px;"></span> 🌴 LA Herald (Total: <strong>{total_la}</strong>)</span>
              <span style="display: inline-flex; align-items: center; gap: 0.25rem;"><span style="display:inline-block; width:10px; height:10px; background:#0284c7; border-radius:2px;"></span> 🗽 The Evening World (Total: <strong>{total_ny}</strong>)</span>
            </div>
          </div>
          <div style="display: flex; gap: 0.35rem; align-items: flex-end; justify-content: space-between; overflow-x: auto; padding: 0.25rem 0;">
            {''.join(columns_html)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(hit: Any, query: str):
    doc = hit.to_dict() if hasattr(hit, "to_dict") else (hit if isinstance(hit, dict) else {})
    fields = getattr(hit, "fields", None) or doc
    score = getattr(hit, "score", getattr(hit, "_score", doc.get("_score", 0.0)))
    ns = getattr(hit, "namespace", str(int(fields.get("year", 1910))))

    date_str = str(fields.get("date", "Unknown"))
    seq_val = int(fields.get("sequence", 1))
    chunk_idx = int(fields.get("chunk_index", 0)) + 1
    chunk_tot = int(fields.get("chunk_count", 1))
    np_slug = fields.get("newspaper_slug", "los_angeles_herald")

    if np_slug == "the_evening_world":
        badge = '<span class="meta-tag" style="background:#e0f2fe; color:#0369a1; font-weight:700;">🗽 The Evening World (NY)</span>'
        box_class = "chunk-box-ny"
    elif np_slug == "the_sun":
        badge = '<span class="meta-tag" style="background:#fef08a; color:#854d0e; font-weight:700;">☀️ The Sun (NY)</span>'
        box_class = "chunk-box"
    elif np_slug == "the_beatrice_daily_express":
        badge = '<span class="meta-tag" style="background:#ecfccb; color:#3f6212; font-weight:700;">🌾 Beatrice Express (NE)</span>'
        box_class = "chunk-box"
    elif np_slug == "chicago_eagle":
        badge = '<span class="meta-tag" style="background:#f3e8ff; color:#6b21a8; font-weight:700;">🦅 Chicago Eagle (IL)</span>'
        box_class = "chunk-box"
    elif np_slug == "the_san_francisco_call":
        badge = '<span class="meta-tag" style="background:#ccfbf1; color:#115e59; font-weight:700;">🌁 SF Call (CA)</span>'
        box_class = "chunk-box"
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
    highlighted_html = hl.highlight_text(text_body, query) if query else text_body[:400] + "..."

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


# ── Tab Navigation ──────────────────────────────────────────────────────────
tab_story, tab_workbench, tab_calendar, tab_docs = st.tabs([
    "🏛️ Interactive Story Mode (Coast-to-Coast 1890–1910)",
    "🔬 Archival Research Workbench (Advanced Search)",
    "📅 Calendar View ('On This Day in History')",
    "📖 Pinecone Architecture & Lucene Guide",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: INTERACTIVE STORY MODE
# ─────────────────────────────────────────────────────────────────────────────
with tab_story:
    chapter_options = [f"Ch. {c['number']}: {c['title']} ({c['period']})" for c in sd.CHAPTERS]
    selected_ch_str = st.radio("Select a Historical Chapter", chapter_options, horizontal=True)

    ch_idx = chapter_options.index(selected_ch_str) if selected_ch_str in chapter_options else 0
    chapter = sd.CHAPTERS[ch_idx]

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        story_mode = st.selectbox("Ranking Mode", MODES, index=0, key="story_mode")
    with col_ctrl2:
        story_year = st.selectbox("Historical Scope", ["All Years (Cross-Year Comparison)"] + config.TARGET_YEARS, index=0, key="story_year")
    with col_ctrl3:
        story_k = st.slider("Top K per Year", 1, 15, 4, key="story_top_k")

    default_active_q = chapter["semantic_query"] if story_mode in ("Semantic", "Hybrid") else chapter["query"]
    story_q_input = st.text_area("Active Story Query Expression (Editable)", value=default_active_q, key=f"sq_{ch_idx}")
    run_story_btn = st.button("⚡ Execute Story Query", type="primary")

    # Render Chapter Banner
    figures_pills = " • ".join([f"**{fig}**" for fig in chapter.get("figures", [])])
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.1rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="background:#0f172a; color:#fff; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.78rem; font-weight:700;">Chapter {chapter['number']} of {len(sd.CHAPTERS)}</span>
            <span style="color:#64748b; font-size:0.85rem; font-weight:600;">{chapter['badge']}</span>
          </div>
          <h2 style="margin:0.35rem 0 0.15rem 0; font-family:Georgia, serif; color:#0f172a;">{chapter['title']}</h2>
          <div style="color:#475569; font-style:italic; margin-bottom:0.75rem;">{chapter['subtitle']}</div>
          <p style="font-family:Georgia, serif; font-size:0.96rem; line-height:1.6; color:#1e293b;">{chapter['narrative']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if chapter.get("midwest_angle"):
        c_west, c_east, c_midwest = st.columns(3)
        with c_west:
            st.info(chapter["west_angle"])
        with c_east:
            st.info(chapter["east_angle"])
        with c_midwest:
            st.success(chapter["midwest_angle"])
    else:
        c_west, c_east = st.columns(2)
        with c_west:
            st.info(chapter["west_angle"])
        with c_east:
            st.info(chapter["east_angle"])
    st.caption(f"Key Figures: {figures_pills}")

    # Query Execution for Chapter
    active_q = story_q_input.strip() if story_q_input else default_active_q
    target_ns = config.TARGET_YEARS if story_year.startswith("All Years") else [story_year.strip()]

    with st.spinner("Retrieving chapter records from Pinecone across coasts..."):
        res = q.search_side_by_side(
            index=index,
            oai=oai,
            query=active_q,
            namespaces=target_ns,
            mode=story_mode,
            match_op=None,
            match_terms=None,
            top_k_per_year=int(story_k),
        )

    la_hits = res["la"]["merged"]
    ny_hits = res["ny"]["merged"]
    total_hits = len(la_hits) + len(ny_hits)
    metrics = res.get("metrics", {})

    render_latency_hud(metrics, total_hits, len(la_hits), len(ny_hits), story_mode, len(target_ns))
    render_timeline_distribution(res["la"]["year_counts"], res["ny"]["year_counts"])

    col_la, col_ny = st.columns(2)
    with col_la:
        st.subheader(f"🌴 Los Angeles Herald ({len(la_hits)} hits)")
        if not la_hits:
            st.info("No matching records found in the Los Angeles Herald.")
        for hit in la_hits:
            render_card(hit, active_q)

    with col_ny:
        st.subheader(f"🗽 The Evening World ({len(ny_hits)} hits)")
        if not ny_hits:
            st.info("No matching records found in The Evening World (New York).")
        for hit in ny_hits:
            render_card(hit, active_q)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ARCHIVAL RESEARCH WORKBENCH (ADVANCED SEARCH)
# ─────────────────────────────────────────────────────────────────────────────
with tab_workbench:
    with st.expander("🏛️ Curated Inquiries: Library of Congress Digital Strategy Showcase", expanded=False):
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
            "📍 [Proximity Slop] 1910 Dominguez Air Meet: \"aviation meet\"~4",
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
        showcase_labels[9]: '"aviation meet"~4',
    }

    default_val = showcase_query_map.get(chosen_showcase, '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy')

    col_q, col_btn = st.columns([5, 1])
    wb_query_str = col_q.text_input(
        "Search Query Expression",
        value=default_val,
        placeholder="e.g. '+\"Theodore Roosevelt\" +(speech OR visit) -advertisement' or '\"aviation meet\"~4'",
        label_visibility="collapsed",
    )
    wb_search_clicked = col_btn.button("Search Archive", type="primary", use_container_width=True)

    col_wb1, col_wb2, col_wb3, col_wb4 = st.columns(4)
    with col_wb1:
        wb_state = st.selectbox("State Scope", STATE_OPTIONS, index=0, key="wb_state")
    with col_wb2:
        wb_newspaper = st.selectbox("Newspaper Scope", NEWSPAPER_OPTIONS, index=0, key="wb_newspaper")
    with col_wb3:
        wb_mode = st.selectbox("Ranking Mode", MODES, index=0, key="wb_mode")
    with col_wb4:
        wb_month = st.selectbox("Month Filter", MONTH_OPTIONS, index=0, key="wb_month")

    col_wb5, col_wb6 = st.columns([3, 1])
    with col_wb5:
        wb_years = st.multiselect("Historical Years (Namespaces)", config.TARGET_YEARS, default=config.TARGET_YEARS, key="wb_years")
    with col_wb6:
        wb_k = st.slider("Top K per Year", 1, 20, 4, key="wb_k")

    if wb_search_clicked or wb_query_str:
        wb_target_ns = wb_years if wb_years else config.TARGET_YEARS
        target_state = wb_state if wb_state != "All States" else None
        slug_map = {
            "Los Angeles Herald": "los_angeles_herald",
            "The San Francisco Call": "the_san_francisco_call",
            "The Evening World": "the_evening_world",
            "The Sun": "the_sun",
            "The Beatrice Daily Express": "the_beatrice_daily_express",
            "Chicago Eagle": "chicago_eagle",
        }
        slug = None
        for title_sub, s in slug_map.items():
            if title_sub in str(wb_newspaper):
                slug = s
                break

        if not slug and (wb_newspaper.startswith("Both") and target_state is None):
            with st.spinner("Searching across both newspapers..."):
                base_filter = q.build_filter(
                    month=wb_month if wb_month != "(any)" else None,
                    match_op=None,
                    match_terms=wb_query_str,
                )
                wb_res = q.search_side_by_side(
                    index=index,
                    oai=oai,
                    query=wb_query_str,
                    namespaces=wb_target_ns,
                    mode=wb_mode,
                    match_op=None,
                    match_terms=None,
                    top_k_per_year=int(wb_k),
                    base_filter_dict=base_filter,
                )
            la_hits = wb_res["la"]["merged"]
            ny_hits = wb_res["ny"]["merged"]
            total_hits = len(la_hits) + len(ny_hits)
            metrics = wb_res.get("metrics", {})

            render_latency_hud(metrics, total_hits, len(la_hits), len(ny_hits), wb_mode, len(wb_target_ns))
            render_timeline_distribution(wb_res["la"]["year_counts"], wb_res["ny"]["year_counts"])

            col_la, col_ny = st.columns(2)
            with col_la:
                st.subheader(f"🌴 Los Angeles Herald ({len(la_hits)} hits)")
                for hit in la_hits:
                    render_card(hit, wb_query_str)
            with col_ny:
                st.subheader(f"🗽 The Evening World ({len(ny_hits)} hits)")
                for hit in ny_hits:
                    render_card(hit, wb_query_str)
        else:
            filt = q.build_filter(
                state=target_state,
                newspaper_slug=slug,
                month=wb_month if wb_month != "(any)" else None,
                match_op=None,
                match_terms=wb_query_str,
            )
            with st.spinner("Executing targeted search..."):
                search_data = q.search_cross_years(
                    index=index,
                    oai=oai,
                    query=wb_query_str,
                    namespaces=wb_target_ns,
                    mode=wb_mode,
                    match_op=None,
                    match_terms=None,
                    top_k_per_year=int(wb_k),
                    filter_dict=filt,
                )
            merged = search_data["merged"]
            metrics = search_data.get("metrics", {})
            render_latency_hud(metrics, len(merged), 0, 0, wb_mode, len(wb_target_ns))
            render_timeline_distribution(search_data["year_counts"], {yr: 0 for yr in config.TARGET_YEARS})
            for hit in merged:
                render_card(hit, wb_query_str)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CALENDAR VIEW ("ON THIS DAY IN HISTORY")
# ─────────────────────────────────────────────────────────────────────────────
with tab_calendar:
    st.subheader("📅 On This Day in History: Cross-Year Date Archive Explorer")
    st.caption("Select any calendar date to retrieve and visually compare front pages published on that exact day across all **9 years (1890–1910)** across all archived states.")

    c_m, c_d, c_s, c_fp, c_btn = st.columns([2, 1, 2, 2, 2])
    with c_m:
        cal_month = st.selectbox("Calendar Month", CALENDAR_MONTHS, index=0)
    with c_d:
        cal_day = st.selectbox("Calendar Day", CALENDAR_DAYS, index=1)
    with c_s:
        cal_state = st.selectbox("State Filter", STATE_OPTIONS, index=0)
    with c_fp:
        cal_fp = st.checkbox("Front Pages Only", value=True)
    with c_btn:
        st.write("")
        st.write("")
        cal_explore_btn = st.button("📅 Explore Date", type="primary")

    m_int = int(cal_month.split()[0].lstrip("0") or "1")
    d_int = int(cal_day.split()[0].lstrip("0") or "1")

    with st.spinner(f"Retrieving front pages for {cal_month.split('-')[-1].strip()} {d_int} across all 9 years..."):
        cal_res = q.search_calendar_date(
            index=index,
            month=m_int,
            day=d_int,
            front_page_only=cal_fp,
            state=cal_state,
            top_k_per_year=4,
        )

    tot = cal_res.get("total_hits", 0)
    ms = cal_res.get("metrics", {}).get("total_ms", 0)
    st.success(f"⚡ Retrieved **{tot}** page records published on **{m_int:02d}-{d_int:02d}** across 9 year namespaces in **{ms:.0f}ms**.")

    by_year = cal_res.get("by_year", {})
    for yr in config.TARGET_YEARS:
        hits = by_year.get(yr, [])
        if not hits:
            continue
        st.markdown(f"#### 📅 {yr} Edition — {cal_month.split('-')[-1].strip()} {d_int}, {yr} ({len(hits)} issues)")
        cols = st.columns(min(len(hits), 3))
        for idx, hit in enumerate(hits):
            with cols[idx % len(cols)]:
                render_card(hit, "")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: REFERENCE & ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown(
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

        ---

        ### Dataset Architecture
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
