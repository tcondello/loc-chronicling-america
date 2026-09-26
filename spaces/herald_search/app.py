"""Gradio Web Application for Coast-to-Coast Historical Newspaper Search (1890–1910).

Exposes Pinecone Documents API full-text search, semantic search, and hybrid search
with dynamic visual page rendering via Library of Congress IIIF Image API,
featuring:
1. Interactive Story Mode: 7 Curated narrative chapters with editable queries & live re-execution
2. Archival Research Workbench: State filter (CA/NY), multi-year selector, and month filter
3. Interactive Calendar View ("On This Day in History"): Select any calendar day to explore all 9 years
4. Real-Time Latency & Throughput HUD: Sub-second metrics across 9 parallel year namespaces
5. Time Walker & Timeline Distribution: Article density histogram across 1890–1910
"""

import html
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from dotenv import load_dotenv

import config
import highlight as hl
from iiif import jp2_to_iiif_url, normalize_iiif_url
import query as q
import story_data as sd

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

YEAR_OPTIONS = config.TARGET_YEARS

# ── Curated Library of Congress Digital Strategy Showcase ────────────────────
SHOWCASE_QUERIES = [
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "1906 SF Earthquake: West Coast Panic vs. Wall Street Impact",
        "desc": "Compare how the two coasts responded to the catastrophic April 1906 earthquake. LA mobilized emergency supply trains, while New York covered Wall Street financial shocks and severed telegraph cables.",
        "query": '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Theodore Roosevelt: Western Conservation vs. NYC Politics",
        "desc": "Examine divergent press focus on Roosevelt: LA Herald championed western forestry and reclamation, while NYC Evening World tracked Tammany Hall political machines and Wall Street trusts.",
        "query": '+"Theodore Roosevelt" +(policy OR conservation OR trust OR campaign OR speech)',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Aviation Dawn (1908–1910): Dominguez Meet vs. Hudson Flights",
        "desc": "Captures the birth of American aviation across coasts: LA Herald's coverage of the landmark January 1910 Dominguez Field Air Meet vs. NYC Evening World's reports on Curtiss and Wright demonstration flights.",
        "query": '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude)',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Progressive Era Suffrage: California Enfranchisement vs. NYC Rallies",
        "desc": "Traces women's voting rights mobilization: Western states moving toward California's 1911 equal suffrage victory vs. New York City's growing mass parades on Fifth Avenue.",
        "query": '+("woman suffrage" OR "equal suffrage") +(convention OR ballot OR vote OR parade)',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "1890s Monetary Battles: Western Free Silver vs. Wall Street Gold Standard",
        "desc": "Contrast western populism favoring silver coinage in California against New York banking house orthodoxy and gold reserve defense during the 1890s financial crises.",
        "query": '+(currency OR silver OR "Wall Street") +(gold OR panic OR treasury)',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "⚡ East Coast vs. West Coast Comparison",
        "title": "Halley\'s Comet May 1910: Mount Wilson Telescopes vs. Manhattan Rooftops",
        "desc": "Contrast Southern California's scientific observations atop Mount Wilson with Manhattan's rooftop vantage points and cyanogen tail anxieties.",
        "query": '("Halley\'s comet"~3 OR "tail of the comet"~2) +(astronomer OR observatory OR sky)',
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": ["1910"],
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 4,
    },
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Roosevelt Microfilm Typo (~1)",
        "desc": "Microfilm ink bleed corrupts 'Roosevelt' to 'Roosevclt' (c for e). Fuzzy matching within 1 edit distance rescues corrupted historical records across both newspapers.",
        "query": "Roosevclt~1",
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "📜 OCR Noise & Typo Remediation",
        "title": "Aviation Variant & Dropout (aeroplan~1)",
        "desc": "Recovers archaic spelling 'aeroplane', early American 'airplane', and OCR dropouts ('aeroplan') in early aviation news.",
        "query": "aeroplan~1",
        "state": "All States",
        "newspaper_scope": "Both Newspapers (Side-by-Side Comparison)",
        "mode": "Query string (Lucene)",
        "years": config.TARGET_YEARS,
        "month": "(any)",
        "season": "(any)",
        "front_page": False,
        "top_k": 3,
    },
    {
        "category": "📍 Historical Proximity & Column Co-Occurrence",
        "title": "1910 Dominguez Air Meet (Phrase Slop ~4)",
        "desc": "Solves multi-column page layout false positives by requiring 'aviation' and 'meet' to appear within 4 words of each other.",
        "query": '"aviation meet"~4',
        "state": "California",
        "newspaper_scope": "Los Angeles Herald (California, West Coast)",
        "mode": "Query string (Lucene)",
        "years": ["1910"],
        "month": "01 - January",
        "season": "Winter",
        "front_page": False,
        "top_k": 4,
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
.hidden-state-bridge {
    display: none !important;
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
/* Modern HTML Filter Hub */
.archive-filter-hub {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.filter-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem;
    margin-bottom: 0.75rem;
}
.filter-group {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    flex: 1;
    min-width: 250px;
}
.filter-group.full-width {
    flex: 1 1 100%;
}
.group-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #475569;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.filter-hint {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}
.seg-control {
    display: inline-flex;
    flex-wrap: wrap;
    background: #f1f5f9;
    padding: 3px;
    border-radius: 6px;
    gap: 3px;
    border: 1px solid #e2e8f0;
}
.seg-btn {
    border: 0;
    background: transparent;
    color: #475569;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.32rem 0.72rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.seg-btn:hover {
    color: #0f172a;
    background: rgba(255,255,255,0.7);
}
.seg-btn.active {
    background: #ffffff;
    color: #0f172a;
    font-weight: 700;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}
.seg-btn .tag {
    font-size: 0.68rem;
    padding: 0.05rem 0.38rem;
    border-radius: 9999px;
    background: #e2e8f0;
    color: #475569;
    font-weight: 600;
}
.seg-btn.active .tag {
    background: #e0f2fe;
    color: #0284c7;
}
.year-rail {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    background: #f8fafc;
    padding: 5px 9px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}
.year-btn {
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #334155;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.year-btn:hover {
    border-color: #38bdf8;
    color: #0284c7;
}
.year-btn.active {
    background: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
}
.chip-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 0.38rem;
    margin-top: 0.25rem;
}
.op-chip {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #334155;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
}
.op-chip:hover {
    background: #e0f2fe;
    border-color: #7dd3fc;
    color: #0369a1;
}
.op-chip.clear-chip {
    color: #dc2626;
    border-color: #fecaca;
    background: #fff5f5;
}
.op-chip.clear-chip:hover {
    background: #fee2e2;
    border-color: #f87171;
}
.preset-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.3rem;
}
.preset-pill {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    color: #1e293b;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    cursor: pointer;
    transition: all 0.15s;
}
.preset-pill:hover {
    background: #f0fdf4;
    border-color: #86efac;
    color: #15803d;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.chapter-strip {
    display: flex;
    gap: 0.6rem;
    overflow-x: auto;
    padding: 0.3rem 0 0.6rem 0;
    margin-bottom: 0.75rem;
}
.chapter-card {
    flex: 0 0 170px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.chapter-card:hover {
    border-color: #38bdf8;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.chapter-card.active {
    background: #f0f9ff;
    border-color: #0284c7;
    border-left: 4px solid #0284c7;
}
.ch-badge {
    font-size: 0.68rem;
    font-weight: 700;
    color: #0284c7;
    text-transform: uppercase;
}
.ch-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
}
.ch-meta {
    font-size: 0.7rem;
    color: #64748b;
}
"""

CUSTOM_JS = """
<script>
window.selectChapter = function(idx) {
  document.querySelectorAll('.chapter-card').forEach((c, i) => {
    if (i === idx) c.classList.add('active');
    else c.classList.remove('active');
  });
  const radioInputs = document.querySelectorAll('#hidden-story-chapter-radio input[type="radio"]');
  if (radioInputs && radioInputs[idx]) {
    radioInputs[idx].click();
  }
};

window.insertSyntax = function(token, cursorOffset) {
  const el = document.querySelector('#wb-query-box textarea') || document.querySelector('#wb-query-box input');
  if (!el) return;
  const start = el.selectionStart || el.value.length;
  const end = el.selectionEnd || el.value.length;
  const val = el.value;
  el.value = val.substring(0, start) + token + val.substring(end);
  const newPos = start + (cursorOffset !== undefined ? cursorOffset : token.length);
  el.focus();
  el.setSelectionRange(newPos, newPos);
  el.dispatchEvent(new Event('input', { bubbles: true }));
};

window.clearQuery = function() {
  const el = document.querySelector('#wb-query-box textarea') || document.querySelector('#wb-query-box input');
  if (!el) return;
  el.value = '';
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.focus();
};

window.setFilter = function(type, btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const val = btn.getAttribute('data-val');

  let targetId = '';
  if (type === 'scope') targetId = '#hidden-scope-input';
  else if (type === 'mode') targetId = '#hidden-mode-input';
  else if (type === 'season') targetId = '#hidden-season-input';

  const inp = document.querySelector(targetId + ' input') || document.querySelector(targetId + ' textarea');
  if (inp) {
    inp.value = val;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  }
};

window.setYear = function(yr, btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll('.year-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const inp = document.querySelector('#hidden-years-input input') || document.querySelector('#hidden-years-input textarea');
  if (inp) {
    inp.value = yr;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  }
};

window.toggleFrontPage = function(btn) {
  btn.classList.toggle('active');
  const isChecked = btn.classList.contains('active');
  const cb = document.querySelector('#hidden-fp-input input[type="checkbox"]');
  if (cb) {
    cb.checked = isChecked;
    cb.dispatchEvent(new Event('change', { bubbles: true }));
  }
};

window.setTopK = function(val, btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll('[id^="topk-"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const inp = document.querySelector('#hidden-topk-input input');
  if (inp) {
    inp.value = val;
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  }
};

window.loadPreset = function(idx) {
  const presets = [
    {
      q: '+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    },
    {
      q: '+"Theodore Roosevelt" +(policy OR conservation OR trust OR campaign OR speech)',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    },
    {
      q: '("flying machine"^2 OR aeroplane^3 OR monoplane OR biplane) AND (flight OR speed OR altitude)',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    },
    {
      q: '+("woman suffrage" OR "equal suffrage") +(convention OR ballot OR vote OR parade)',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    },
    {
      q: '+(currency OR silver OR "Wall Street") +(gold OR panic OR treasury)',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    },
    {
      q: "(\\"Halley\'s comet\\"~3 OR \\"tail of the comet\\"~2) +(astronomer OR observatory OR sky)",
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: '1910',
      season: '(any)'
    },
    {
      q: 'Roosevclt~1',
      mode: 'Query string (Lucene)',
      scope: 'Both Newspapers (Side-by-Side Comparison)',
      year: 'All',
      season: '(any)'
    }
  ];

  const p = presets[idx];
  if (!p) return;

  const qInp = document.querySelector('#wb-query-box textarea') || document.querySelector('#wb-query-box input');
  if (qInp) {
    qInp.value = p.q;
    qInp.dispatchEvent(new Event('input', { bubbles: true }));
  }

  const scopeBtn = document.querySelector(`[data-val="${p.scope}"]`);
  if (scopeBtn) window.setFilter('scope', scopeBtn);

  const modeBtn = document.querySelector(`[data-val="${p.mode}"]`);
  if (modeBtn) window.setFilter('mode', modeBtn);

  const yrBtn = document.querySelector(`[data-val="${p.year}"]`);
  if (yrBtn) window.setYear(p.year, yrBtn);

  setTimeout(() => {
    const sBtn = document.querySelector('#wb-search-btn') ? (document.querySelector('#wb-search-btn').tagName === 'BUTTON' ? document.querySelector('#wb-search-btn') : document.querySelector('#wb-search-btn button')) : null;
    if (sBtn) sBtn.click();
  }, 100);
};
</script>
"""


def analyze_query_syntax(query_str: str, mode: str) -> str:
    """Produce a real-time visual breakdown of active Lucene operators in the query."""
    if not query_str or not query_str.strip():
        return ""
    if mode != "Query string (Lucene)":
        return f'<div style="font-size: 0.85rem; color: #64748b;">⚡ Ranking Mode: <strong>{mode}</strong></div>'

    chips = []
    req_matches = re.findall(r'\+(?:\"([^\"]+)\"|([A-Za-z0-9_]+))', query_str)
    req = [m[0] or m[1] for m in req_matches]
    if req:
        chips.append(f'<span class="syntax-badge" style="background:#dcfce7; color:#15803d; border-color:#86efac;">Required (+): {html.escape(", ".join(req))}</span>')

    exc_matches = re.findall(r'\-(?:\"([^\"]+)\"|([A-Za-z0-9_]+))', query_str)
    exc = [m[0] or m[1] for m in exc_matches]
    if exc:
        chips.append(f'<span class="syntax-badge" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">Excluded (-): {html.escape(", ".join(exc))}</span>')

    slop = re.findall(r'"([^"]+)"~(\d+)', query_str)
    if slop:
        for p, s in slop:
            chips.append(f'<span class="syntax-badge" style="background:#fef3c7; color:#b45309; border-color:#fde68a;">Proximity (~{s}): "{html.escape(p)}" within {s} words</span>')

    fuzzy = re.findall(r"([A-Za-z0-9]+)~(?:(\d+))?", query_str)
    if fuzzy:
        for term, d in fuzzy:
            dist = d if d else "auto"
            chips.append(f'<span class="syntax-badge" style="background:#ede9fe; color:#6d28d9; border-color:#ddd6fe;">Fuzzy (~{dist}): {html.escape(term)}</span>')

    boost = re.findall(r"([A-Za-z0-9_\"]+)\^([0-9\.]+)", query_str)
    if boost:
        for term, b in boost:
            chips.append(f'<span class="syntax-badge" style="background:#e0f2fe; color:#0369a1; border-color:#bae6fd;">Boosted (x{b}): {html.escape(term)}</span>')

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
    chunk_idx = int(fields.get("chunk_index", 0)) + 1
    total_chunks = int(fields.get("chunk_count", 1))
    raw_text = fields.get("text", "")
    highlighted_html = hl.highlight_text(raw_text, query_str) if query_str else html.escape(raw_text[:400]) + "..."

    np_slug = fields.get("newspaper_slug", "los_angeles_herald")
    if np_slug == "the_evening_world":
        badge_html = '<span style="background: #e0f2fe; color: #0369a1; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #bae6fd;">🗽 The Evening World (NY)</span>'
        accent_color = "#0284c7"
    elif np_slug == "the_sun":
        badge_html = '<span style="background: #fef08a; color: #854d0e; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #fde047;">☀️ The Sun (NY)</span>'
        accent_color = "#ca8a04"
    elif np_slug == "the_beatrice_daily_express":
        badge_html = '<span style="background: #ecfccb; color: #3f6212; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #d9f99d;">🌾 Beatrice Express (NE)</span>'
        accent_color = "#65a30d"
    elif np_slug == "chicago_eagle":
        badge_html = '<span style="background: #f3e8ff; color: #6b21a8; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #d8b4fe;">🦅 Chicago Eagle (IL)</span>'
        accent_color = "#9333ea"
    elif np_slug == "the_san_francisco_call":
        badge_html = '<span style="background: #ccfbf1; color: #115e59; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #99f6e4;">🌁 SF Call (CA)</span>'
        accent_color = "#0d9488"
    else:
        badge_html = '<span style="background: #fef3c7; color: #92400e; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid #fde68a;">🌴 LA Herald (CA)</span>'
        accent_color = "#d97706"

    file_path = fields.get("file_path") or fields.get("image_url", "")
    raw_thumb = fields.get("iiif_thumb_url") or ""
    thumb_url = normalize_iiif_url(raw_thumb or file_path, size="600,")
    high_res_url = normalize_iiif_url(file_path or raw_thumb, size="1600,")
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


def render_latency_hud(metrics: Dict[str, Any], total_hits: int, la_hits: int, ny_hits: int, mode: str, ns_count: int) -> str:
    """Render a prominent real-time latency and throughput HUD."""
    total_ms = metrics.get("total_ms", 0.0)
    pinecone_ms = metrics.get("pinecone_ms", 0.0)
    embed_ms = metrics.get("embed_ms", 0.0)
    is_cached = metrics.get("cached", False)

    cache_pill = ""
    if is_cached:
        cache_pill = '<span style="background: #065f46; color: #6ee7b7; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700; border: 1px solid #10b981;">⚡ IN-MEMORY CACHE HIT (&lt;1ms)</span>'

    embed_pill = ""
    if embed_ms > 0:
        embed_pill = f'<span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #38bdf8;">🤖 Embedding: <strong>{embed_ms:.0f}ms</strong></span>'

    return f"""
    <div style="display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; background: #0f172a; color: #f8fafc; padding: 0.75rem 1.1rem; border-radius: 8px; margin-bottom: 1rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; border-left: 4px solid #38bdf8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <span style="color: #38bdf8; font-weight: 800; letter-spacing: 0.05em;">⚡ PINECONE METRICS</span>
      {cache_pill}
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #f1f5f9;">⏱️ Total Latency: <strong style="color: #38bdf8;">{total_ms:.0f}ms</strong> ({total_ms/1000.0:.2f}s)</span>
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #e2e8f0;">🌲 Pinecone Parallel Query: <strong style="color: #4ade80;">{pinecone_ms:.0f}ms</strong></span>
      {embed_pill}
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #fde047;">📚 Total Hits: <strong>{total_hits}</strong></span>
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #c084fc;">🎯 Mode: <strong>{mode}</strong></span>
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px; color: #cbd5e1;">🗂️ Namespaces: <strong>{ns_count}</strong></span>
    </div>
    """


def render_chapter_strip_html(active_idx: int = 0) -> str:
    """Render a horizontal chapter strip for interactive story navigation."""
    cards_html = []
    for idx, c in enumerate(sd.CHAPTERS):
        active_cls = " active" if idx == active_idx else ""
        short_title = c["title"].split(":")[0] if ":" in c["title"] else c["title"]
        card = f"""
        <button type="button" class="chapter-card{active_cls}" data-idx="{idx}" onclick="selectChapter({idx})">
          <span class="ch-badge">Ch. {c['number']} • {c['period']}</span>
          <span class="ch-title">{html.escape(short_title)}</span>
          <span class="ch-meta">{html.escape(c.get('badge', ''))}</span>
        </button>
        """
        cards_html.append(card)

    return f"""
    <div style="margin-bottom: 0.75rem;">
      <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 0.35rem;">
        🏛️ Chronological Story Chapters (Click to explore):
      </div>
      <div class="chapter-strip" id="chapter-strip-container">
        {''.join(cards_html)}
      </div>
    </div>
    """


def render_workbench_filter_html() -> str:
    """Render the unified, modern HTML Filter Hub for Archival Research Workbench."""
    years_html = []
    for yr in config.TARGET_YEARS:
        years_html.append(f'<button type="button" class="year-btn" data-val="{yr}" onclick="setYear(\'{yr}\', this)">{yr}</button>')

    return f"""
    <div class="archive-filter-hub">
      <!-- Row 0: Quick Operators Toolbar -->
      <div style="margin-bottom: 0.8rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.65rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
          <span class="group-label">🛠️ Lucene Quick Operators</span>
          <span class="filter-hint">Inserts token syntax directly at cursor in query box</span>
        </div>
        <div class="chip-rail">
          <button type="button" class="op-chip" onclick="insertSyntax('+')"><strong>+must</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('-')"><strong>-exclude</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('&quot;&quot;', 1)"><strong>&quot;exact phrase&quot;</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('&quot;&quot;~4', 3)"><strong>&quot;phrase&quot;~4 slop</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('~1')"><strong>term~1 fuzzy</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('^2')"><strong>term^2 boost</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax('/pattern.*/')"><strong>/regex/</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax(' AND ')"><strong>AND</strong></button>
          <button type="button" class="op-chip" onclick="insertSyntax(' OR ')"><strong>OR</strong></button>
          <button type="button" class="op-chip clear-chip" onclick="clearQuery()"><strong>Clear ✕</strong></button>
        </div>
      </div>

      <!-- Row 1: Scope & Mode -->
      <div class="filter-row">
        <div class="filter-group">
          <label class="group-label">📰 Newspaper Scope</label>
          <div class="seg-control" id="scope-selector">
            <button type="button" class="seg-btn active" data-val="Both Newspapers (Side-by-Side Comparison)" onclick="setFilter('scope', this)">
              <span>⚖️</span> Both Newspapers <span class="tag">Side-by-Side</span>
            </button>
            <button type="button" class="seg-btn" data-val="Los Angeles Herald (California, West Coast)" onclick="setFilter('scope', this)">
              <span>🌴</span> LA Herald <span class="tag">West Coast</span>
            </button>
            <button type="button" class="seg-btn" data-val="The Evening World (New York, East Coast)" onclick="setFilter('scope', this)">
              <span>🗽</span> Evening World <span class="tag">East Coast</span>
            </button>
          </div>
        </div>

        <div class="filter-group">
          <label class="group-label">🎯 Ranking Engine Mode</label>
          <div class="seg-control" id="mode-selector">
            <button type="button" class="seg-btn active" data-val="Query string (Lucene)" onclick="setFilter('mode', this)">
              ⚡ Lucene FTS
            </button>
            <button type="button" class="seg-btn" data-val="Hybrid" onclick="setFilter('mode', this)">
              🔀 Hybrid
            </button>
            <button type="button" class="seg-btn" data-val="Full-text (BM25)" onclick="setFilter('mode', this)">
              🌲 BM25
            </button>
            <button type="button" class="seg-btn" data-val="Semantic" onclick="setFilter('mode', this)">
              🧠 Semantic
            </button>
          </div>
        </div>
      </div>

      <!-- Row 2: Year Namespaces Rail -->
      <div class="filter-row">
        <div class="filter-group full-width">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <label class="group-label">📅 Historical Years (Pinecone Namespaces)</label>
            <span class="filter-hint">Click individual year or All 9 Years</span>
          </div>
          <div class="year-rail" id="year-selector">
            <button type="button" class="year-btn active" data-val="All" onclick="setYear('All', this)">All 9 Years (1890–1910)</button>
            <span style="color:#cbd5e1; font-size:0.8rem; margin:0 2px;">|</span>
            {' '.join(years_html[:3])}
            <span style="color:#cbd5e1; font-size:0.8rem; margin:0 2px;">•</span>
            {' '.join(years_html[3:])}
          </div>
        </div>
      </div>

      <!-- Row 3: Season & Options -->
      <div class="filter-row" style="margin-bottom:0.4rem;">
        <div class="filter-group">
          <label class="group-label">❄️ Season Window</label>
          <div class="seg-control" id="season-selector">
            <button type="button" class="seg-btn active" data-val="(any)" onclick="setFilter('season', this)">(Any Season)</button>
            <button type="button" class="seg-btn" data-val="Winter" onclick="setFilter('season', this)">❄️ Winter (Jan/Dec)</button>
            <button type="button" class="seg-btn" data-val="Summer" onclick="setFilter('season', this)">☀️ Summer (Jul/Aug)</button>
          </div>
        </div>

        <div class="filter-group">
          <label class="group-label">📰 Layout &amp; Depth</label>
          <div class="seg-control" id="options-selector">
            <button type="button" class="seg-btn toggle-btn" id="fp-toggle-btn" onclick="toggleFrontPage(this)">
              📰 Front Pages Only (p.1)
            </button>
            <button type="button" class="seg-btn active" data-val="4" onclick="setTopK(4, this)" id="topk-4">Depth: 4</button>
            <button type="button" class="seg-btn" data-val="8" onclick="setTopK(8, this)" id="topk-8">Depth: 8</button>
            <button type="button" class="seg-btn" data-val="12" onclick="setTopK(12, this)" id="topk-12">Depth: 12</button>
          </div>
        </div>
      </div>

      <!-- Row 4: 1-Click Curated Presets -->
      <div style="margin-top: 0.65rem; border-top: 1px solid #f1f5f9; padding-top: 0.55rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
          <span class="group-label">🏛️ Curated Research Inquiries (1-Click Presets)</span>
          <span class="filter-hint">Populates query and executes comparison</span>
        </div>
        <div class="preset-rail">
          <button type="button" class="preset-pill" onclick="loadPreset(0)">⚡ 1906 SF Earthquake</button>
          <button type="button" class="preset-pill" onclick="loadPreset(1)">🌲 TR Western Conservation</button>
          <button type="button" class="preset-pill" onclick="loadPreset(2)">✈️ 1910 Aviation Meet</button>
          <button type="button" class="preset-pill" onclick="loadPreset(3)">🗳️ Equal Suffrage Movement</button>
          <button type="button" class="preset-pill" onclick="loadPreset(4)">💰 Free Silver vs Gold</button>
          <button type="button" class="preset-pill" onclick="loadPreset(5)">☄️ Halley's Comet 1910</button>
          <button type="button" class="preset-pill" onclick="loadPreset(6)">📜 OCR Typo Roosevclt~1</button>
        </div>
      </div>
    </div>
    """


def render_timeline_bar(la_counts: Dict[str, int], ny_counts: Dict[str, int]) -> str:

    """Render a responsive horizontal article distribution histogram across 1890-1910."""
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

    return f"""
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
    """


def render_chapter_banner(chapter: Dict[str, Any]) -> str:
    """Render historical chapter narrative context card."""
    figures_pills = " ".join([
        f'<span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; color: #334155; font-weight: 500;">{fig}</span>'
        for fig in chapter.get("figures", [])
    ])

    return f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
      <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <div>
          <span style="background: #0f172a; color: #ffffff; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Chapter {chapter['number']} of 7</span>
          <span style="background: #e2e8f0; color: #334155; padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-left: 0.4rem;">{chapter['badge']}</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-weight: 600;">Historical Period: <strong style="color: #0f172a;">{chapter['period']}</strong></div>
      </div>

      <h2 style="margin: 0.35rem 0 0.15rem 0; font-size: 1.45rem; color: #0f172a; font-family: Georgia, serif; font-weight: 700;">{chapter['title']}</h2>
      <div style="font-size: 0.95rem; color: #475569; font-weight: 500; margin-bottom: 0.85rem; font-style: italic;">{chapter['subtitle']}</div>

      <p style="font-family: Georgia, serif; font-size: 0.96rem; line-height: 1.65; color: #1e293b; margin-bottom: 1rem;">
        {chapter['narrative']}
      </p>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; margin-bottom: 0.85rem;">
        <div style="background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.7rem 0.9rem; border-radius: 4px; font-size: 0.84rem; color: #78350f; line-height: 1.5;">
          {chapter['west_angle']}
        </div>
        <div style="background: #f0f9ff; border-left: 4px solid #0284c7; padding: 0.7rem 0.9rem; border-radius: 4px; font-size: 0.84rem; color: #075985; line-height: 1.5;">
          {chapter['east_angle']}
        </div>
      </div>

      <div style="display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; font-size: 0.8rem; margin-top: 0.5rem;">
        <strong style="color: #334155;">Key Figures & Forces:</strong>
        {figures_pills}
      </div>
    </div>
    """


# ── Story Mode Execution Handlers ───────────────────────────────────────────
def resolve_chapter(chapter_label: str) -> Dict[str, Any]:
    ch_idx = 0
    for idx, c in enumerate(sd.CHAPTERS):
        if c["title"] in chapter_label or f"Chapter {c['number']}" in chapter_label:
            ch_idx = idx
            break
    return sd.CHAPTERS[ch_idx]


def on_chapter_change(chapter_label: str, mode: str, top_k: int, year_filter: str):
    """When a new chapter is selected from the radio, populate its default query and execute."""
    chapter = resolve_chapter(chapter_label)
    default_q = chapter["semantic_query"] if mode in ("Semantic", "Hybrid") else chapter["query"]
    _, banner, hud, timeline, cards = execute_story_query(chapter_label, default_q, mode, top_k, year_filter)
    return default_q, banner, hud, timeline, cards


def execute_story_query(
    chapter_label: str,
    query_str: str,
    mode: str,
    top_k: int,
    year_filter: str,
) -> Tuple[str, str, str, str, str]:
    """Execute search for a story chapter using the editable query expression."""
    if init_error or not index_client or not openai_client:
        err_html = f'<div style="padding: 1rem; color: #991b1b; background: #fee2e2; border-radius: 8px;">Missing API Keys: {init_error or "Check PINECONE_API_KEY and OPENAI_API_KEY"}</div>'
        return query_str, "", err_html, "", ""

    chapter = resolve_chapter(chapter_label)
    active_query = query_str.strip() if query_str and query_str.strip() else chapter["query"]

    # Target namespaces
    if year_filter and not year_filter.startswith("All Years"):
        namespaces = [year_filter.strip()]
    else:
        namespaces = config.TARGET_YEARS

    # Execute side-by-side search
    res = q.search_side_by_side(
        index=index_client,
        oai=openai_client,
        query=active_query,
        namespaces=namespaces,
        mode=mode,
        match_op=None,
        match_terms=None,
        top_k_per_year=int(top_k),
    )

    la_hits = res["la"]["merged"]
    ny_hits = res["ny"]["merged"]
    total_hits = len(la_hits) + len(ny_hits)
    metrics = res.get("metrics", {})

    banner_html = render_chapter_banner(chapter)
    hud_html = render_latency_hud(metrics, total_hits, len(la_hits), len(ny_hits), mode, len(namespaces))
    timeline_html = render_timeline_bar(res["la"]["year_counts"], res["ny"]["year_counts"])

    la_cards = [hit_to_card_html(h, active_query) for h in la_hits]
    ny_cards = [hit_to_card_html(h, active_query) for h in ny_hits]

    la_content = "\n".join(la_cards) if la_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in the Los Angeles Herald for this chapter.</div>'
    ny_content = "\n".join(ny_cards) if ny_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in The Evening World (New York) for this chapter.</div>'

    cards_html = f"""
    <div class="newspaper-split">
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

    return active_query, banner_html, hud_html, timeline_html, cards_html


# ── Freeform Workbench Execution Handler ──────────────────────────────────────
def run_workbench_search(
    query_str: str,
    newspaper_scope: str = "Both Newspapers (Side-by-Side Comparison)",
    mode: str = "Query string (Lucene)",
    years_val: Any = "All",
    season: str = "(any)",
    front_page: bool = False,
    top_k: Union[int, float] = 4,
    state_scope: str = "All States",
    match_label: str = "None (off)",
    month: str = "(any)",
) -> Tuple[str, str, str]:
    """Execute search query with scope, multi-year namespace, season, and front page filtering."""
    if init_error or not index_client or not openai_client:
        err_msg = f'<div style="padding: 1rem; background: #fef2f2; color: #991b1b; border-radius: 8px;">Configuration Notice: Missing API keys. {init_error}</div>'
        return err_msg, "", ""

    if not query_str or not query_str.strip():
        return '<div style="color: #64748b; padding: 0.5rem 0;">Enter a search term above to query the archive.</div>', "", ""

    match_op = MATCH_LABELS.get(match_label)
    breakdown_html = analyze_query_syntax(query_str, mode)

    # Determine target namespaces
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

    # Determine newspaper / state filter
    target_state = state_scope if state_scope != "All States" else None
    
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
        if title_sub in str(newspaper_scope):
            slug = s
            break

    # Case 1: Both Newspapers (Side-by-Side)
    if not slug and (str(newspaper_scope).startswith("Both") and target_state is None):
        base_filter = q.build_filter(
            month=month if month != "(any)" else None,
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

        la_hits = res["la"]["merged"]
        ny_hits = res["ny"]["merged"]
        total_hits = len(la_hits) + len(ny_hits)
        metrics = res.get("metrics", {})

        hud_html = render_latency_hud(metrics, total_hits, len(la_hits), len(ny_hits), mode, len(target_namespaces))
        timeline_html = render_timeline_bar(res["la"]["year_counts"], res["ny"]["year_counts"])

        la_cards = [hit_to_card_html(h, query_str) for h in la_hits]
        ny_cards = [hit_to_card_html(h, query_str) for h in ny_hits]

        la_content = "\n".join(la_cards) if la_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in the Los Angeles Herald.</div>'
        ny_content = "\n".join(ny_cards) if ny_cards else '<div style="padding: 1.5rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No matching records found in The Evening World (New York).</div>'

        cards_html = f"""
        {breakdown_html}
        <div class="newspaper-split">
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
        return hud_html, timeline_html, cards_html

    # Case 2: Multi-Newspaper / Single-State / Nationwide Scope
    scope_label = target_state or (config.NEWSPAPERS.get(slug, {}).get("title") if slug else "All Newspapers (Nationwide)")

    filt = q.build_filter(
        state=target_state,
        newspaper_slug=slug,
        month=month if month != "(any)" else None,
        season=season if season != "(any)" else None,
        front_page_only=front_page,
        match_op=match_op,
        match_terms=query_str,
    )

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

    merged_hits = search_data["merged"]
    total_count = search_data["total_hits"]
    metrics = search_data.get("metrics", {})

    hud_html = render_latency_hud(metrics, total_count, 0, 0, mode, len(target_namespaces))
    timeline_html = render_timeline_bar(search_data["year_counts"], {yr: 0 for yr in config.TARGET_YEARS})

    cards = [hit_to_card_html(hit, query_str) for hit in merged_hits]
    cards_html = f"{breakdown_html}\n" + ("\n".join(cards) if cards else f'<div style="padding: 1rem; color: #64748b;">No documents found matching "{query_str}" in {scope_label}.</div>')
    return hud_html, timeline_html, cards_html


# ── Calendar View Handlers ──────────────────────────────────────────────────
def render_calendar_gallery(res: Dict[str, Any]) -> str:
    """Render a chronological multi-year gallery of front pages."""
    by_year = res.get("by_year", {})
    years = config.TARGET_YEARS

    year_sections = []
    for yr in years:
        hits = by_year.get(yr, [])
        if not hits:
            continue

        cards = [hit_to_card_html(h, "") for h in hits]
        cards_grid = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; margin-top: 0.75rem;">
          {''.join(cards)}
        </div>
        """

        section = f"""
        <div style="margin-bottom: 2rem; border-top: 2px solid #e2e8f0; padding-top: 1rem;">
          <div style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.6rem;">
            <span style="background: #0f172a; color: #ffffff; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.95rem;">📅 {yr} Edition</span>
            <span style="font-size: 0.85rem; color: #64748b;">Front pages published on <strong>{res.get('date_label')}</strong>, {yr} ({len(hits)} issues retrieved)</span>
          </div>
          {cards_grid}
        </div>
        """
        year_sections.append(section)

    if not year_sections:
        return f'<div style="padding: 2rem; text-align: center; color: #64748b; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">No issues found in our collection for the date <strong>{res.get("date_label")}</strong> across the 9 years.</div>'

    return "\n".join(year_sections)


def run_calendar_search(
    month_choice: str,
    day_choice: str,
    state_choice: str,
    front_page_only: bool,
) -> Tuple[str, str]:
    """Search for issues on a specific calendar month and day across all years."""
    if init_error or not index_client:
        return f'<div style="color:red;">Error: {init_error}</div>', ""

    m_str = month_choice.split()[0].lstrip("0") or "1"
    d_str = day_choice.split()[0].lstrip("0") or "1"
    m_int = int(m_str) if m_str.isdigit() else 1
    d_int = int(d_str) if d_str.isdigit() else 1

    res = q.search_calendar_date(
        index=index_client,
        month=m_int,
        day=d_int,
        front_page_only=front_page_only,
        state=state_choice,
        top_k_per_year=4,
    )

    metrics = res.get("metrics", {})
    total_hits = res.get("total_hits", 0)
    la_hits = len(res.get("by_newspaper", {}).get("la", []))
    ny_hits = len(res.get("by_newspaper", {}).get("ny", []))

    hud_html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; background: #0f172a; color: #f8fafc; padding: 0.75rem 1.1rem; border-radius: 8px; margin-bottom: 1rem; font-family: monospace; font-size: 0.85rem; border-left: 4px solid #38bdf8;">
      <span style="color: #38bdf8; font-weight: 800;">📅 ON THIS DAY IN HISTORY:</span>
      <span>Date: <strong style="color:#fde047;">{m_int:02d}-{d_int:02d}</strong> across 9 Year Namespaces</span>
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px;">⏱️ Latency: <strong>{metrics.get('total_ms', 0):.0f}ms</strong></span>
      <span style="background: #1e293b; padding: 0.2rem 0.55rem; border-radius: 4px;">📚 Retrieved: <strong>{total_hits}</strong> pages (<span style="color:#fbbf24;">{la_hits} LA</span> / <span style="color:#60a5fa;">{ny_hits} NY</span>)</span>
    </div>
    """

    gallery_html = render_calendar_gallery(res)
    return hud_html, gallery_html


# Helper function to append Lucene tokens to the query box
def insert_token(current_query: str, token: str) -> Tuple[str, str]:
    new_query = f"{current_query.strip()} {token}".strip() if current_query.strip() else token
    return new_query, "Query string (Lucene)"


# ── Gradio Blocks Interface ─────────────────────────────────────────────────
chapter_choices = [f"Ch. {c['number']}: {c['title']} ({c['period']})" for c in sd.CHAPTERS]

with gr.Blocks(title="Coast-to-Coast Historical Newspaper Search (1890–1910)") as demo:
    demo.head = CUSTOM_JS
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.Markdown(
        """
        # 📰 Coast-to-Coast Historical Newspaper Search (1890–1910)
        ### Cross-Country Comparative Archive • Library of Congress Chronicling America & Pinecone Documents API
        Experience **20 transformative years of American history** contrasting **🌴 Los Angeles Herald** (West Coast) and **🗽 The Evening World** (New York, East Coast) using **Pinecone Hybrid & Lucene Query Engines**, **Dense Semantic Vectors**, and **Library of Congress IIIF Dynamic Image CDN**.
        """
    )

    with gr.Tabs() as tabs:
        # ─────────────────────────────────────────────────────────────────────
        # TAB 1: INTERACTIVE STORY MODE (7 HISTORICAL CHAPTERS)
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("🏛️ Interactive Story Mode (Coast-to-Coast 1890–1910)", id="tab_story"):
            gr.HTML(value=render_chapter_strip_html(0))

            chapter_radio = gr.Radio(
                choices=chapter_choices,
                value=chapter_choices[0],
                label="Chronological Story Chapters (Click or use cards above to select)",
                elem_id="hidden-story-chapter-radio",
            )

            # Editable Query Input for Story Mode
            with gr.Row():
                with gr.Column(scale=5):
                    story_query_input = gr.Textbox(
                        value=sd.CHAPTERS[0]["query"],
                        label="Active Story Query Expression (Editable)",
                        placeholder="Edit the query expression to test historical hypotheses...",
                        lines=2,
                    )
                with gr.Column(scale=1, min_width=140):
                    story_run_btn = gr.Button("⚡ Execute Story Query", variant="primary", size="lg")

            with gr.Row():
                story_mode_choice = gr.Radio(
                    choices=MODES,
                    value="Query string (Lucene)",
                    label="Ranking Engine Mode",
                )
                story_year_filter = gr.Dropdown(
                    choices=["All Years (Cross-Year Comparison)"] + config.TARGET_YEARS,
                    value="All Years (Cross-Year Comparison)",
                    label="Time Walker Scope",
                )
                story_top_k = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=4,
                    step=1,
                    label="Results per Year Namespace",
                )

            # Story Mode Outputs
            story_banner_out = gr.HTML(label="Chapter Overview")
            story_hud_out = gr.HTML(label="Pinecone Latency HUD")
            story_timeline_out = gr.HTML(label="Article Density Distribution")
            story_cards_out = gr.HTML(label="Side-by-Side Newspaper Cards")

            # Changing chapter updates query box and executes
            chapter_radio.change(
                fn=on_chapter_change,
                inputs=[chapter_radio, story_mode_choice, story_top_k, story_year_filter],
                outputs=[story_query_input, story_banner_out, story_hud_out, story_timeline_out, story_cards_out],
            )

            # Clicking Execute or pressing enter runs custom query
            story_run_inputs = [chapter_radio, story_query_input, story_mode_choice, story_top_k, story_year_filter]
            story_run_outputs = [story_query_input, story_banner_out, story_hud_out, story_timeline_out, story_cards_out]

            story_run_btn.click(fn=execute_story_query, inputs=story_run_inputs, outputs=story_run_outputs)
            story_query_input.submit(fn=execute_story_query, inputs=story_run_inputs, outputs=story_run_outputs)

            story_mode_choice.change(fn=execute_story_query, inputs=story_run_inputs, outputs=story_run_outputs)
            story_year_filter.change(fn=execute_story_query, inputs=story_run_inputs, outputs=story_run_outputs)
            story_top_k.change(fn=execute_story_query, inputs=story_run_inputs, outputs=story_run_outputs)

        # ─────────────────────────────────────────────────────────────────────
        # TAB 2: ARCHIVAL RESEARCH WORKBENCH (ADVANCED SEARCH)
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("🔬 Archival Research Workbench (Advanced Search)", id="tab_workbench"):
            with gr.Row():
                with gr.Column(scale=5):
                    query_input = gr.Textbox(
                        value='+"San Francisco" +(earthquake OR catastrophe OR fire OR ruins) -patent -remedy',
                        label="Search Query Expression (Supports full Lucene FTS syntax, boolean logic, fuzzy matching)",
                        placeholder="e.g. '+\"Theodore Roosevelt\" +(speech OR visit) -advertisement' or '\"aviation meet\"~4'",
                        lines=2,
                        elem_id="wb-query-box",
                    )
                with gr.Column(scale=1, min_width=140):
                    search_btn = gr.Button("⚡ Search Archive", variant="primary", size="lg", elem_id="wb-search-btn")

            # Custom High-Density HTML Filter Hub
            gr.HTML(value=render_workbench_filter_html())

            # Reactive Hidden State Bridge for Gradio callbacks
            with gr.Row(elem_classes=["hidden-state-bridge"]):
                hidden_scope = gr.Textbox(value="Both Newspapers (Side-by-Side Comparison)", elem_id="hidden-scope-input")
                hidden_mode = gr.Textbox(value="Query string (Lucene)", elem_id="hidden-mode-input")
                hidden_years = gr.Textbox(value="All", elem_id="hidden-years-input")
                hidden_season = gr.Textbox(value="(any)", elem_id="hidden-season-input")
                hidden_fp = gr.Checkbox(value=False, elem_id="hidden-fp-input")
                hidden_topk = gr.Number(value=4, elem_id="hidden-topk-input")

            # Curated Inquiries Accordion
            with gr.Accordion("🏛️ Curated Inquiries: Library of Congress Digital Strategy Showcase", open=False):
                showcase_names = [f"[{item['category'].split()[0]}] {item['title']}: {item['query']}" for item in SHOWCASE_QUERIES]
                showcase_dropdown = gr.Dropdown(
                    choices=showcase_names,
                    label="Select a Curated Research Inquiry",
                    value=showcase_names[0],
                )
                showcase_desc = gr.Markdown(
                    value=f"**Archival Challenge:** {SHOWCASE_QUERIES[0]['desc']}\n\n**Query:** `{SHOWCASE_QUERIES[0]['query']}`"
                )
                load_run_btn = gr.Button("🚀 Load & Execute Inquiry", variant="secondary")

            wb_hud_out = gr.HTML(label="Pinecone Latency HUD")
            wb_timeline_out = gr.HTML(label="Article Density Distribution")
            wb_results_out = gr.HTML(label="Search Results")

            def update_showcase_desc(selected_label: str) -> str:
                idx = showcase_names.index(selected_label) if selected_label in showcase_names else 0
                item = SHOWCASE_QUERIES[idx]
                return f"**Archival Challenge & Rationale:** {item['desc']}\n\n**Active Expression:** `{item['query']}`"

            showcase_dropdown.change(fn=update_showcase_desc, inputs=[showcase_dropdown], outputs=[showcase_desc])

            def load_showcase_query(selected_label: str):
                idx = showcase_names.index(selected_label) if selected_label in showcase_names else 0
                item = SHOWCASE_QUERIES[idx]
                q_str = item["query"]
                scope_val = item.get("newspaper_scope", NEWSPAPER_OPTIONS[0])
                mode_val = item["mode"]
                yrs_val = item.get("years", config.TARGET_YEARS)
                season_val = item["season"]
                fp_val = item["front_page"]
                top_k_val = item["top_k"]

                hud, timeline, cards = run_workbench_search(
                    query_str=q_str,
                    newspaper_scope=scope_val,
                    mode=mode_val,
                    years_val=yrs_val,
                    season=season_val,
                    front_page=fp_val,
                    top_k=top_k_val,
                )
                yrs_str = "All" if len(yrs_val) == len(config.TARGET_YEARS) else (yrs_val[0] if len(yrs_val) == 1 else ",".join(yrs_val))
                return q_str, scope_val, mode_val, yrs_str, season_val, fp_val, top_k_val, hud, timeline, cards

            load_run_btn.click(
                fn=load_showcase_query,
                inputs=[showcase_dropdown],
                outputs=[
                    query_input,
                    hidden_scope,
                    hidden_mode,
                    hidden_years,
                    hidden_season,
                    hidden_fp,
                    hidden_topk,
                    wb_hud_out,
                    wb_timeline_out,
                    wb_results_out,
                ],
            )

            wb_inputs = [
                query_input,
                hidden_scope,
                hidden_mode,
                hidden_years,
                hidden_season,
                hidden_fp,
                hidden_topk,
            ]
            wb_outputs = [wb_hud_out, wb_timeline_out, wb_results_out]

            search_btn.click(fn=run_workbench_search, inputs=wb_inputs, outputs=wb_outputs)
            query_input.submit(fn=run_workbench_search, inputs=wb_inputs, outputs=wb_outputs)

        # ─────────────────────────────────────────────────────────────────────
        # TAB 3: 📅 CALENDAR VIEW ("ON THIS DAY IN HISTORY")
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("📅 Calendar View ('On This Day in History')", id="tab_calendar"):
            gr.Markdown(
                """
                ### 📅 On This Day in History: Cross-Year Date Archive Explorer
                Select any calendar date below to retrieve and visually compare front pages published on that exact day across all **9 years (1890–1910)** in **California** (*Los Angeles Herald*) and **New York** (*The Evening World*).
                """
            )

            gr.Markdown("**Quick Date Jump (Archived Landmark Dates):**")
            with gr.Row():
                btn_d1 = gr.Button("🎉 Jan 01 (New Year's Day)", size="sm")
                btn_d2 = gr.Button("📰 Jan 02", size="sm")
                btn_d3 = gr.Button("📰 Jan 03", size="sm")
                btn_d4 = gr.Button("☀️ Jul 01", size="sm")
                btn_d5 = gr.Button("☀️ Jul 02", size="sm")
                btn_d6 = gr.Button("🇺🇸 Jul 04 (Independence Day)", size="sm")
                btn_d7 = gr.Button("❄️ Dec 01", size="sm")
                btn_d8 = gr.Button("❄️ Dec 02", size="sm")

            with gr.Row():
                cal_month_dropdown = gr.Dropdown(
                    choices=CALENDAR_MONTHS,
                    value=CALENDAR_MONTHS[0],
                    label="Calendar Month",
                )
                cal_day_dropdown = gr.Dropdown(
                    choices=CALENDAR_DAYS,
                    value="02",
                    label="Calendar Day",
                )
                cal_state_dropdown = gr.Dropdown(
                    choices=STATE_OPTIONS,
                    value=STATE_OPTIONS[0],
                    label="State Filter",
                )
                cal_front_page_cb = gr.Checkbox(
                    value=True,
                    label="Front Pages Only (Page 1)",
                )
                cal_search_btn = gr.Button("📅 Explore Calendar Date Across All Years", variant="primary", size="lg")

            cal_hud_out = gr.HTML(label="Calendar Latency HUD")
            cal_gallery_out = gr.HTML(label="Multi-Year Front Page Gallery")

            cal_inputs = [cal_month_dropdown, cal_day_dropdown, cal_state_dropdown, cal_front_page_cb]
            cal_outputs = [cal_hud_out, cal_gallery_out]

            cal_search_btn.click(fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs)

            # Wire Quick Date Buttons
            btn_d1.click(fn=lambda: ("01 - January", "01"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d2.click(fn=lambda: ("01 - January", "02"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d3.click(fn=lambda: ("01 - January", "03"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d4.click(fn=lambda: ("07 - July", "01"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d5.click(fn=lambda: ("07 - July", "02"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d6.click(fn=lambda: ("07 - July", "04"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d7.click(fn=lambda: ("12 - December", "01"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )
            btn_d8.click(fn=lambda: ("12 - December", "02"), outputs=[cal_month_dropdown, cal_day_dropdown]).then(
                fn=run_calendar_search, inputs=cal_inputs, outputs=cal_outputs
            )

        # ─────────────────────────────────────────────────────────────────────
        # TAB 4: REFERENCE & ARCHITECTURE
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("📖 Pinecone Architecture & Lucene Guide", id="tab_docs"):
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

    # Initial load trigger: automatically populate Chapter 1 on startup
    demo.load(
        fn=lambda: on_chapter_change(chapter_choices[0], "Query string (Lucene)", 4, "All Years (Cross-Year Comparison)"),
        inputs=None,
        outputs=[story_query_input, story_banner_out, story_hud_out, story_timeline_out, story_cards_out],
    )

if __name__ == "__main__":
    demo.launch(head=CUSTOM_JS)
