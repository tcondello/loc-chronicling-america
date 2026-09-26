---
title: Coast-to-Coast Historical Newspaper Search (1890–1910)
emoji: 📰
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# 📰 Coast-to-Coast & Heartland Historical Newspaper Search (1890–1910)
### Multi-State Comparative Archive • Library of Congress Chronicling America & Pinecone Documents API

Search, compare, and visually explore 20 pivotal years of American newsprint across 6 iconic newspapers from 4 states:
- 🌴 **Los Angeles Herald** (California, West Coast • LCCN `sn85042462`)
- 🌁 **The San Francisco Call** (California, Bay Area • LCCN `sn85066387`)
- 🗽 **The Evening World** (New York, East Coast • LCCN `sn83030193`)
- ☀️ **The Sun** (New York, East Coast • LCCN `sn83030272`)
- 🌾 **The Beatrice Daily Express** (Nebraska, Heartland • LCCN `sn84020107`)
- 🦅 **Chicago Eagle** (Illinois, Midwest • LCCN `sn84025828`)

Powered by **Pinecone Document Schema Full-Text Search (BM25 & Lucene)**, **OpenAI Dense Vectors (`text-embedding-3-small`)**, **Chonkie Recursive Chunking**, and dynamic **Library of Congress IIIF Dynamic Image CDN**. Containing **46,846 indexed chunks** across 9 year namespaces.

---

## ⚡ Cross-Country Comparative Analysis

By pairing matching issues from California and New York across the same 9 years (`1890`–`1892`, `1905`–`1910`) and matching dates, researchers can examine how coastal perspectives diverged on epochal historical moments:

1. **1906 San Francisco Earthquake & Fire**:
   * *West Coast*: Local tremors in Southern California, emergency refugee trains, and immediate relief supply trains dispatched from Los Angeles.
   * *East Coast*: Wall Street financial shocks, severed transcontinental telegraph lines, insurance liabilities, and federal aid mobilization.
2. **Theodore Roosevelt Administration**:
   * *West Coast*: Western public lands, national parks, forest reserves, and reclamation projects.
   * *East Coast*: New York state politics, Tammany Hall political machines, and Wall Street trust-busting.
3. **The Dawn of American Aviation (1908–1910)**:
   * *West Coast*: Landmark coverage of the January 1910 Los Angeles International Air Meet at Dominguez Field.
   * *East Coast*: Hudson-Fulton celebration flights, Glenn Curtiss demonstrations, and European aviator tours.
4. **Progressive Era Equal Suffrage Movement**:
   * *West Coast*: Mobilization toward California's historic 1911 constitutional amendment granting women the right to vote.
   * *East Coast*: Massive municipal organizing, labor alliances, and Fifth Avenue suffrage parades in New York City.
5. **1890s Monetary Crises**:
   * *West Coast*: Populist free silver monetization and bimetallism advocacy.
   * *East Coast*: Wall Street gold standard orthodoxy and banking treasury reserves.

---

## Search Modes & Capabilities

| Mode | Scoring Method | Description |
| :--- | :--- | :--- |
| **Query string (Lucene)** | `query_string` | Full Lucene boolean expressions (`AND`, `OR`, `NOT`, `+`, `-`), phrase matching (`"..."`), proximity slop (`~N`), and term boosts (`^N`). |
| **Hybrid** | `dense_vector` + `$match_*` | OpenAI semantic cosine ranking combined with a strict lexical hard filter (`$match_all`, `$match_phrase`, or `$match_any`). |
| **Full-text (BM25)** | `text` | Native Pinecone BM25 lexical token-OR ranking with English Snowball stemming. |
| **Semantic** | `dense_vector` | 1536-dimensional dense vector similarity via OpenAI `text-embedding-3-small`. |

---

## Visual Page Rendering via Library of Congress IIIF CDN

Historical newspapers were digitized as raw JPEG 2000 (`.jp2`) master scans. Rather than downloading or decoding 20 MB scans locally:
* Every result translates deterministically to a **lightweight, CDN-backed IIIF endpoint** on `tile.loc.gov`.
* Thumbnails load in ~80ms at `/full/600,/0/default.jpg`.
* High-res previews load dynamically at `/full/1600,/0/default.jpg`.
* Direct deep links connect back to authoritative records on `loc.gov` and original page PDFs.

---

## Index & Data Architecture

* **Unified Index, 9 Year Namespaces:** A single Pinecone index (`herald-hybrid-fts`) partitioned into per-year namespaces (`"1890"` through `"1910"`).
* **Multi-Newspaper Partitioning:** Chunks share the same year namespaces and are distinguished by indexed metadata (`newspaper_slug`, `newspaper_title`, `city`, `state`, `region`).
* **Structured IDs:**
  - *LA Herald*: `herald#<YYYY-MM-DD>#ed-<edition>#p<seq>#c<chunk_idx>`
  - *Evening World*: `evening_world#<YYYY-MM-DD>#ed-<edition>#p<seq>#c<chunk_idx>`
* **Chonkie Recursive Chunking:** Pages (~5,000 words) are recursively chunked into ~500-token blocks using `chonkie.RecursiveChunker(tokenizer="cl100k_base")`.
* **Pruned Metadata:** Heavy ALTO XML computer-vision arrays are omitted, keeping metadata payload under 1 KB per chunk.
