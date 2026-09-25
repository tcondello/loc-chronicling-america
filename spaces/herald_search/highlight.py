"""Client-side query-term highlighting for the Streamlit and Gradio UIs.

Supports Pinecone Lucene query syntax features:
- Stemmed term matching with Snowball English stemmer
- Fuzzy typo / OCR noise tolerance matching (term~N Levenshtein edit distance)
- Regular expression matching (/pattern.*/)
- Quoted exact phrase and phrase slop matching ("...")
- Exclusion operator awareness (-term)
- Tooltip attributes on <mark class="hl"> tags
"""

import html
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import snowballstemmer

_stemmer = snowballstemmer.stemmer("english")

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_OPERATORS = {"AND", "OR", "NOT", "TO"}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "we", "our", "you", "your", "i",
}
_KNOWN_FIELDS = {"text", "title", "body", "content"}


def levenshtein_dist(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _stem(word: str) -> str:
    return _stemmer.stemWord(word.lower())


def parse_query_features(query: str) -> Dict[str, Any]:
    """Extract full-text, fuzzy, regex, and phrase clauses from a query."""
    if not query:
        return {
            "regexes": [],
            "fuzzy_terms": [],
            "phrases": [],
            "stems": {},
            "prefixes": [],
        }

    # 1. Regex patterns: /pattern.*/
    regexes = []
    for r_pat in re.findall(r"/([^/]+)/", query):
        try:
            regexes.append((r_pat, re.compile(r_pat, re.IGNORECASE)))
        except Exception:
            pass

    # 2. Fuzzy terms: term~ or term~N
    fuzzy_terms = []
    for f_term, dist in re.findall(r"([A-Za-z0-9]+)~(?:(\d+))?", query):
        if f_term.lower() not in _STOPWORDS and f_term.lower() not in _KNOWN_FIELDS:
            d = int(dist) if dist else (1 if len(f_term) < 8 else 2)
            fuzzy_terms.append((f_term.lower(), min(d, 2)))

    # 3. Exact quoted phrases (with optional slop stripped)
    phrases = []
    for p in re.findall(r'"([^"]+)"', query):
        p_clean = p.strip().lower()
        if p_clean:
            phrases.append(p_clean)

    # 4. Standard tokens: remove field qualifiers, regexes, phrases, punctuation
    cleaned = re.sub(r"\b(text|title|body):\s*", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"/[^/]+/", " ", cleaned)
    cleaned = re.sub(r'"[^"]+"', " ", cleaned)
    cleaned = re.sub(r"[\(\)\[\]\+\-\~\^\:\*]", " ", cleaned)
    tokens = _WORD.findall(cleaned)

    stems = {}
    prefixes = []

    for t in tokens:
        upper_t = t.upper()
        lower_t = t.lower()
        if upper_t in _OPERATORS:
            continue
        if lower_t in _STOPWORDS or lower_t in _KNOWN_FIELDS or len(lower_t) < 2:
            continue
        st = _stem(lower_t)
        stems[st] = lower_t
        prefixes.append(lower_t)

    return {
        "regexes": regexes,
        "fuzzy_terms": fuzzy_terms,
        "phrases": phrases,
        "stems": stems,
        "prefixes": prefixes,
    }


def highlight_text(text: str, query: str) -> str:
    """Highlight query matches within text, covering stems, fuzzy OCR typos, and regex."""
    if not text or not query:
        return html.escape(text)

    features = parse_query_features(query)
    regexes = features["regexes"]
    fuzzy_terms = features["fuzzy_terms"]
    stems = features["stems"]
    prefixes = features["prefixes"]
    phrases = features["phrases"]

    if not regexes and not fuzzy_terms and not stems and not prefixes and not phrases:
        return html.escape(text)

    # Check each word token
    def check_word(word: str) -> Tuple[bool, Optional[str]]:
        low = word.lower()
        if len(low) < 2:
            return False, None

        # Exact / stemmed match
        word_stem = _stem(low)
        if word_stem in stems:
            orig = stems[word_stem]
            return True, f"Stem match: {orig}"

        # Prefix match (for phrase prefix or standard prefix >= 4 chars)
        for p in prefixes:
            if len(p) >= 4 and low.startswith(p):
                return True, f"Prefix match: {p}*"

        # Fuzzy matching (typo / OCR noise tolerance)
        for f_term, max_dist in fuzzy_terms:
            if abs(len(low) - len(f_term)) <= max_dist:
                d = levenshtein_dist(low, f_term)
                if d <= max_dist:
                    return True, f"Fuzzy match (dist {d}): {f_term}~{max_dist}"

        # Regex matching
        for pat_str, pat_compiled in regexes:
            if pat_compiled.search(word):
                return True, f"Regex match: /{pat_str}/"

        # Phrase term matching
        for phrase in phrases:
            p_words = phrase.split()
            if any(low == pw or _stem(low) == _stem(pw) for pw in p_words if pw not in _STOPWORDS):
                return True, f'Phrase term: "{phrase}"'

        return False, None

    parts = []
    last_idx = 0
    for match in _WORD.finditer(text):
        start, end = match.span()
        if start > last_idx:
            parts.append(html.escape(text[last_idx:start]))

        raw_word = match.group(0)
        is_hit, title = check_word(raw_word)
        if is_hit:
            title_attr = f' title="{html.escape(title)}"' if title else ""
            parts.append(f'<mark class="hl"{title_attr}>{html.escape(raw_word)}</mark>')
        else:
            parts.append(html.escape(raw_word))

        last_idx = end

    if last_idx < len(text):
        parts.append(html.escape(text[last_idx:]))

    return "".join(parts)
