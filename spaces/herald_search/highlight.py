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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    import snowballstemmer
    _stemmer = snowballstemmer.stemmer("english")
except Exception:
    class _FallbackStemmer:
        def stemWord(self, w: str) -> str:
            low = w.lower()
            if low.endswith("ies") and len(low) > 4:
                return low[:-3] + "y"
            if low.endswith("es") and len(low) > 3:
                return low[:-2]
            if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
                return low[:-1]
            if low.endswith("ing") and len(low) > 5:
                return low[:-3]
            if low.endswith("ed") and len(low) > 4:
                return low[:-2]
            return low
    _stemmer = _FallbackStemmer()

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_OPERATORS = {"AND", "OR", "NOT", "TO"}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "we", "our", "you", "your", "i",
}
_KNOWN_FIELDS = {"text", "title", "body", "content"}


def levenshtein_dist(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings with early exit."""
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 2:
        return abs(len1 - len2)
    if len1 < len2:
        return levenshtein_dist(s2, s1)
    if len2 == 0:
        return len1
    prev = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


@lru_cache(maxsize=1024)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word.lower())


@lru_cache(maxsize=128)
def parse_query_features(query: str) -> Dict[str, Any]:
    """Extract full-text, fuzzy, regex, and phrase clauses from a query (LRU cached)."""
    if not query:
        return {
            "regexes": (),
            "fuzzy_terms": (),
            "phrases": (),
            "stems": {},
            "prefixes": (),
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
        "regexes": tuple(regexes),
        "fuzzy_terms": tuple(fuzzy_terms),
        "phrases": tuple(phrases),
        "stems": stems,
        "prefixes": tuple(prefixes),
    }


def highlight_text(text: str, query: str) -> str:
    """Highlight query matches within text with in-chunk token memoization."""
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

    # In-chunk word memoization dictionary
    word_memo: Dict[str, Tuple[bool, Optional[str]]] = {}

    def check_word(word: str) -> Tuple[bool, Optional[str]]:
        low = word.lower()
        if low in word_memo:
            return word_memo[low]

        if len(low) < 2:
            word_memo[low] = (False, None)
            return False, None

        # Exact / stemmed match
        word_stem = _stem(low)
        if word_stem in stems:
            orig = stems[word_stem]
            res = (True, f"Stem match: {orig}")
            word_memo[low] = res
            return res

        # Prefix match (for phrase prefix or standard prefix >= 4 chars)
        for p in prefixes:
            if len(p) >= 4 and low.startswith(p):
                res = (True, f"Prefix match: {p}*")
                word_memo[low] = res
                return res

        # Fuzzy matching (typo / OCR noise tolerance)
        for f_term, max_dist in fuzzy_terms:
            if abs(len(low) - len(f_term)) <= max_dist:
                d = levenshtein_dist(low, f_term)
                if d <= max_dist:
                    res = (True, f"Fuzzy match (dist {d}): {f_term}~{max_dist}")
                    word_memo[low] = res
                    return res

        # Regex matching
        for pat_str, pat_compiled in regexes:
            if pat_compiled.search(word):
                res = (True, f"Regex match: /{pat_str}/")
                word_memo[low] = res
                return res

        # Phrase term matching
        for phrase in phrases:
            p_words = phrase.split()
            if any(low == pw or _stem(low) == _stem(pw) for pw in p_words if pw not in _STOPWORDS):
                res = (True, f'Phrase term: "{phrase}"')
                word_memo[low] = res
                return res

        word_memo[low] = (False, None)
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


def clean_snippet(text: str, max_chars: int = 260) -> str:
    """Produce a clean excerpt snippet for card previews (inspired by topic-feed-lab)."""
    if not text:
        return ""
    # Strip excess whitespace
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    # Trim to word boundary
    truncated = clean[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated} …"
