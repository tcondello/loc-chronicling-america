"""Lucene-like query language compiled to DuckDB predicates.

The Hugging Face Parquet files do not speak Lucene. This compiler accepts a
small Lucene-like dialect and turns it into parameterized SQL over the OCR
``text`` column (``ILIKE`` and RE2 ``regexp_matches``).

Space means AND. Classic Lucene defaults to OR; story search needs both
words. Use ``OR`` when either word is enough. Quotes are a phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


class ExploreQueryError(ValueError):
    """The query text is not in the supported dialect."""


FIELDS = {"year", "month", "day", "page", "sequence", "state", "paper", "newspaper", "city"}

_OPS = {"AND", "OR", "NOT"}


@dataclass
class CompiledQuery:
    """A text predicate plus field filters pulled out of the query."""

    where_sql: str
    params: List[object]
    explanation: str
    excerpt_regex: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    page: Optional[int] = None
    state: Optional[str] = None
    paper: Optional[str] = None
    city: Optional[str] = None


@dataclass
class _Term:
    value: str
    prefix: bool = False


@dataclass
class _Phrase:
    value: str


@dataclass
class _Regex:
    value: str


@dataclass
class _Field:
    name: str
    value: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class _Not:
    child: "_Node"


@dataclass
class _And:
    children: List["_Node"]


@dataclass
class _Or:
    children: List["_Node"]


_Node = Union[_Term, _Phrase, _Regex, _Field, _Not, _And, _Or]


@dataclass
class _Tok:
    kind: str
    value: str = ""


def query_help() -> str:
    return """\
Search dialect (Lucene-like, compiled to SQL)

  buried alive              both words, anywhere in the OCR (AND)
  "buried alive"            that phrase, in order
  haunted OR ghost          either word
  witch -hazel              witch, and not hazel
  +haunted -opera           same idea with required / excluded terms
  (haunted OR ghost) house  group, then AND with house
  haunt*                    substring prefix (a bare word already matches inside a longer word)
  /buried aliv[ea]/         RE2 regular expression, case-insensitive

Field filters (kept beside AND, not inside OR or NOT):

  year:1890                 one year
  year:[1890 TO 1895]       inclusive range
  month:10                  October
  page:1                    page sequence
  paper:the_evening_world   paper slug
  state:new_york            state folder
  city:"new york"           city

Not available from Lucene: fuzzy (~), boost (^), single-character ?,
proximity ("phrase"~5), and BM25 ranking. Hits come back in date order.
A bare word is a case-insensitive substring, which suits broken OCR.
"""


def compile_query(source: str) -> CompiledQuery:
    text = (source or "").strip()
    if not text:
        raise ExploreQueryError("Enter a word, phrase, or /regex/ to search.")
    tokens = _tokenize(text)
    node, pos = _parse_or(tokens, 0)
    if pos != len(tokens):
        raise ExploreQueryError(f"Unexpected text near '{tokens[pos].value or tokens[pos].kind}'.")
    filters = _Filters()
    sql, params, explanation, positives = _lower(node, filters, under_or=False, under_not=False)
    if not sql:
        raise ExploreQueryError("Add a word, phrase, or /regex/ in addition to year, paper, or city filters.")
    return CompiledQuery(
        where_sql=sql,
        params=params,
        explanation=explanation,
        excerpt_regex=_excerpt_regex(positives),
        year_from=filters.year_from,
        year_to=filters.year_to,
        month=filters.month,
        day=filters.day,
        page=filters.page,
        state=filters.state,
        paper=filters.paper,
        city=filters.city,
    )


@dataclass
class _Filters:
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    page: Optional[int] = None
    state: Optional[str] = None
    paper: Optional[str] = None
    city: Optional[str] = None


def _excerpt_regex(positives: Sequence[str]) -> str:
    if not positives:
        return ".{0,240}"
    body = "|".join(_re_escape(item) for item in positives[:12])
    return f".{{0,180}}({body}).{{0,240}}"


def _re_escape(value: str) -> str:
    out = []
    for ch in value:
        if ch in r"\.^$|?*+()[]{}":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _lower(
    node: _Node,
    filters: _Filters,
    *,
    under_or: bool,
    under_not: bool,
) -> Tuple[str, List[object], str, List[str]]:
    if isinstance(node, _Field):
        if under_or or under_not:
            raise ExploreQueryError(
                "Put year, month, page, paper, state, and city next to AND, not inside OR or NOT."
            )
        _apply_field(filters, node)
        return "", [], "", []

    if isinstance(node, _Not):
        sql, params, explanation, _positives = _lower(
            node.child, filters, under_or=under_or, under_not=True
        )
        if not sql:
            raise ExploreQueryError("NOT needs a word, phrase, or /regex/.")
        return f"NOT ({sql})", params, f"NOT {explanation}", []

    if isinstance(node, _And):
        bits_sql: List[str] = []
        bits_params: List[object] = []
        bits_explain: List[str] = []
        positives: List[str] = []
        for child in node.children:
            sql, params, explanation, found = _lower(
                child, filters, under_or=under_or, under_not=under_not
            )
            if sql:
                bits_sql.append(sql)
                bits_params.extend(params)
                bits_explain.append(explanation)
            if not under_not:
                positives.extend(found)
        if not bits_sql:
            return "", [], "", positives
        if len(bits_sql) == 1:
            return bits_sql[0], bits_params, bits_explain[0], positives
        joined = " AND ".join(f"({part})" if part.startswith("NOT ") or " OR " in part else part for part in bits_sql)
        return joined, bits_params, " AND ".join(bits_explain), positives

    if isinstance(node, _Or):
        bits_sql = []
        bits_params = []
        bits_explain = []
        positives = []
        for child in node.children:
            sql, params, explanation, found = _lower(
                child, filters, under_or=True, under_not=under_not
            )
            if not sql:
                raise ExploreQueryError("OR cannot combine a year or paper filter with text. Put filters outside the OR.")
            bits_sql.append(sql)
            bits_params.extend(params)
            bits_explain.append(explanation)
            if not under_not:
                positives.extend(found)
        if len(bits_sql) == 1:
            return bits_sql[0], bits_params, bits_explain[0], positives
        return "(" + " OR ".join(bits_sql) + ")", bits_params, "(" + " OR ".join(bits_explain) + ")", positives

    if isinstance(node, _Phrase):
        pattern = _phrase_pattern(node.value)
        return "regexp_matches(text, ?, 'i')", [pattern], f'phrase "{node.value}"', [node.value]

    if isinstance(node, _Term):
        shown = node.value + ("*" if node.prefix else "")
        return _like(node.value), [_like_param(node.value)], f'"{shown}"', [node.value]

    if isinstance(node, _Regex):
        return "regexp_matches(text, ?, 'i')", [node.value], f"/{node.value}/", [node.value]

    raise ExploreQueryError("Could not compile that query.")


def _phrase_pattern(value: str) -> str:
    """Match a phrase even when OCR put a line break between the words."""
    words = [part for part in value.split() if part]
    return r"\s+".join(_re_escape(part) for part in words)


def _like(_value: str) -> str:
    return "text ILIKE ? ESCAPE '\\'"


def _like_param(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _apply_field(filters: _Filters, node: _Field) -> None:
    name = node.name
    if name == "year":
        if node.start is None or node.end is None:
            raise ExploreQueryError("year needs a number or [start TO end].")
        filters.year_from = node.start if filters.year_from is None else max(filters.year_from, node.start)
        filters.year_to = node.end if filters.year_to is None else min(filters.year_to, node.end)
        if filters.year_from > filters.year_to:
            raise ExploreQueryError("Those year filters do not overlap.")
        return
    if name in {"month", "day", "page", "sequence"}:
        if node.start is None:
            raise ExploreQueryError(f"{name} needs a number.")
        if name == "month":
            filters.month = node.start
        elif name == "day":
            filters.day = node.start
        else:
            filters.page = node.start
        return
    if node.value is None:
        raise ExploreQueryError(f"{name} needs a value.")
    slug = _normalize_slug(node.value) if name in {"state", "paper", "newspaper"} else node.value.strip()
    if name == "state":
        filters.state = slug
    elif name in {"paper", "newspaper"}:
        filters.paper = slug
    elif name == "city":
        filters.city = node.value.strip()


def normalize_slug(value: str) -> str:
    """Fold a state or paper name into the dataset folder slug."""
    return _normalize_slug(value)


def _normalize_slug(value: str) -> str:
    slug = value.strip().lower().replace("-", "_").replace(" ", "_")
    allowed = "".join(ch for ch in slug if ch.isalnum() or ch == "_")
    if not allowed:
        raise ExploreQueryError(f"'{value}' is not a paper or state name.")
    return allowed


def _tokenize(source: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            phrase, i = _take_quoted(source, i)
            tokens.append(_Tok("PHRASE", phrase))
            continue
        if ch == "/" and _regex_can_start(source, i):
            pattern, nxt = _take_regex(source, i)
            tokens.append(_Tok("REGEX", pattern))
            i = nxt
            continue
        if ch == "(":
            tokens.append(_Tok("LP"))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Tok("RP"))
            i += 1
            continue
        if ch in "+-" and i + 1 < n and not source[i + 1].isspace():
            tokens.append(_Tok("PLUS" if ch == "+" else "MINUS"))
            i += 1
            continue
        word, nxt = _take_word_or_field(source, i)
        tokens.append(word)
        i = nxt
    return tokens


def _regex_can_start(source: str, i: int) -> bool:
    if i == 0:
        return True
    prev = source[i - 1]
    return prev.isspace() or prev in "(+-"


def _take_quoted(source: str, i: int) -> Tuple[str, int]:
    j = i + 1
    buf: List[str] = []
    while j < len(source) and source[j] != '"':
        buf.append(source[j])
        j += 1
    if j >= len(source):
        raise ExploreQueryError("Missing the closing quote.")
    phrase = "".join(buf).strip()
    if not phrase:
        raise ExploreQueryError("Empty quotes.")
    return phrase, j + 1


def _take_regex(source: str, i: int) -> Tuple[str, int]:
    j = i + 1
    buf: List[str] = []
    while j < len(source):
        if source[j] == "\\" and j + 1 < len(source):
            buf.append(source[j + 1])
            j += 2
            continue
        if source[j] == "/":
            pattern = "".join(buf)
            if not pattern:
                raise ExploreQueryError("Empty /regex/.")
            return pattern, j + 1
        buf.append(source[j])
        j += 1
    raise ExploreQueryError("Missing the closing / on a regular expression.")


def _take_word_or_field(source: str, i: int) -> Tuple[_Tok, int]:
    j = i
    while j < len(source) and (source[j].isalnum() or source[j] in "_"):
        j += 1
    word = source[i:j]
    if not word:
        raise ExploreQueryError(
            f"Unsupported character '{source[i]}'. Fuzzy (~), boost (^), and ? are not available."
        )
    if j < len(source) and source[j] == ":" and word.lower() in FIELDS:
        return _take_field(word.lower(), source, j + 1)
    if j < len(source) and source[j] == "*":
        if j + 1 < len(source) and not source[j + 1].isspace() and source[j + 1] not in "()":
            raise ExploreQueryError("A * wildcard is only supported at the end of a word.")
        return _Tok("PREFIX", word), j + 1
    if any(ch in word for ch in "~^?"):
        raise ExploreQueryError("Fuzzy (~), boost (^), and ? wildcards are not available. Use a prefix * or /regex/.")
    upper = word.upper()
    if upper in _OPS:
        return _Tok(upper), j
    return _Tok("WORD", word), j


def _take_field(name: str, source: str, i: int) -> Tuple[_Tok, int]:
    if i >= len(source):
        raise ExploreQueryError(f"{name}: needs a value.")
    if source[i] == '"':
        value, nxt = _take_quoted(source, i)
        return _Tok("FIELD", f"{name}\t{value}"), nxt
    if source[i] == "[":
        start, end, nxt = _take_range(source, i)
        return _Tok("RANGE", f"{name}\t{start}\t{end}"), nxt
    j = i
    while j < len(source) and not source[j].isspace() and source[j] not in "()":
        j += 1
    value = source[i:j].strip()
    if not value:
        raise ExploreQueryError(f"{name}: needs a value.")
    if any(ch in value for ch in "~^"):
        raise ExploreQueryError("Fuzzy (~) and boost (^) are not available on field values.")
    return _Tok("FIELD", f"{name}\t{value}"), j


def _take_range(source: str, i: int) -> Tuple[int, int, int]:
    # [1890 TO 1895]
    body_end = source.find("]", i)
    if body_end < 0:
        raise ExploreQueryError("Missing ] on a year range.")
    body = source[i + 1 : body_end].strip()
    parts = body.split()
    if len(parts) != 3 or parts[1].upper() != "TO":
        raise ExploreQueryError("Write ranges as [1890 TO 1895].")
    return _number(parts[0], "range"), _number(parts[2], "range"), body_end + 1


def _number(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ExploreQueryError(f"{label} needs a whole number, not '{value}'.") from exc


def _parse_or(tokens: Sequence[_Tok], pos: int) -> Tuple[_Node, int]:
    node, pos = _parse_and(tokens, pos)
    children = [node]
    while pos < len(tokens) and tokens[pos].kind == "OR":
        right, pos = _parse_and(tokens, pos + 1)
        children.append(right)
    if len(children) == 1:
        return node, pos
    return _Or(children), pos


def _parse_and(tokens: Sequence[_Tok], pos: int) -> Tuple[_Node, int]:
    node, pos = _parse_not(tokens, pos)
    children = [node]
    while pos < len(tokens) and tokens[pos].kind not in {"OR", "RP"}:
        if tokens[pos].kind == "AND":
            pos += 1
        nxt, pos = _parse_not(tokens, pos)
        children.append(nxt)
    if len(children) == 1:
        return node, pos
    return _And(children), pos


def _parse_not(tokens: Sequence[_Tok], pos: int) -> Tuple[_Node, int]:
    if pos >= len(tokens):
        raise ExploreQueryError("The query ended where a word was expected.")
    tok = tokens[pos]
    if tok.kind in {"NOT", "MINUS"}:
        child, pos = _parse_not(tokens, pos + 1)
        return _Not(child), pos
    if tok.kind == "PLUS":
        return _parse_not(tokens, pos + 1)
    return _parse_atom(tokens, pos)


def _parse_atom(tokens: Sequence[_Tok], pos: int) -> Tuple[_Node, int]:
    if pos >= len(tokens):
        raise ExploreQueryError("The query ended where a word was expected.")
    tok = tokens[pos]
    if tok.kind == "LP":
        node, pos = _parse_or(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos].kind != "RP":
            raise ExploreQueryError("Missing ).")
        return node, pos + 1
    if tok.kind == "PHRASE":
        return _Phrase(tok.value), pos + 1
    if tok.kind == "REGEX":
        return _Regex(tok.value), pos + 1
    if tok.kind == "WORD":
        return _Term(tok.value), pos + 1
    if tok.kind == "PREFIX":
        return _Term(tok.value, prefix=True), pos + 1
    if tok.kind == "FIELD":
        name, value = tok.value.split("\t", 1)
        if name == "year":
            number = _number(value, "year")
            return _Field(name, start=number, end=number), pos + 1
        if name in {"month", "day", "page", "sequence"}:
            return _Field(name, start=_number(value, name)), pos + 1
        return _Field(name, value=value), pos + 1
    if tok.kind == "RANGE":
        name, start, end = tok.value.split("\t")
        if name != "year":
            raise ExploreQueryError("Ranges are only supported for year, written as year:[1890 TO 1895].")
        lo, hi = int(start), int(end)
        if lo > hi:
            raise ExploreQueryError("Range start is after the end.")
        return _Field(name, start=lo, end=hi), pos + 1
    raise ExploreQueryError(f"Unexpected '{tok.value or tok.kind}'.")
