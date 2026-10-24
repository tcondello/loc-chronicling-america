"""Tests for the Lucene-like OCR search dialect."""

import pytest

from loc_chronicling_america.explore_query import ExploreQueryError, compile_query


def test_space_means_and_and_quotes_are_a_phrase():
    words = compile_query("buried alive")
    phrase = compile_query('"buried alive"')
    assert words.where_sql.count("ILIKE") == 2
    assert " AND " in words.explanation
    assert phrase.where_sql.startswith("regexp_matches")
    assert phrase.explanation == 'phrase "buried alive"'
    assert phrase.params == [r"buried\s+alive"]


def test_or_not_and_grouping():
    compiled = compile_query("(haunted OR ghost) house -opera")
    assert compiled.where_sql.count("ILIKE") == 4
    assert " OR " in compiled.where_sql
    assert "NOT" in compiled.where_sql
    assert compiled.explanation.startswith("(")
    assert 'NOT "opera"' in compiled.explanation


def test_year_range_and_month_stay_out_of_the_text_predicate():
    compiled = compile_query('year:[1890 TO 1895] month:10 "haunted house"')
    assert compiled.year_from == 1890
    assert compiled.year_to == 1895
    assert compiled.month == 10
    assert "regexp_matches" in compiled.where_sql
    assert "1890" not in compiled.where_sql


def test_year_inside_or_is_rejected():
    with pytest.raises(ExploreQueryError, match="OR"):
        compile_query("year:1890 OR ghost")


def test_lucene_fuzzy_boost_and_question_mark_are_rejected():
    with pytest.raises(ExploreQueryError):
        compile_query("ghost~")
    with pytest.raises(ExploreQueryError):
        compile_query("ghost^2")
    with pytest.raises(ExploreQueryError):
        compile_query("gho?t")


def test_regex_and_prefix_are_accepted():
    regex = compile_query(r"/buried aliv[ea]/")
    assert regex.where_sql.startswith("regexp_matches")
    assert regex.params == ["buried aliv[ea]"]
    prefix = compile_query("haunt*")
    assert prefix.params == ["%haunt%"]


def test_quotes_in_a_phrase_stay_in_parameters():
    compiled = compile_query("\"o'brien\"")
    assert compiled.params == ["o'brien"]
    assert "o'brien" not in compiled.where_sql


def test_local_rows_match_the_compiled_predicate():
    duckdb = pytest.importorskip("duckdb")
    compiled = compile_query('"buried alive" -opera')
    con = duckdb.connect()
    con.execute("CREATE TABLE pages (text VARCHAR)")
    con.execute(
        "INSERT INTO pages VALUES (?), (?), (?)",
        [
            "He was buried alive last Saturday night",
            "The opera tonight is The Flying Dutchman",
            "She was buried alive during the opera",
        ],
    )
    rows = con.execute(
        f"SELECT text FROM pages WHERE {compiled.where_sql}",
        compiled.params,
    ).fetchall()
    assert rows == [("He was buried alive last Saturday night",)]
