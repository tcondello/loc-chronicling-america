"""Tests for METS XML and batch manifest parsers."""

from loc_chronicling_america.parsers.mets import parse_batch_manifest, parse_mets_issue


def test_parse_mets_issue(mets_xml: str):
    issue = parse_mets_issue(mets_xml)

    assert issue.lccn == "00225879"
    assert issue.date_issued == "1915-07-03"
    assert issue.volume == "1"
    assert issue.issue_number == "1"
    assert issue.edition_number == "1"
    assert "The monitor" in (issue.title or "")

    assert len(issue.pages) == 1
    page1 = issue.pages[0]
    assert page1.sequence == 1
    assert page1.alto_path == "0004.xml"
    assert page1.pdf_path == "0004.pdf"


def test_parse_batch_manifest(batch_xml: str):
    issues = parse_batch_manifest(batch_xml)

    assert len(issues) == 2
    assert issues[0].lccn == "00225879"
    assert issues[0].issue_date == "1915-07-03"
    assert issues[0].edition_order == 1
    assert "1915070301_1.xml" in issues[0].issue_xml_path

    assert issues[1].issue_date == "1915-07-10"
