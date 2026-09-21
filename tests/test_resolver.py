"""Tests for RecordResolver."""

import pytest
from loc_chronicling_america.resolver import RecordResolver


def test_parse_url_resource():
    resolver = RecordResolver()
    url = "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery"
    lccn, date, ed = resolver.parse_url(url)
    assert lccn == "00225879"
    assert date == "1915-07-03"
    assert ed == 1


def test_parse_url_item():
    resolver = RecordResolver()
    url = "https://www.loc.gov/item/sn85026945/1847-03-03/ed-2/"
    lccn, date, ed = resolver.parse_url(url)
    assert lccn == "sn85026945"
    assert date == "1847-03-03"
    assert ed == 2


def test_parse_url_compact_sequence():
    resolver = RecordResolver()
    url = "https://www.loc.gov/resource/sn84020657/1917012601/"
    lccn, date, ed = resolver.parse_url(url)
    assert lccn == "sn84020657"
    assert date == "1917-01-26"
    assert ed == 1


def test_parse_url_invalid():
    resolver = RecordResolver()
    with pytest.raises(ValueError):
        resolver.parse_url("https://www.loc.gov/about/")


def test_live_resolve_user_example():
    """Live integration test: resolve the user's specific Omaha Monitor issue."""
    resolver = RecordResolver()
    url = "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery"

    record = resolver.resolve(url)

    assert "monitor" in record.title.lower()
    assert record.lccn == "00225879"
    assert record.date == "1915-07-03"
    assert record.edition == 1
    assert record.batch_name == "nbu_indescribablebeast_ver01"
    assert record.page_count == 8
    assert "nbu_indescribablebeast_ver01.tar.bz2" in record.bulk_ocr_url
    assert "nbu_indescribablebeast_ver01" in record.raw_batch_url

    # Check page 1 assets
    page1 = record.get_page(1)
    assert page1.sequence == 1
    assert page1.pdf_url is not None
    assert page1.jp2_url is not None
    assert page1.alto_xml_url is not None
    assert "0004.pdf" in page1.pdf_url
    assert "0004.jp2" in page1.jp2_url
    assert "0004.xml" in page1.alto_xml_url
