"""Tests for ALTO XML parser."""

from loc_chronicling_america.parsers.alto import parse_alto_xml


def test_parse_alto(alto_xml: str):
    doc = parse_alto_xml(alto_xml)

    assert doc.page_id == "PAGE1"
    assert doc.page_width == 15000
    assert doc.page_height == 20000
    assert doc.measurement_unit == "inch1200"

    # Blocks
    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.block_id == "BLOCK1"
    assert block.language == "eng"
    assert len(block.lines) == 2

    # Words
    words = doc.words
    assert len(words) == 4
    assert [w.content for w in words] == ["THE", "MONITOR", "OMAHA,", "NEBRASKA"]

    # Word coordinates and confidence
    w2 = words[1]
    assert w2.content == "MONITOR"
    assert w2.box == (1000, 100, 2500, 300)
    assert w2.confidence == 0.99

    # Text extraction
    full_text = doc.extract_full_text()
    assert "THE MONITOR" in full_text
    assert "OMAHA, NEBRASKA" in full_text
