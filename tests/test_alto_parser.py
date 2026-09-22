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


def test_alto_layout_dict(alto_xml: str):
    doc = parse_alto_xml(alto_xml)
    layout = doc.to_layout_dict()

    assert layout["page_width"] == 15000
    assert layout["page_height"] == 20000
    assert layout["page_unit"] == "inch1200"

    # Words and normalized boxes
    assert layout["words"] == ["THE", "MONITOR", "OMAHA,", "NEBRASKA"]
    assert len(layout["boxes"]) == 4
    assert len(layout["word_confidences"]) == 4

    # Check that coordinates are normalized [0..1000]
    # S1: HPOS=100, VPOS=100, WIDTH=800, HEIGHT=300, PW=15000, PH=20000
    # x0 = round(100/15000*1000) = 7, y0 = round(100/20000*1000) = 5
    # x1 = round(900/15000*1000) = 60, y1 = round(400/20000*1000) = 20
    assert layout["boxes"][0] == [7, 5, 60, 20]
    for box in layout["boxes"]:
        assert len(box) == 4
        assert all(0 <= v <= 1000 for v in box)
        assert box[0] <= box[2]
        assert box[1] <= box[3]

    # Lines
    assert len(layout["lines"]) == 2
    assert layout["lines"][0]["text"] == "THE MONITOR"
    assert len(layout["lines"][0]["box"]) == 4

    # Blocks
    assert len(layout["blocks"]) == 1
    assert layout["blocks"][0]["block_id"] == "BLOCK1"
    assert "THE MONITOR" in layout["blocks"][0]["text"]
    assert len(layout["blocks"][0]["box"]) == 4


def test_parse_alto_to_layout_dict(alto_xml: str):
    from loc_chronicling_america.parsers.alto import parse_alto_to_layout_dict

    layout, full_text = parse_alto_to_layout_dict(alto_xml.encode("utf-8"))
    assert layout is not None
    assert full_text is not None
    assert "THE MONITOR" in full_text
    assert layout["words"] == ["THE", "MONITOR", "OMAHA,", "NEBRASKA"]
    assert layout["page_width"] == 15000
    assert layout["page_height"] == 20000


