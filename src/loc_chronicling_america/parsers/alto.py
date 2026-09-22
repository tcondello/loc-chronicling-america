"""High-performance parser for ALTO (Analyzed Layout and Text Object) XML files.

Supports ALTO v2, v3, and v4 schemas as used across National Digital Newspaper
Program (NDNP) scans from the Library of Congress.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AltoWord:
    """Represents an individual recognized word with coordinates and OCR confidence."""

    content: str
    hpos: Optional[int] = None
    vpos: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    confidence: Optional[float] = None  # WC (Word Confidence) attribute, 0.0 to 1.0
    word_id: Optional[str] = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Returns (hpos, vpos, width, height)."""
        return (self.hpos or 0, self.vpos or 0, self.width or 0, self.height or 0)


@dataclass
class AltoLine:
    """Represents a recognized line of text in an ALTO block."""

    line_id: Optional[str]
    words: List[AltoWord] = field(default_factory=list)
    hpos: Optional[int] = None
    vpos: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def text(self) -> str:
        """Joined words for this line."""
        return " ".join(w.content for w in self.words)


@dataclass
class AltoBlock:
    """Represents a text block (e.g. column, paragraph, or article section)."""

    block_id: Optional[str]
    lines: List[AltoLine] = field(default_factory=list)
    language: Optional[str] = None
    hpos: Optional[int] = None
    vpos: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def text(self) -> str:
        """Full text of this block with line breaks."""
        return "\n".join(l.text for l in self.lines if l.text)


@dataclass
class AltoDocument:
    """Parsed ALTO XML document representing a scanned newspaper page."""

    page_id: Optional[str] = None
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    measurement_unit: Optional[str] = None
    ocr_engine: Optional[str] = None
    blocks: List[AltoBlock] = field(default_factory=list)

    @property
    def words(self) -> List[AltoWord]:
        """Flattened list of all recognized words on the page."""
        all_words = []
        for block in self.blocks:
            for line in block.lines:
                all_words.extend(line.words)
        return all_words

    @property
    def lines(self) -> List[AltoLine]:
        """Flattened list of all lines on the page."""
        all_lines = []
        for block in self.blocks:
            all_lines.extend(block.lines)
        return all_lines

    def extract_full_text(self) -> str:
        """Extract full plain text in logical reading order."""
        block_texts = [b.text for b in self.blocks if b.text]
        return "\n\n".join(block_texts)

    def to_layout_dict(self) -> dict:
        """Extract normalized layout bounding boxes (0-1000 scale) for Document AI / Parquet.

        Returns a dictionary containing:
            - page_width, page_height, page_unit, ocr_engine
            - words: list of word tokens
            - boxes: list of [x0, y0, x1, y1] normalized to 0-1000
            - word_confidences: list of float confidences (0.0 - 1.0)
            - lines: list of {"box": [x0, y0, x1, y1], "text": str}
            - blocks: list of {"block_id": str, "box": [x0, y0, x1, y1], "text": str}
        """
        pw = max(self.page_width or 1000, 1)
        ph = max(self.page_height or 1000, 1)

        def norm_box(x: Optional[int], y: Optional[int], w: Optional[int], h: Optional[int]) -> List[int]:
            vx = max(x or 0, 0)
            vy = max(y or 0, 0)
            vw = max(w or 0, 0)
            vh = max(h or 0, 0)
            x0 = min(max(int(round((vx / pw) * 1000)), 0), 1000)
            y0 = min(max(int(round((vy / ph) * 1000)), 0), 1000)
            x1 = min(max(int(round(((vx + vw) / pw) * 1000)), 0), 1000)
            y1 = min(max(int(round(((vy + vh) / ph) * 1000)), 0), 1000)
            return [x0, y0, x1, y1]

        words_list: List[str] = []
        boxes_list: List[List[int]] = []
        conf_list: List[float] = []
        lines_list: List[dict] = []
        blocks_list: List[dict] = []

        for block in self.blocks:
            block_lines_text = []
            for line in block.lines:
                line_words_text = []
                for word in line.words:
                    if not word.content:
                        continue
                    words_list.append(word.content)
                    boxes_list.append(norm_box(word.hpos, word.vpos, word.width, word.height))
                    conf_list.append(float(word.confidence if word.confidence is not None else 1.0))
                    line_words_text.append(word.content)

                if line_words_text:
                    l_text = " ".join(line_words_text)
                    block_lines_text.append(l_text)
                    lines_list.append({
                        "box": norm_box(line.hpos, line.vpos, line.width, line.height),
                        "text": l_text,
                    })

            if block_lines_text:
                blocks_list.append({
                    "block_id": block.block_id or "",
                    "box": norm_box(block.hpos, block.vpos, block.width, block.height),
                    "text": "\n".join(block_lines_text),
                })

        return {
            "page_width": self.page_width or 0,
            "page_height": self.page_height or 0,
            "page_unit": self.measurement_unit or "inch1200",
            "ocr_engine": self.ocr_engine or "Unknown",
            "words": words_list,
            "boxes": boxes_list,
            "word_confidences": conf_list,
            "lines": lines_list,
            "blocks": blocks_list,
        }


def _safe_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_alto_xml(xml_content: str | bytes) -> AltoDocument:
    """Parse ALTO XML string or bytes into an AltoDocument.

    Handles namespace stripping and robust element lookup across schema variations.
    """
    if isinstance(xml_content, str):
        # Handle encoding declaration if present in string
        xml_bytes = xml_content.encode("utf-8")
    else:
        xml_bytes = xml_content

    root = ET.fromstring(xml_bytes)

    # Helper to strip namespace
    def local_tag(elem: ET.Element) -> str:
        return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    doc = AltoDocument()

    # Measurement Unit
    for elem in root.iter():
        if local_tag(elem) == "MeasurementUnit" and elem.text:
            doc.measurement_unit = elem.text.strip()
            break

    # OCR Software / Engine
    software_parts = []
    for elem in root.iter():
        lt = local_tag(elem)
        if lt == "softwareName" and elem.text and elem.text.strip():
            software_parts.append(elem.text.strip())
        elif lt == "softwareVersion" and elem.text and elem.text.strip() and software_parts:
            software_parts[-1] += f" {elem.text.strip()}"
    if software_parts:
        doc.ocr_engine = " / ".join(software_parts[:2])

    # Find Page
    for elem in root.iter():
        if local_tag(elem) == "Page":
            doc.page_id = elem.attrib.get("ID")
            doc.page_width = _safe_int(elem.attrib.get("WIDTH"))
            doc.page_height = _safe_int(elem.attrib.get("HEIGHT"))
            break

    # Find TextBlocks
    for block_elem in root.iter():
        if local_tag(block_elem) == "TextBlock":
            block = AltoBlock(
                block_id=block_elem.attrib.get("ID"),
                language=block_elem.attrib.get("language"),
                hpos=_safe_int(block_elem.attrib.get("HPOS")),
                vpos=_safe_int(block_elem.attrib.get("VPOS")),
                width=_safe_int(block_elem.attrib.get("WIDTH")),
                height=_safe_int(block_elem.attrib.get("HEIGHT")),
            )

            # Find TextLines within TextBlock
            for line_elem in block_elem.iter():
                if local_tag(line_elem) == "TextLine":
                    line = AltoLine(
                        line_id=line_elem.attrib.get("ID"),
                        hpos=_safe_int(line_elem.attrib.get("HPOS")),
                        vpos=_safe_int(line_elem.attrib.get("VPOS")),
                        width=_safe_int(line_elem.attrib.get("WIDTH")),
                        height=_safe_int(line_elem.attrib.get("HEIGHT")),
                    )

                    for word_elem in line_elem.iter():
                        if local_tag(word_elem) == "String":
                            content = word_elem.attrib.get("CONTENT", "")
                            if content:
                                word = AltoWord(
                                    content=content,
                                    word_id=word_elem.attrib.get("ID"),
                                    hpos=_safe_int(word_elem.attrib.get("HPOS")),
                                    vpos=_safe_int(word_elem.attrib.get("VPOS")),
                                    width=_safe_int(word_elem.attrib.get("WIDTH")),
                                    height=_safe_int(word_elem.attrib.get("HEIGHT")),
                                    confidence=_safe_float(word_elem.attrib.get("WC")),
                                )
                                line.words.append(word)

                    if line.words:
                        block.lines.append(line)

            if block.lines:
                doc.blocks.append(block)

    return doc
