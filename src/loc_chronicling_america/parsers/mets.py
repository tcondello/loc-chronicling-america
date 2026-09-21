"""Parser for METS (Metadata Encoding and Transmission Standard) XML files and NDNP batch XMLs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetsPage:
    """Page files referenced in a METS issue file."""

    sequence: int
    page_id: Optional[str] = None
    alto_path: Optional[str] = None
    pdf_path: Optional[str] = None
    jp2_path: Optional[str] = None


@dataclass
class MetsIssue:
    """Issue metadata parsed from METS XML."""

    lccn: Optional[str] = None
    title: Optional[str] = None
    date_issued: Optional[str] = None
    volume: Optional[str] = None
    issue_number: Optional[str] = None
    edition_number: Optional[str] = "1"
    pages: List[MetsPage] = field(default_factory=list)


@dataclass
class BatchIssueRef:
    """Reference to an issue inside a batch.xml or batch_1.xml manifest."""

    lccn: str
    issue_date: str
    edition_order: int
    issue_xml_path: str


def _local_tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def parse_mets_issue(xml_content: str | bytes) -> MetsIssue:
    """Parse a newspaper issue METS XML document into a MetsIssue."""
    if isinstance(xml_content, str):
        xml_bytes = xml_content.encode("utf-8")
    else:
        xml_bytes = xml_content

    root = ET.fromstring(xml_bytes)
    issue = MetsIssue()

    # Extract MODS metadata
    for elem in root.iter():
        tag = _local_tag(elem)
        if tag == "identifier" and elem.attrib.get("type") == "lccn" and elem.text:
            issue.lccn = elem.text.strip()
        elif tag == "dateIssued" and elem.text:
            issue.date_issued = elem.text.strip()
        elif tag == "detail" and elem.attrib.get("type") == "volume":
            for child in elem:
                if _local_tag(child) == "number" and child.text:
                    issue.volume = child.text.strip()
        elif tag == "detail" and elem.attrib.get("type") == "issue":
            for child in elem:
                if _local_tag(child) == "number" and child.text:
                    issue.issue_number = child.text.strip()
        elif tag == "detail" and elem.attrib.get("type") == "edition":
            for child in elem:
                if _local_tag(child) == "number" and child.text:
                    issue.edition_number = child.text.strip()

    # Extract label for title if available
    if "LABEL" in root.attrib:
        issue.title = root.attrib["LABEL"]

    # Extract files from fileSec
    file_map: Dict[str, str] = {}  # file_id -> href
    for file_elem in root.iter():
        if _local_tag(file_elem) == "file":
            fid = file_elem.attrib.get("ID")
            for flocat in file_elem:
                if _local_tag(flocat) == "FLocat":
                    href = flocat.attrib.get("{http://www.w3.org/1999/xlink}href") or flocat.attrib.get("href")
                    if fid and href:
                        file_map[fid] = href

    # Extract structural map (structMap) to link page numbers to files
    seq = 1
    for div in root.iter():
        if _local_tag(div) == "div" and div.attrib.get("TYPE") == "page":
            page = MetsPage(sequence=seq, page_id=div.attrib.get("ID"))
            for fptr in div:
                if _local_tag(fptr) == "fptr":
                    file_id = fptr.attrib.get("FILEID")
                    if file_id and file_id in file_map:
                        href = file_map[file_id]
                        if href.endswith(".xml"):
                            page.alto_path = href
                        elif href.endswith(".pdf"):
                            page.pdf_path = href
                        elif href.endswith(".jp2"):
                            page.jp2_path = href

            if page.alto_path or page.pdf_path or page.jp2_path:
                issue.pages.append(page)
                seq += 1

    return issue


def parse_batch_manifest(xml_content: str | bytes) -> List[BatchIssueRef]:
    """Parse an NDNP batch.xml or batch_1.xml file listing issues in a batch."""
    if isinstance(xml_content, str):
        xml_bytes = xml_content.encode("utf-8")
    else:
        xml_bytes = xml_content

    root = ET.fromstring(xml_bytes)
    issues: List[BatchIssueRef] = []

    for elem in root.iter():
        if _local_tag(elem) == "issue":
            lccn = elem.attrib.get("lccn", "")
            issue_date = elem.attrib.get("issueDate", "")
            edition_order = int(elem.attrib.get("editionOrder", "1"))
            path = elem.text.strip() if elem.text else ""
            if lccn and issue_date and path:
                issues.append(
                    BatchIssueRef(
                        lccn=lccn,
                        issue_date=issue_date,
                        edition_order=edition_order,
                        issue_xml_path=path,
                    )
                )

    return issues
