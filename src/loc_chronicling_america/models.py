"""Data models for Library of Congress Chronicling America records, batches, and pages."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TitleInfo(BaseModel):
    """Metadata for a digitized newspaper title."""

    lccn: str = Field(description="Library of Congress Control Number")
    name: str = Field(description="Newspaper title")
    state: Optional[str] = Field(default=None, description="State of publication")
    city: Optional[str] = Field(default=None, description="City of publication")
    county: Optional[str] = Field(default=None, description="County of publication")
    start_year: Optional[int] = Field(default=None, description="First year of publication")
    end_year: Optional[int] = Field(default=None, description="Last year of publication")
    issue_count: Optional[int] = Field(default=None, description="Total digitized issues count")
    first_issue_date: Optional[str] = Field(default=None, description="Date of first issue")
    last_issue_date: Optional[str] = Field(default=None, description="Date of last issue")
    ethnicity: Optional[str] = Field(default=None, description="Target demographic/ethnicity (if any)")
    language: Optional[str] = Field(default="English", description="Primary language")
    loc_url: Optional[str] = Field(default=None, description="LoC permalink")

    @property
    def formatted_years(self) -> str:
        s = str(self.start_year) if self.start_year else "?"
        e = str(self.end_year) if self.end_year else "?"
        return f"{s}-{e}"


class BatchInfo(BaseModel):
    """Metadata for a National Digital Newspaper Program (NDNP) batch."""

    name: str = Field(description="Batch identifier, e.g. 'nbu_indescribablebeast_ver01'")
    awardee: Optional[str] = Field(default=None, description="Awardee code, e.g. 'nbu' (Univ of Nebraska)")
    state: Optional[str] = Field(default=None, description="State of awardee institution")
    archive_url: str = Field(description="Direct download URL to .tar.bz2 bulk OCR archive")
    raw_batch_url: str = Field(description="Directory URL to raw BagIt batch files")
    size_bytes: Optional[int] = Field(default=None, description="Bulk archive size in bytes")
    page_count: Optional[int] = Field(default=None, description="Number of newspaper pages in batch")
    issue_count: Optional[int] = Field(default=None, description="Number of newspaper issues in batch")
    lccns: List[str] = Field(default_factory=list, description="LCCNs of newspapers included in this batch")
    sha256: Optional[str] = Field(default=None, description="SHA-256 checksum of bulk archive")
    ingested_date: Optional[str] = Field(default=None, description="Date ingested by LoC")

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        if self.size_bytes:
            return round(self.size_bytes / (1024 * 1024), 2)
        return 0.0

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        if self.size_bytes:
            return round(self.size_bytes / (1024 * 1024 * 1024), 2)
        return 0.0


class PageRecord(BaseModel):
    """Represents a single digitized newspaper page and its various asset formats."""

    sequence: int = Field(description="Page sequence number (1-based index within the issue)")
    title: str = Field(description="Page title description, e.g. 'Image 1 of The monitor...'")
    lccn: str = Field(description="Newspaper LCCN")
    date: str = Field(description="Publication date (YYYY-MM-DD)")
    edition: int = Field(default=1, description="Edition number")
    reel_number: Optional[str] = Field(default=None, description="Microfilm reel number")
    width: Optional[int] = Field(default=None, description="Pixel width of scan")
    height: Optional[int] = Field(default=None, description="Pixel height of scan")

    # Direct Asset URLs
    pdf_url: Optional[str] = Field(default=None, description="Single-page PDF download URL")
    jp2_url: Optional[str] = Field(default=None, description="High-resolution JPEG 2000 master image URL")
    alto_xml_url: Optional[str] = Field(default=None, description="ALTO XML OCR layout and text URL")
    text_service_url: Optional[str] = Field(default=None, description="LoC plain text OCR service URL")
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail image URL")
    iiif_info_url: Optional[str] = Field(default=None, description="IIIF Image API info.json URL")

    # Cached content
    _text_cache: Optional[str] = None
    _alto_cache: Optional[str] = None

    def get_text(self, client: Optional[Any] = None) -> str:
        """Fetch and return the OCR plain text for this page.

        If client is not passed, a default Downloader will be used.
        """
        if self._text_cache is not None:
            return self._text_cache

        # First preference: direct text service
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()

        if self.text_service_url:
            try:
                raw_text = downloader.fetch_text(self.text_service_url)
                if raw_text and raw_text.strip():
                    stripped = raw_text.strip()
                    if stripped.startswith("{"):
                        import json
                        try:
                            parsed_json = json.loads(stripped)
                            for k, v in parsed_json.items():
                                if isinstance(v, dict) and "full_text" in v:
                                    self._text_cache = str(v["full_text"]).strip()
                                    return self._text_cache
                                elif k == "full_text":
                                    self._text_cache = str(v).strip()
                                    return self._text_cache
                        except Exception:
                            pass
                    self._text_cache = stripped
                    return self._text_cache
            except Exception:
                pass

        # Second preference: parse ALTO XML if available
        if self.alto_xml_url:
            from .parsers.alto import parse_alto_xml

            alto_xml = downloader.fetch_text(self.alto_xml_url)
            self._alto_cache = alto_xml
            doc = parse_alto_xml(alto_xml)
            self._text_cache = doc.extract_full_text()
            return self._text_cache

        return ""

    def get_alto(self, client: Optional[Any] = None) -> Any:
        """Fetch and parse the ALTO XML document containing word coordinates and confidence scores."""
        from .downloader import get_default_downloader
        from .parsers.alto import parse_alto_xml

        downloader = client or get_default_downloader()
        if self._alto_cache is None:
            if not self.alto_xml_url:
                raise ValueError(f"No ALTO XML URL available for page {self.sequence}")
            self._alto_cache = downloader.fetch_text(self.alto_xml_url)

        return parse_alto_xml(self._alto_cache)

    def iiif_image_url(self, pct: int = 100) -> Optional[str]:
        """Construct IIIF image URL at the requested percentage scaling (1-100)."""
        if self.iiif_info_url and self.iiif_info_url.endswith("/info.json"):
            base = self.iiif_info_url[:-len("/info.json")]
            return f"{base}/full/pct:{pct}/0/default.jpg"
        if self.thumbnail_url and "full/pct:" in self.thumbnail_url:
            import re
            return re.sub(r"pct:\d+(\.\d+)?", f"pct:{pct}", self.thumbnail_url)
        return None

    def download_image(
        self,
        dest_dir: Path | str,
        pct: int = 100,
        filename: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> Path:
        """Download high-resolution scan image (JPEG) via IIIF."""
        url = self.iiif_image_url(pct=pct)
        if not url:
            raise ValueError(f"No IIIF image available for page {self.sequence}")
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()
        name = filename or f"{self.lccn}_{self.date}_ed-{self.edition}_seq-{self.sequence}.jpg"
        dest = Path(dest_dir) / name
        return downloader.download_file(url, dest, description=f"Page {self.sequence} Image ({pct}%)")

    def download_pdf(
        self,
        dest_dir: Path | str,
        filename: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> Path:
        """Download single-page PDF.

        If direct storage PDF download is restricted by LoC, seamlessly fetches
        the 100% full-resolution scan and wraps it into a standard 1-page PDF.
        """
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()
        name = filename or f"{self.lccn}_{self.date}_ed-{self.edition}_seq-{self.sequence}.pdf"
        dest = Path(dest_dir) / name

        if self.pdf_url:
            try:
                return downloader.download_file(self.pdf_url, dest, description=f"Page {self.sequence} PDF")
            except Exception:
                pass

        # Fallback: create PDF from full-res IIIF JPEG scan
        img_url = self.iiif_image_url(pct=100)
        if not img_url:
            raise ValueError(f"No PDF or scan available for page {self.sequence}")

        temp_img = dest.with_suffix(".temp.jpg")
        downloader.download_file(img_url, temp_img, description=f"Page {self.sequence} PDF (from scan)")
        jpeg_bytes = temp_img.read_bytes()
        temp_img.unlink(missing_ok=True)

        w = self.width or 3600
        h = self.height or 5200
        pdf_bytes = _create_single_page_pdf(jpeg_bytes, w, h)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pdf_bytes)
        return dest

    def download_jp2(self, dest_dir: Path | str, filename: Optional[str] = None, client: Optional[Any] = None) -> Path:
        """Download master JP2 scan to destination directory."""
        if not self.jp2_url:
            raise ValueError(f"No JP2 image available for page {self.sequence}")
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()
        name = filename or f"{self.lccn}_{self.date}_ed-{self.edition}_seq-{self.sequence}.jp2"
        dest = Path(dest_dir) / name
        return downloader.download_file(self.jp2_url, dest, description=f"Page {self.sequence} JP2")

    def download_alto(self, dest_dir: Path | str, filename: Optional[str] = None, client: Optional[Any] = None) -> Path:
        """Download ALTO XML to destination directory."""
        if not self.alto_xml_url:
            raise ValueError(f"No ALTO XML available for page {self.sequence}")
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()
        name = filename or f"{self.lccn}_{self.date}_ed-{self.edition}_seq-{self.sequence}.xml"
        dest = Path(dest_dir) / name
        return downloader.download_file(self.alto_xml_url, dest, description=f"Page {self.sequence} ALTO XML")

    def download_text(self, dest_dir: Path | str, filename: Optional[str] = None, client: Optional[Any] = None) -> Path:
        """Extract and save plain text OCR to destination file."""
        text = self.get_text(client=client)
        name = filename or f"{self.lccn}_{self.date}_ed-{self.edition}_seq-{self.sequence}.txt"
        dest = Path(dest_dir) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest


def _create_single_page_pdf(jpeg_data: bytes, width: int, height: int) -> bytes:
    """Create a minimal, valid single-page PDF embedding raw JPEG data using /DCTDecode."""
    pt_w = 612
    pt_h = int(pt_w * (height / width)) if width > 0 else 792

    stream_len = len(jpeg_data)
    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    offsets = []

    # Obj 1: Catalog
    offsets.append(len(pdf))
    pdf.extend(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Obj 2: Pages
    offsets.append(len(pdf))
    pdf.extend(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

    # Obj 3: Page
    offsets.append(len(pdf))
    pdf.extend(f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pt_w} {pt_h}] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n".encode())

    # Obj 4: Content stream (draw image)
    content = f"q {pt_w} 0 0 {pt_h} 0 0 cm /Im1 Do Q".encode()
    offsets.append(len(pdf))
    pdf.extend(f"4 0 obj\n<< /Length {len(content)} >>\nstream\n".encode())
    pdf.extend(content)
    pdf.extend(b"\nendstream\nendobj\n")

    # Obj 5: Image XObject (embed raw JPEG directly using /DCTDecode)
    offsets.append(len(pdf))
    pdf.extend(f"5 0 obj\n<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {stream_len} >>\nstream\n".encode())
    pdf.extend(jpeg_data)
    pdf.extend(b"\nendstream\nendobj\n")

    # Cross-reference table
    xref_offset = len(pdf)
    pdf.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())

    pdf.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(pdf)


class IssueRecord(BaseModel):
    """Complete metadata and page assets for a newspaper issue."""

    title: str = Field(description="Issue title, e.g. 'The monitor (Omaha, Neb.), July 3, 1915'")
    newspaper_title: str = Field(description="Base newspaper title, e.g. 'The Monitor'")
    lccn: str = Field(description="Library of Congress Control Number")
    date: str = Field(description="Publication date (YYYY-MM-DD)")
    edition: int = Field(default=1, description="Edition number")
    place_of_publication: Optional[str] = Field(default=None, description="Place of publication")
    state: Optional[str] = Field(default=None, description="State")
    city: Optional[str] = Field(default=None, description="City")
    
    # Exact source batch information
    batch_name: str = Field(description="Exact batch identifier where this issue is stored, e.g. 'nbu_indescribablebeast_ver01'")
    bulk_ocr_url: str = Field(description="Direct URL to .tar.bz2 bulk OCR archive for this batch")
    raw_batch_url: str = Field(description="Direct URL to raw BagIt batch directory")
    
    # LoC item URL and IIIF Manifest
    loc_item_url: str = Field(description="Canonical LoC item URL")
    iiif_manifest_url: Optional[str] = Field(default=None, description="IIIF Presentation manifest URL")

    # Pages
    pages: List[PageRecord] = Field(default_factory=list, description="Pages in this issue")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def parsed_date(self) -> datetime.date:
        """Date object for publication date."""
        return datetime.date.fromisoformat(self.date)

    def get_page(self, sequence: int) -> PageRecord:
        """Get page by 1-based sequence number."""
        for p in self.pages:
            if p.sequence == sequence:
                return p
        raise IndexError(f"Page sequence {sequence} not found in issue (has {len(self.pages)} pages)")

    def get_batch(self, client: Optional[Any] = None) -> Any:
        """Get the full Batch object corresponding to this issue's source batch."""
        from .batch import Batch
        from .downloader import get_default_downloader

        downloader = client or get_default_downloader()
        return Batch(name=self.batch_name, downloader=downloader)

    def download_all(
        self,
        dest_dir: Path | str,
        asset_types: List[str] | str = "pdf",
        client: Optional[Any] = None,
    ) -> Dict[str, List[Path]]:
        """Download assets for all pages in this issue.

        Args:
            dest_dir: Target directory.
            asset_types: Asset type(s) to download: 'pdf', 'txt', 'alto', 'jp2', or 'all'.
            client: Optional downloader client.

        Returns:
            Dictionary mapping asset type to list of downloaded Path objects.
        """
        dest_path = Path(dest_dir) / f"{self.lccn}_{self.date}_ed-{self.edition}"
        dest_path.mkdir(parents=True, exist_ok=True)

        if isinstance(asset_types, str):
            if asset_types == "all":
                types = ["pdf", "txt", "alto", "jp2", "image"]
            else:
                types = [asset_types]
        else:
            types = asset_types

        results: Dict[str, List[Path]] = {t: [] for t in types}

        for page in self.pages:
            if "pdf" in types:
                results["pdf"].append(page.download_pdf(dest_path, client=client))
            if "txt" in types:
                results["txt"].append(page.download_text(dest_path, client=client))
            if "alto" in types and page.alto_xml_url:
                results["alto"].append(page.download_alto(dest_path, client=client))
            if "jp2" in types and page.jp2_url:
                results["jp2"].append(page.download_jp2(dest_path, client=client))
            if ("image" in types or "jpg" in types) and page.iiif_image_url():
                results["image" if "image" in types else "jpg"].append(
                    page.download_image(dest_path, client=client)
                )

        return results


class DownloadRecord(BaseModel):
    """Model tracking a downloaded asset in the local SQLite catalog."""

    item_id: str
    asset_type: str
    file_path: str
    file_size_bytes: int
    sha256: Optional[str] = None
    batch_name: Optional[str] = None
    lccn: Optional[str] = None
    date: Optional[str] = None
    page: Optional[int] = None
    downloaded_at: str
