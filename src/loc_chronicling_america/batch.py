"""Batch representation for discovering, streaming, and downloading NDNP bulk newspaper packages."""

from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from .downloader import Downloader, get_default_downloader
from .models import BatchInfo
from .parsers.mets import BatchIssueRef, parse_batch_manifest


@dataclass
class ArchivePageItem:
    """A page extracted from a bulk OCR archive (.tar.bz2)."""

    lccn: str
    year: str
    month: str
    day: str
    edition: int
    sequence: int
    text: Optional[str] = None
    alto_xml: Optional[str] = None
    reel_id: Optional[str] = None
    layout_data: Optional[Dict[str, Any]] = None

    @property
    def date(self) -> str:
        return f"{self.year}-{self.month}-{self.day}"

    @property
    def loc_url(self) -> str:
        return f"https://www.loc.gov/resource/{self.lccn}/{self.date}/ed-{self.edition}/?sp={self.sequence}"

    @property
    def pdf_url(self) -> str:
        return f"https://chroniclingamerica.loc.gov/lccn/{self.lccn}/{self.date}/ed-{self.edition}/seq-{self.sequence}.pdf"

    @property
    def image_url(self) -> str:
        return f"https://chroniclingamerica.loc.gov/lccn/{self.lccn}/{self.date}/ed-{self.edition}/seq-{self.sequence}.jp2"


class Batch:
    """High-level interface for inspecting, downloading, and reading a Chronicling America batch."""

    def __init__(
        self,
        name: str,
        info: Optional[BatchInfo] = None,
        downloader: Optional[Downloader] = None,
    ):
        clean_name = name[len("batch_"):] if name.startswith("batch_") else name
        self.name = clean_name
        self._info = info
        self.downloader = downloader or get_default_downloader()

        self.bulk_archive_url = f"https://chroniclingamerica.loc.gov/data/ocr/{self.name}.tar.bz2"
        self.raw_batch_url = f"https://chroniclingamerica.loc.gov/data/batches/{self.name}/"
        self.metadata_url = f"https://chroniclingamerica.loc.gov/data/ocr/metadata/{self.name}.tar.bz2.json"

    def get_info(self) -> BatchInfo:
        """Fetch or return cached metadata for this batch."""
        if self._info and self._info.size_bytes is not None:
            return self._info

        # Check local SQLite catalog first
        from .db import CatalogDB
        db_info = CatalogDB().get_batch(self.name)
        if db_info and db_info.size_bytes:
            self._info = db_info
            return self._info

        try:
            data = self.downloader.fetch_json(self.metadata_url)
            awardee = self.name.split("_")[0] if "_" in self.name else None
            self._info = BatchInfo(
                name=self.name,
                awardee=awardee,
                archive_url=data.get("url") or self.bulk_archive_url,
                raw_batch_url=self.raw_batch_url,
                size_bytes=data.get("size") or (db_info.size_bytes if db_info else None),
                page_count=data.get("page_count"),
                issue_count=data.get("issue_count"),
                lccns=data.get("lccns", []),
                sha256=data.get("sha256"),
                ingested_date=data.get("ingested"),
            )
            return self._info
        except Exception:
            if db_info:
                self._info = db_info
                return self._info
            awardee = self.name.split("_")[0] if "_" in self.name else None
            self._info = BatchInfo(
                name=self.name,
                awardee=awardee,
                archive_url=self.bulk_archive_url,
                raw_batch_url=self.raw_batch_url,
            )
            return self._info

    def get_bag_info(self) -> Dict[str, str]:
        """Fetch and parse bag-info.txt from the raw BagIt batch directory."""
        url = f"{self.raw_batch_url}bag-info.txt"
        text = self.downloader.fetch_text(url)
        info: Dict[str, str] = {}
        for line in text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()
        return info

    def get_manifest_md5(self) -> Dict[str, str]:
        """Fetch manifest-md5.txt listing all files in the batch and their MD5 checksums."""
        url = f"{self.raw_batch_url}manifest-md5.txt"
        text = self.downloader.fetch_text(url)
        manifest: Dict[str, str] = {}
        for line in text.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                md5, filepath = parts
                manifest[filepath.strip()] = md5.strip()
        return manifest

    def get_issue_refs(self) -> List[BatchIssueRef]:
        """Fetch and parse batch_1.xml or batch.xml listing all issues in this batch."""
        for filename in ["batch_1.xml", "batch.xml"]:
            try:
                url = f"{self.raw_batch_url}data/{filename}"
                text = self.downloader.fetch_text(url)
                return parse_batch_manifest(text)
            except Exception:
                continue
        return []

    def download(
        self,
        dest_dir: Path | str,
        verify_checksum: bool = True,
        show_progress: bool = True,
    ) -> Path:
        """Download the .tar.bz2 bulk OCR archive for this batch.

        Args:
            dest_dir: Directory where the .tar.bz2 file will be saved.
            verify_checksum: Whether to verify SHA-256 against published metadata.
            show_progress: Show download progress bar.

        Returns:
            Path to the downloaded archive.
        """
        dest_path = Path(dest_dir) / f"{self.name}.tar.bz2"
        expected_sha256 = None

        if verify_checksum:
            try:
                info = self.get_info()
                expected_sha256 = info.sha256
            except Exception:
                pass

        return self.downloader.download_file(
            url=self.bulk_archive_url,
            dest_path=dest_path,
            expected_sha256=expected_sha256,
            show_progress=show_progress,
            description=f"Batch {self.name}",
        )

    def iter_archive(
        self,
        archive_path: Optional[Path | str] = None,
        extract_xml: bool = False,
    ) -> Iterator[ArchivePageItem]:
        """Iterate over all pages directly from a downloaded or streamed .tar.bz2 archive.

        Does NOT require extracting gigabytes of files to disk.

        Args:
            archive_path: Path to local .tar.bz2 archive. If None, looks for previously downloaded archive.
            extract_xml: Whether to also parse/load full ALTO XML for each page (slower, high memory).

        Yields:
            ArchivePageItem with plain text, metadata, and optional ALTO XML.
        """
        if not archive_path:
            raise ValueError("archive_path must be specified or batch must be downloaded first.")

        path = Path(archive_path)
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {path}")

        # Structure inside tar:
        # Standard: {lccn}/{year}/{month}/{day}/ed-{edition}/seq-{seq}/ocr.txt
        # With reel: {lccn}/{reel}/{year}/{month}/{day}/ed-{edition}/seq-{seq}/ocr.txt
        pattern_reel = re.compile(
            r"(?:.*?/)?(?P<lccn>[^/]+)/(?P<reel>\d{7,})/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/ed-(?P<edition>\d+)/seq-(?P<seq>\d+)/ocr\.(?P<ext>txt|xml)$"
        )
        pattern_std = re.compile(
            r"(?:.*?/)?(?P<lccn>[^/]+)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/ed-(?P<edition>\d+)/seq-(?P<seq>\d+)/ocr\.(?P<ext>txt|xml)$"
        )

        pages_dict: Dict[tuple, ArchivePageItem] = {}

        with tarfile.open(path, mode="r:bz2") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                m = pattern_reel.search(member.name) or pattern_std.search(member.name)
                if not m:
                    continue

                lccn = m.group("lccn")
                reel = m.group("reel") if "reel" in m.groupdict() else None
                year = m.group("year")
                month = m.group("month")
                day = m.group("day")
                ed_str = m.group("edition")
                seq_str = m.group("seq")
                ext = m.group("ext")
                key = (lccn, year, month, day, int(ed_str), int(seq_str))

                if key not in pages_dict:
                    pages_dict[key] = ArchivePageItem(
                        lccn=lccn,
                        year=year,
                        month=month,
                        day=day,
                        edition=int(ed_str),
                        sequence=int(seq_str),
                        reel_id=reel,
                    )

                page_item = pages_dict[key]
                if reel and not page_item.reel_id:
                    page_item.reel_id = reel

                if ext == "txt":
                    f = tar.extractfile(member)
                    if f:
                        page_item.text = f.read().decode("utf-8", errors="replace")
                elif ext == "xml" and extract_xml:
                    f = tar.extractfile(member)
                    if f:
                        xml_bytes = f.read()
                        page_item.alto_xml = xml_bytes.decode("utf-8", errors="replace")
                        try:
                            from loc_chronicling_america.parsers.alto import parse_alto_xml
                            doc = parse_alto_xml(xml_bytes)
                            page_item.layout_data = doc.to_layout_dict()
                            if not page_item.text:
                                page_item.text = doc.extract_full_text()
                        except Exception:
                            pass

                # If both or txt is loaded, yield and clear to save memory
                if page_item.text is not None and (not extract_xml or page_item.alto_xml is not None):
                    yield page_item
                    del pages_dict[key]

        # Flush any remaining
        for item in pages_dict.values():
            yield item

    def __repr__(self) -> str:
        return f"<Batch name='{self.name}'>"
