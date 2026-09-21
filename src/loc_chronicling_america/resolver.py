"""Resolves Library of Congress Chronicling America URLs and identifiers to complete IssueRecords and their source batches."""

from __future__ import annotations

import re
from typing import Optional, Tuple
from urllib.parse import urlparse
from .downloader import Downloader, get_default_downloader
from .models import IssueRecord, PageRecord


class RecordResolver:
    """Resolves LoC URLs (resource/item/lccn), identifiers, and dates to complete IssueRecords."""

    def __init__(self, downloader: Optional[Downloader] = None):
        self.downloader = downloader or get_default_downloader()

    def parse_url(self, url: str) -> Tuple[str, str, int]:
        """Parse an LoC newspaper URL to extract (lccn, date, edition).

        Supports patterns like:
        - https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery
        - https://www.loc.gov/item/00225879/1915-07-03/ed-1/
        - https://chroniclingamerica.loc.gov/lccn/sn85026945/1847-03-03/ed-1/
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Match pattern: (resource|item|lccn)/{lccn}/{date}/ed-{edition}
        match = re.search(
            r"(?:resource|item|lccn)/([a-zA-Z0-9_\-]+)/(\d{4}-\d{2}-\d{2})(?:/ed-(\d+))?",
            path,
        )
        if not match:
            # Also match date with sequence without ed-, e.g. /1915070301/
            match_compact = re.search(
                r"(?:resource|item|lccn)/([a-zA-Z0-9_\-]+)/(\d{4})(\d{2})(\d{2})(\d{2})?",
                path,
            )
            if match_compact:
                lccn = match_compact.group(1)
                date = f"{match_compact.group(2)}-{match_compact.group(3)}-{match_compact.group(4)}"
                ed_str = match_compact.group(5)
                edition = int(ed_str) if ed_str else 1
                return lccn, date, edition

            raise ValueError(
                f"Could not extract LCCN, date, and edition from URL: {url}\n"
                f"Expected format like: https://www.loc.gov/resource/00225879/1915-07-03/ed-1/"
            )

        lccn = match.group(1)
        date = match.group(2)
        edition = int(match.group(3)) if match.group(3) else 1
        return lccn, date, edition

    def resolve(
        self,
        url_or_lccn: str,
        date: Optional[str] = None,
        edition: int = 1,
    ) -> IssueRecord:
        """Resolve a URL, or an LCCN + date into a complete IssueRecord.

        Args:
            url_or_lccn: Full LoC URL or LCCN identifier (e.g. '00225879').
            date: Publication date (YYYY-MM-DD), required if url_or_lccn is just an LCCN.
            edition: Edition number (default: 1).

        Returns:
            IssueRecord with title, batch information, and all page assets.
        """
        if "://" in url_or_lccn:
            lccn, pub_date, ed = self.parse_url(url_or_lccn)
        else:
            if not date:
                raise ValueError("Must provide 'date' parameter when passing an LCCN instead of a URL.")
            lccn = url_or_lccn
            pub_date = date
            ed = edition

        # Query LoC item endpoint as JSON
        item_api_url = f"https://www.loc.gov/item/{lccn}/{pub_date}/ed-{ed}/?fo=json"
        data = self.downloader.fetch_json(item_api_url)

        def _to_str(val: Any, default: Optional[str] = None) -> Optional[str]:
            if val is None:
                return default
            if isinstance(val, list):
                return str(val[0]) if val else default
            if isinstance(val, dict):
                return str(val.get("label") or val.get("value") or default or "")
            return str(val)

        item = data.get("item", {})
        raw_title = _to_str(item.get("title"), default=f"{lccn} ({pub_date})")
        title = raw_title or f"{lccn} ({pub_date})"
        newspaper_title = _to_str(item.get("newspaper_title"), default=title) or title
        place_of_publication = _to_str(item.get("place_of_publication"))
        state = _to_str(item.get("location_state"))
        city = _to_str(item.get("location_city"))

        # Extract source batch name
        batches = item.get("batch", [])
        if not batches:
            raise ValueError(f"No batch found for record {lccn} on {pub_date} in LoC response.")

        batch_name = batches[0]
        # Clean up batch name if it has prefix
        if batch_name.startswith("batch_"):
            clean_batch_name = batch_name[len("batch_"):]
        else:
            clean_batch_name = batch_name

        # Construct bulk and raw URLs
        # LoC data endpoints use either batch_name or clean name
        bulk_ocr_url = f"https://chroniclingamerica.loc.gov/data/ocr/{clean_batch_name}.tar.bz2"
        raw_batch_url = f"https://chroniclingamerica.loc.gov/data/batches/{clean_batch_name}/"

        # Parse Pages from resources
        pages: list[PageRecord] = []
        resources = data.get("resources", [])
        if resources and "files" in resources[0]:
            file_groups = resources[0]["files"]
            for idx, group in enumerate(file_groups, start=1):
                pdf_url = None
                jp2_url = None
                alto_xml_url = None
                text_service_url = None
                thumbnail_url = None
                iiif_info_url = None
                reel_number = None
                width = None
                height = None
                page_title = f"Page {idx} of {title}"

                for f in group:
                    mime = f.get("mimetype", "")
                    url = f.get("url")

                    if mime == "application/pdf" and url:
                        pdf_url = url
                    elif mime == "image/jp2" and url:
                        jp2_url = url
                    elif mime == "text/xml" and url:
                        alto_xml_url = url
                    elif f.get("fulltext_service"):
                        text_service_url = f.get("fulltext_service")

                    if mime == "image/jpeg" and url and not thumbnail_url:
                        thumbnail_url = url
                    if f.get("info"):
                        iiif_info_url = f.get("info")
                    if f.get("reel_number"):
                        reel_number = f.get("reel_number")
                    if f.get("width"):
                        width = int(f.get("width"))
                    if f.get("height"):
                        height = int(f.get("height"))
                    if f.get("title"):
                        page_title = _to_str(f.get("title")) or page_title

                pages.append(
                    PageRecord(
                        sequence=idx,
                        title=page_title,
                        lccn=lccn,
                        date=pub_date,
                        edition=ed,
                        reel_number=reel_number,
                        width=width,
                        height=height,
                        pdf_url=pdf_url,
                        jp2_url=jp2_url,
                        alto_xml_url=alto_xml_url,
                        text_service_url=text_service_url,
                        thumbnail_url=thumbnail_url,
                        iiif_info_url=iiif_info_url,
                    )
                )

        return IssueRecord(
            title=title,
            newspaper_title=newspaper_title,
            lccn=lccn,
            date=pub_date,
            edition=ed,
            place_of_publication=place_of_publication,
            state=state,
            city=city,
            batch_name=clean_batch_name,
            bulk_ocr_url=bulk_ocr_url,
            raw_batch_url=raw_batch_url,
            loc_item_url=f"https://www.loc.gov/item/{lccn}/{pub_date}/ed-{ed}/",
            iiif_manifest_url=item.get("iiif_manifest_url"),
            pages=pages,
        )
