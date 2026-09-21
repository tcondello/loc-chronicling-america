"""Main client interface for loc-chronicling-america."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional
from .batch import Batch
from .db import CatalogDB
from .downloader import Downloader, get_default_downloader
from .models import BatchInfo, IssueRecord, TitleInfo
from .resolver import RecordResolver


class ChroniclingAmerica:
    """Primary client for discovering, resolving, and downloading Chronicling America records."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        downloader: Optional[Downloader] = None,
        auto_init_catalog: bool = False,
    ):
        self.downloader = downloader or get_default_downloader()
        self.catalog = CatalogDB(db_path=db_path)
        self.resolver = RecordResolver(downloader=self.downloader)

        if auto_init_catalog and self.catalog.count_batches() == 0:
            self.sync_batches()

    # ----------------------------------------------------------------------
    # Core Resolution: URL / LCCN -> Record
    # ----------------------------------------------------------------------

    def resolve(
        self,
        url_or_lccn: str,
        date: Optional[str] = None,
        edition: int = 1,
    ) -> IssueRecord:
        """Resolve an LoC URL (resource, item, gallery) or LCCN + date into a complete IssueRecord.

        Example:
            record = loc.resolve("https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery")
            print(record.title)
            print(record.batch_name)  # Exact batch name: 'nbu_indescribablebeast_ver01'
            print(record.pages[0].get_text()[:200])

        Args:
            url_or_lccn: LoC resource/item URL or LCCN string.
            date: Publication date (YYYY-MM-DD), required only if url_or_lccn is an LCCN.
            edition: Edition number (default 1).

        Returns:
            IssueRecord containing metadata, pages, asset URLs, and exact batch info.
        """
        return self.resolver.resolve(url_or_lccn=url_or_lccn, date=date, edition=edition)

    # ----------------------------------------------------------------------
    # Batch Access
    # ----------------------------------------------------------------------

    def get_batch(self, name: str) -> Batch:
        """Get a Batch object for bulk inspections and downloads.

        Args:
            name: Batch identifier, e.g. 'nbu_indescribablebeast_ver01'.
        """
        info = self.catalog.get_batch(name)
        return Batch(name=name, info=info, downloader=self.downloader)

    def search_batches(
        self,
        query: Optional[str] = None,
        state: Optional[str] = None,
        awardee: Optional[str] = None,
        lccn: Optional[str] = None,
        min_pages: Optional[int] = None,
        max_size_mb: Optional[float] = None,
        limit: int = 50,
    ) -> List[BatchInfo]:
        """Search batches in the local SQLite catalog.

        If the catalog is currently empty, it will automatically populate from LoC endpoints.
        """
        if self.catalog.count_batches() == 0:
            self.sync_batches(limit=100)

        return self.catalog.search_batches(
            query=query,
            state=state,
            awardee=awardee,
            lccn=lccn,
            min_pages=min_pages,
            max_size_mb=max_size_mb,
            limit=limit,
        )

    # ----------------------------------------------------------------------
    # Titles Access
    # ----------------------------------------------------------------------

    def get_title(self, lccn: str) -> Optional[TitleInfo]:
        """Lookup newspaper title metadata by LCCN."""
        return self.catalog.get_title(lccn)

    def search_titles(
        self,
        query: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        year: Optional[int] = None,
        ethnicity: Optional[str] = None,
        limit: int = 50,
        live: bool = True,
    ) -> List[TitleInfo]:
        """Search newspaper titles in the local catalog with live LoC fallback.

        Args:
            query: Title search query (e.g. 'The Monitor').
            state: Filter by state (e.g. 'Nebraska').
            city: Filter by city (e.g. 'Omaha').
            year: Filter by active year.
            ethnicity: Filter by target demographic.
            limit: Maximum number of results.
            live: If True, query live LoC API when local results are empty and cache findings.
        """
        results = self.catalog.search_titles(
            query=query,
            state=state,
            city=city,
            year=year,
            ethnicity=ethnicity,
            limit=limit,
        )

        if not results and live and query:
            try:
                import urllib.parse
                q_terms = [query]
                if state:
                    q_terms.append(state)
                if city:
                    q_terms.append(city)
                full_q = "+".join(urllib.parse.quote_plus(term) for term in q_terms)
                search_url = f"https://www.loc.gov/collections/chronicling-america/titles/?fo=json&at=content.results&q={full_q}"
                data = self.downloader.fetch_json(search_url)
                items = data.get("content.results", [])
                new_titles: List[TitleInfo] = []
                for item in items:
                    lccn_list = item.get("number_lccn") or item.get("lccn") or []
                    lccn = lccn_list[0] if isinstance(lccn_list, list) and lccn_list else str(lccn_list)
                    if not lccn:
                        continue
                    name = item.get("title") or item.get("newspaper_title") or "Unknown"

                    loc_state = item.get("location_state")
                    st = None
                    if isinstance(loc_state, dict):
                        st = loc_state.get("label") or loc_state.get("value")
                    elif isinstance(loc_state, list) and loc_state:
                        st = str(loc_state[0])
                    elif loc_state:
                        st = str(loc_state)

                    loc_city = item.get("location_city")
                    ct = loc_city[0] if isinstance(loc_city, list) and loc_city else str(loc_city or "")

                    first_issue = item.get("number_first_issue", {}).get("label") if isinstance(item.get("number_first_issue"), dict) else None
                    last_issue = item.get("number_last_issue", {}).get("label") if isinstance(item.get("number_last_issue"), dict) else None
                    sy = int(first_issue[:4]) if first_issue and len(first_issue) >= 4 and first_issue[:4].isdigit() else None
                    ey = int(last_issue[:4]) if last_issue and len(last_issue) >= 4 and last_issue[:4].isdigit() else None

                    ic_val = item.get("number_issue_count")
                    ic = None
                    if isinstance(ic_val, dict) and "value" in ic_val:
                        try:
                            ic = int(ic_val["value"])
                        except Exception:
                            pass

                    eth = item.get("subject_ethnicity")
                    ethnicity_str = eth.get("label") if isinstance(eth, dict) else (str(eth) if eth else None)

                    new_titles.append(
                        TitleInfo(
                            lccn=lccn,
                            name=name,
                            state=st,
                            city=ct,
                            start_year=sy,
                            end_year=ey,
                            issue_count=ic,
                            first_issue_date=first_issue,
                            last_issue_date=last_issue,
                            ethnicity=ethnicity_str,
                            loc_url=item.get("url"),
                        )
                    )

                if new_titles:
                    self.catalog.upsert_titles(new_titles)
                    return self.catalog.search_titles(
                        query=query,
                        state=state,
                        city=city,
                        year=year,
                        ethnicity=ethnicity,
                        limit=limit,
                    )
            except Exception:
                pass

        return results

    # ----------------------------------------------------------------------
    # Catalog Synchronization
    # ----------------------------------------------------------------------

    def sync_batches(self, limit: Optional[int] = None) -> int:
        """Synchronize the batches list from Chronicling America into the local SQLite database.

        Parses https://chroniclingamerica.loc.gov/data/ocr/ and metadata endpoint.
        """
        url = "https://chroniclingamerica.loc.gov/data/ocr/"
        html = self.downloader.fetch_text(url)

        # Match table rows: <td><a href="{name}.tar.bz2">...</a></td><td>{date}</td><td>{size}</td>
        rows = re.findall(
            r'<tr>\s*<td><a href="([^"]+\.tar\.bz2)">[^<]+</a></td>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>',
            html,
        )

        def parse_size(s: str) -> int:
            s = s.strip()
            if not s:
                return 0
            parts = s.split()
            if len(parts) == 2:
                val, unit = float(parts[0]), parts[1].upper()
                if "KB" in unit:
                    return int(val * 1024)
                if "MB" in unit:
                    return int(val * 1024 * 1024)
                if "GB" in unit:
                    return int(val * 1024 * 1024 * 1024)
                if "BYTES" in unit:
                    return int(val)
            return 0

        batches: List[BatchInfo] = []
        target_rows = rows[:limit] if limit else rows

        for filename, date_str, size_str in target_rows:
            batch_name = filename.replace(".tar.bz2", "")
            awardee = batch_name.split("_")[0] if "_" in batch_name else None
            state = awardee.upper() if awardee and len(awardee) == 2 else None

            batches.append(
                BatchInfo(
                    name=batch_name,
                    awardee=awardee,
                    state=state,
                    archive_url=f"https://chroniclingamerica.loc.gov/data/ocr/{filename}",
                    raw_batch_url=f"https://chroniclingamerica.loc.gov/data/batches/{batch_name}/",
                    size_bytes=parse_size(size_str),
                    ingested_date=date_str.strip() or None,
                )
            )

        self.catalog.upsert_batches(batches)
        return len(batches)

    def sync_titles(self) -> int:
        """Synchronize digitized newspaper titles into the local SQLite catalog."""
        url = "https://chroniclingamerica.loc.gov/newspapers.json"
        data = self.downloader.fetch_json(url)

        results = data.get("content.results") or data.get("newspapers") or []
        titles: List[TitleInfo] = []

        for item in results:
            lccn_list = item.get("number_lccn") or item.get("lccn") or []
            lccn = lccn_list[0] if isinstance(lccn_list, list) and lccn_list else str(lccn_list)
            if not lccn:
                continue

            name = item.get("title") or item.get("newspaper_title") or "Unknown"

            # Parse state
            loc_state = item.get("location_state")
            state = None
            if isinstance(loc_state, dict):
                state = loc_state.get("label") or loc_state.get("value")
            elif isinstance(loc_state, list) and loc_state:
                state = str(loc_state[0])
            elif loc_state:
                state = str(loc_state)

            # Parse city
            loc_city = item.get("location_city")
            city = loc_city[0] if isinstance(loc_city, list) and loc_city else str(loc_city or "")

            # Issue count
            issue_count = None
            ic = item.get("number_issue_count")
            if isinstance(ic, dict) and "value" in ic:
                try:
                    issue_count = int(ic["value"])
                except Exception:
                    pass
            elif ic:
                try:
                    issue_count = int(ic)
                except Exception:
                    pass

            # Dates
            first_issue = None
            fi = item.get("number_first_issue")
            if isinstance(fi, dict) and "label" in fi:
                first_issue = fi["label"]

            last_issue = None
            li = item.get("number_last_issue")
            if isinstance(li, dict) and "label" in li:
                last_issue = li["label"]

            # Years
            start_year = int(first_issue[:4]) if first_issue and len(first_issue) >= 4 and first_issue[:4].isdigit() else None
            end_year = int(last_issue[:4]) if last_issue and len(last_issue) >= 4 and last_issue[:4].isdigit() else None

            # Ethnicity
            ethnicity = None
            eth = item.get("subject_ethnicity")
            if isinstance(eth, dict) and eth.get("label"):
                ethnicity = eth["label"]
            elif isinstance(eth, str) and eth:
                ethnicity = eth

            titles.append(
                TitleInfo(
                    lccn=lccn,
                    name=name,
                    state=state,
                    city=city,
                    start_year=start_year,
                    end_year=end_year,
                    issue_count=issue_count,
                    first_issue_date=first_issue,
                    last_issue_date=last_issue,
                    ethnicity=ethnicity,
                    loc_url=item.get("url"),
                )
            )

        self.catalog.upsert_titles(titles)
        return len(titles)
