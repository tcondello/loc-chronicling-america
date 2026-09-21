"""Embedded SQLite database for indexing Chronicling America batches, newspaper titles, and downloaded files."""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from .models import BatchInfo, DownloadRecord, TitleInfo


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "chronam"
DEFAULT_DB_PATH = DEFAULT_CACHE_DIR / "catalog.sqlite"


class CatalogDB:
    """SQLite-backed metadata index and local download registry."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or os.environ.get("CHRONAM_DB_PATH", DEFAULT_DB_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    name TEXT PRIMARY KEY,
                    awardee TEXT,
                    state TEXT,
                    archive_url TEXT NOT NULL,
                    raw_batch_url TEXT NOT NULL,
                    size_bytes INTEGER,
                    page_count INTEGER,
                    issue_count INTEGER,
                    lccns TEXT,  -- JSON list of LCCN strings
                    sha256 TEXT,
                    ingested_date TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_batches_state ON batches(state);
                CREATE INDEX IF NOT EXISTS idx_batches_awardee ON batches(awardee);

                CREATE TABLE IF NOT EXISTS titles (
                    lccn TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    state TEXT,
                    city TEXT,
                    county TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    issue_count INTEGER,
                    first_issue_date TEXT,
                    last_issue_date TEXT,
                    ethnicity TEXT,
                    language TEXT,
                    loc_url TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_titles_name ON titles(name);
                CREATE INDEX IF NOT EXISTS idx_titles_state ON titles(state);
                CREATE INDEX IF NOT EXISTS idx_titles_city ON titles(city);

                CREATE TABLE IF NOT EXISTS local_downloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    sha256 TEXT,
                    batch_name TEXT,
                    lccn TEXT,
                    date TEXT,
                    page INTEGER,
                    downloaded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_downloads_item ON local_downloads(item_id, asset_type);
                CREATE INDEX IF NOT EXISTS idx_downloads_batch ON local_downloads(batch_name);
                """
            )

    # ----------------------------------------------------------------------
    # Download Tracking
    # ----------------------------------------------------------------------

    def record_download(
        self,
        item_id: str,
        asset_type: str,
        file_path: Path | str,
        file_size_bytes: int,
        sha256: Optional[str] = None,
        batch_name: Optional[str] = None,
        lccn: Optional[str] = None,
        date: Optional[str] = None,
        page: Optional[int] = None,
    ) -> DownloadRecord:
        """Register a completed file download."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        resolved_path = str(Path(file_path).resolve())

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO local_downloads
                (item_id, asset_type, file_path, file_size_bytes, sha256, batch_name, lccn, date, page, downloaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    asset_type,
                    resolved_path,
                    file_size_bytes,
                    sha256,
                    batch_name,
                    lccn,
                    date,
                    page,
                    now_iso,
                ),
            )

        return DownloadRecord(
            item_id=item_id,
            asset_type=asset_type,
            file_path=resolved_path,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            batch_name=batch_name,
            lccn=lccn,
            date=date,
            page=page,
            downloaded_at=now_iso,
        )

    def is_downloaded(self, item_id: str, asset_type: str) -> Optional[Path]:
        """Check if an asset has been downloaded locally and still exists on disk."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT file_path FROM local_downloads WHERE item_id = ? AND asset_type = ? ORDER BY id DESC LIMIT 1",
                (item_id, asset_type),
            )
            row = cur.fetchone()
            if row:
                path = Path(row["file_path"])
                if path.exists():
                    return path
        return None

    def list_downloads(
        self,
        batch_name: Optional[str] = None,
        lccn: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> List[DownloadRecord]:
        """List local downloads matching criteria."""
        query = "SELECT * FROM local_downloads WHERE 1=1"
        params: List[Any] = []

        if batch_name:
            query += " AND batch_name = ?"
            params.append(batch_name)
        if lccn:
            query += " AND lccn = ?"
            params.append(lccn)
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)

        query += " ORDER BY downloaded_at DESC"

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            records = []
            for row in cur.fetchall():
                records.append(
                    DownloadRecord(
                        item_id=row["item_id"],
                        asset_type=row["asset_type"],
                        file_path=row["file_path"],
                        file_size_bytes=row["file_size_bytes"],
                        sha256=row["sha256"],
                        batch_name=row["batch_name"],
                        lccn=row["lccn"],
                        date=row["date"],
                        page=row["page"],
                        downloaded_at=row["downloaded_at"],
                    )
                )
            return records

    # ----------------------------------------------------------------------
    # Batches Catalog
    # ----------------------------------------------------------------------

    def upsert_batch(self, batch: BatchInfo) -> None:
        """Insert or update batch metadata."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batches
                (name, awardee, state, archive_url, raw_batch_url, size_bytes, page_count, issue_count, lccns, sha256, ingested_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch.name,
                    batch.awardee,
                    batch.state,
                    batch.archive_url,
                    batch.raw_batch_url,
                    batch.size_bytes,
                    batch.page_count,
                    batch.issue_count,
                    json.dumps(batch.lccns),
                    batch.sha256,
                    batch.ingested_date,
                ),
            )

    def upsert_batches(self, batches: List[BatchInfo]) -> None:
        """Bulk insert or update batches."""
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO batches
                (name, awardee, state, archive_url, raw_batch_url, size_bytes, page_count, issue_count, lccns, sha256, ingested_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        b.name,
                        b.awardee,
                        b.state,
                        b.archive_url,
                        b.raw_batch_url,
                        b.size_bytes,
                        b.page_count,
                        b.issue_count,
                        json.dumps(b.lccns),
                        b.sha256,
                        b.ingested_date,
                    )
                    for b in batches
                ],
            )

    def get_batch(self, name: str) -> Optional[BatchInfo]:
        """Get batch by name."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM batches WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                return self._row_to_batch(row)
        return None

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
        """Search batches matching criteria."""
        sql = "SELECT * FROM batches WHERE 1=1"
        params: List[Any] = []

        if query:
            sql += " AND (name LIKE ? OR lccns LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        if state:
            sql += " AND (state LIKE ? OR awardee LIKE ?)"
            params.extend([f"%{state}%", f"%{state}%"])
        if awardee:
            sql += " AND awardee = ?"
            params.append(awardee)
        if lccn:
            sql += " AND lccns LIKE ?"
            params.append(f"%{lccn}%")
        if min_pages:
            sql += " AND page_count >= ?"
            params.append(min_pages)
        if max_size_mb:
            sql += " AND size_bytes <= ?"
            params.append(int(max_size_mb * 1024 * 1024))

        sql += " ORDER BY name ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(sql, params)
            return [self._row_to_batch(r) for r in cur.fetchall()]

    def count_batches(self) -> int:
        """Count total batches in catalog."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM batches")
            return cur.fetchone()[0]

    # ----------------------------------------------------------------------
    # Titles Catalog
    # ----------------------------------------------------------------------

    def upsert_title(self, title: TitleInfo) -> None:
        """Insert or update newspaper title."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO titles
                (lccn, name, state, city, county, start_year, end_year, issue_count, first_issue_date, last_issue_date, ethnicity, language, loc_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title.lccn,
                    title.name,
                    title.state,
                    title.city,
                    title.county,
                    title.start_year,
                    title.end_year,
                    title.issue_count,
                    title.first_issue_date,
                    title.last_issue_date,
                    title.ethnicity,
                    title.language,
                    title.loc_url,
                ),
            )

    def upsert_titles(self, titles: List[TitleInfo]) -> None:
        """Bulk insert or update newspaper titles."""
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO titles
                (lccn, name, state, city, county, start_year, end_year, issue_count, first_issue_date, last_issue_date, ethnicity, language, loc_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.lccn,
                        t.name,
                        t.state,
                        t.city,
                        t.county,
                        t.start_year,
                        t.end_year,
                        t.issue_count,
                        t.first_issue_date,
                        t.last_issue_date,
                        t.ethnicity,
                        t.language,
                        t.loc_url,
                    )
                    for t in titles
                ],
            )

    def get_title(self, lccn: str) -> Optional[TitleInfo]:
        """Get title by LCCN."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM titles WHERE lccn = ?", (lccn,))
            row = cur.fetchone()
            if row:
                return self._row_to_title(row)
        return None

    def search_titles(
        self,
        query: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        year: Optional[int] = None,
        ethnicity: Optional[str] = None,
        limit: int = 50,
    ) -> List[TitleInfo]:
        """Search titles by name, location, year, or demographic."""
        sql = "SELECT * FROM titles WHERE 1=1"
        params: List[Any] = []

        if query:
            sql += " AND (name LIKE ? OR lccn LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        if state:
            sql += " AND state LIKE ?"
            params.append(f"%{state}%")
        if city:
            sql += " AND city LIKE ?"
            params.append(f"%{city}%")
        if year:
            sql += " AND (start_year <= ? AND (end_year >= ? OR end_year IS NULL))"
            params.extend([year, year])
        if ethnicity:
            sql += " AND ethnicity LIKE ?"
            params.append(f"%{ethnicity}%")

        sql += " ORDER BY name ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(sql, params)
            return [self._row_to_title(r) for r in cur.fetchall()]

    def count_titles(self) -> int:
        """Count total newspaper titles in catalog."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM titles")
            return cur.fetchone()[0]

    # ----------------------------------------------------------------------
    # Helper Mappers
    # ----------------------------------------------------------------------

    @staticmethod
    def _row_to_batch(row: sqlite3.Row) -> BatchInfo:
        lccns = []
        if row["lccns"]:
            try:
                lccns = json.loads(row["lccns"])
            except Exception:
                lccns = [s.strip() for s in row["lccns"].split(",") if s.strip()]

        return BatchInfo(
            name=row["name"],
            awardee=row["awardee"],
            state=row["state"],
            archive_url=row["archive_url"],
            raw_batch_url=row["raw_batch_url"],
            size_bytes=row["size_bytes"],
            page_count=row["page_count"],
            issue_count=row["issue_count"],
            lccns=lccns,
            sha256=row["sha256"],
            ingested_date=row["ingested_date"],
        )

    @staticmethod
    def _row_to_title(row: sqlite3.Row) -> TitleInfo:
        return TitleInfo(
            lccn=row["lccn"],
            name=row["name"],
            state=row["state"],
            city=row["city"],
            county=row["county"],
            start_year=row["start_year"],
            end_year=row["end_year"],
            issue_count=row["issue_count"],
            first_issue_date=row["first_issue_date"],
            last_issue_date=row["last_issue_date"],
            ethnicity=row["ethnicity"],
            language=row["language"] or "English",
            loc_url=row["loc_url"],
        )
