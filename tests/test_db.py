"""Tests for SQLite CatalogDB."""

from pathlib import Path
from loc_chronicling_america.db import CatalogDB
from loc_chronicling_america.models import BatchInfo, TitleInfo


def test_catalog_db_batches(tmp_path: Path):
    db_file = tmp_path / "test_catalog.sqlite"
    db = CatalogDB(db_file)

    batch = BatchInfo(
        name="nbu_indescribablebeast_ver01",
        awardee="nbu",
        state="NE",
        archive_url="https://chroniclingamerica.loc.gov/data/ocr/nbu_indescribablebeast_ver01.tar.bz2",
        raw_batch_url="https://chroniclingamerica.loc.gov/data/batches/nbu_indescribablebeast_ver01/",
        size_bytes=895684461,
        page_count=6500,
        issue_count=1324,
        lccns=["00225879", "sn84020108"],
        sha256="abc123def456",
        ingested_date="2020-10-10",
    )

    db.upsert_batch(batch)
    assert db.count_batches() == 1

    fetched = db.get_batch("nbu_indescribablebeast_ver01")
    assert fetched is not None
    assert fetched.name == "nbu_indescribablebeast_ver01"
    assert fetched.awardee == "nbu"
    assert fetched.state == "NE"
    assert fetched.size_mb > 800
    assert "00225879" in fetched.lccns

    # Search by state
    results = db.search_batches(state="NE")
    assert len(results) == 1
    assert results[0].name == "nbu_indescribablebeast_ver01"

    # Search by LCCN
    results_lccn = db.search_batches(lccn="00225879")
    assert len(results_lccn) == 1


def test_catalog_db_titles(tmp_path: Path):
    db_file = tmp_path / "test_catalog.sqlite"
    db = CatalogDB(db_file)

    title = TitleInfo(
        lccn="00225879",
        name="The Monitor",
        state="Nebraska",
        city="Omaha",
        county="Douglas",
        start_year=1915,
        end_year=1928,
        issue_count=650,
        ethnicity="African American",
    )

    db.upsert_title(title)
    assert db.count_titles() == 1

    fetched = db.get_title("00225879")
    assert fetched is not None
    assert fetched.name == "The Monitor"
    assert fetched.city == "Omaha"
    assert fetched.formatted_years == "1915-1928"

    # Search titles
    results = db.search_titles(query="Monitor", state="Nebraska")
    assert len(results) == 1
    assert results[0].lccn == "00225879"


def test_catalog_db_downloads(tmp_path: Path):
    db_file = tmp_path / "test_catalog.sqlite"
    db = CatalogDB(db_file)

    dummy_file = tmp_path / "sample.pdf"
    dummy_file.write_bytes(b"%PDF-1.4 sample content")

    rec = db.record_download(
        item_id="00225879/1915-07-03/seq-1",
        asset_type="pdf",
        file_path=dummy_file,
        file_size_bytes=len(dummy_file.read_bytes()),
        batch_name="nbu_indescribablebeast_ver01",
        lccn="00225879",
        date="1915-07-03",
        page=1,
    )

    assert rec.asset_type == "pdf"
    assert rec.file_size_bytes > 0

    # Check is_downloaded
    found = db.is_downloaded("00225879/1915-07-03/seq-1", "pdf")
    assert found is not None
    assert found == dummy_file.resolve()

    # List downloads
    dls = db.list_downloads(batch_name="nbu_indescribablebeast_ver01")
    assert len(dls) == 1
    assert dls[0].lccn == "00225879"


def test_chronicling_america_with_catalog_db(tmp_path: Path):
    from loc_chronicling_america.client import ChroniclingAmerica
    db_file = tmp_path / "test_catalog.sqlite"
    db = CatalogDB(db_file)
    client = ChroniclingAmerica(catalog_db=db)
    assert client.catalog is db

