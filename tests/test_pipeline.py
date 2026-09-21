"""Tests for the streaming batch transmutation pipeline."""

import gzip
import json
import tarfile
from pathlib import Path
from loc_chronicling_america.db import CatalogDB
from loc_chronicling_america.hf import HuggingFaceDatasetManager
from loc_chronicling_america.models import BatchInfo, TitleInfo
from loc_chronicling_america.pipeline import BatchPipeline, slugify


def test_slugify():
    assert slugify("The Monitor (Omaha, Neb.)") == "the-monitor-omaha-neb"
    assert slugify("New York") == "new-york"
    assert slugify("St. Louis") == "st-louis"


def test_pipeline_queue_db(tmp_path):
    db_path = tmp_path / "test_catalog.sqlite"
    db = CatalogDB(db_path=db_path)

    # Insert sample batches
    batches = [
        BatchInfo(
            name="batch_ne_01",
            awardee="nbu",
            state="NE",
            archive_url="https://example.com/b1.tar.bz2",
            raw_batch_url="https://example.com/b1/",
            size_bytes=5000,
        ),
        BatchInfo(
            name="batch_oh_01",
            awardee="ohu",
            state="OH",
            archive_url="https://example.com/b2.tar.bz2",
            raw_batch_url="https://example.com/b2/",
            size_bytes=8000,
        ),
    ]
    db.upsert_batches(batches)

    # 1. Initialize queue
    pending_count = db.init_pipeline_batches()
    assert pending_count == 2

    # 2. Filter by state
    ne_pending = db.get_pending_pipeline_batches(state="NE")
    assert len(ne_pending) == 1
    assert ne_pending[0].name == "batch_ne_01"

    # 3. Transitions
    db.mark_pipeline_batch_processing("batch_ne_01")
    summary = db.get_pipeline_summary()
    assert summary["processing"] == 1
    assert summary["pending"] == 1

    db.mark_pipeline_batch_completed("batch_ne_01", pages_extracted=150, output_files=["data/ne/paper/1915.jsonl.gz"])
    summary = db.get_pipeline_summary()
    assert summary["completed"] == 1
    assert summary["pages_extracted"] == 150

    db.mark_pipeline_batch_failed("batch_oh_01", error_message="Connection timeout")
    summary = db.get_pipeline_summary()
    assert summary["failed"] == 1


def test_pipeline_batch_processing(tmp_path):
    db_path = tmp_path / "test_catalog.sqlite"
    out_dir = tmp_path / "export_output"
    scratch_dir = tmp_path / "scratch"

    db = CatalogDB(db_path=db_path)

    # Seed title metadata
    db.upsert_title(
        TitleInfo(
            lccn="sn85026945",
            name="The Evening Herald",
            state="Nebraska",
            city="Omaha",
            start_year=1900,
            end_year=1920,
        )
    )

    # Create a synthetic .tar.bz2 bulk archive
    tar_path = scratch_dir / "test_batch_01.tar.bz2"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "w:bz2") as tar:
        # File 1: page 1
        txt_data1 = b"Front page headline text of the evening herald."
        info1 = tarfile.TarInfo(name="sn85026945/1915/07/03/ed-1/seq-1/ocr.txt")
        info1.size = len(txt_data1)
        import io
        tar.addfile(info1, io.BytesIO(txt_data1))

        # File 2: page 2
        txt_data2 = b"Second page market reports and sports news."
        info2 = tarfile.TarInfo(name="sn85026945/1915/07/03/ed-1/seq-2/ocr.txt")
        info2.size = len(txt_data2)
        tar.addfile(info2, io.BytesIO(txt_data2))

    batch_info = BatchInfo(
        name="test_batch_01",
        awardee="nbu",
        state="NE",
        archive_url="https://example.com/test_batch_01.tar.bz2",
        raw_batch_url="https://example.com/test_batch_01/",
    )

    pipeline = BatchPipeline(
        output_dir=out_dir,
        scratch_dir=scratch_dir,
        db_path=db_path,
        keep_tar=False,
    )

    # Execute batch extraction
    pages_extracted, written_files = pipeline.process_batch(batch_info)

    assert pages_extracted == 2
    assert len(written_files) == 1

    # Verify output hierarchy: data/nebraska/the-evening-herald_omaha_sn85026945/1915.jsonl.gz
    expected_file = out_dir / written_files[0]
    assert expected_file.exists()
    assert "nebraska" in written_files[0]
    assert "the-evening-herald_omaha_sn85026945" in written_files[0]
    assert written_files[0].endswith("1915.jsonl.gz")

    # Scratch tarball should be purged
    assert not tar_path.exists(), "Temporary .tar.bz2 was not deleted"

    # Verify JSONL lines inside .jsonl.gz
    with gzip.open(expected_file, "rt", encoding="utf-8") as gz_in:
        lines = gz_in.readlines()
        assert len(lines) == 2
        doc1 = json.loads(lines[0])
        assert doc1["_id"] == "sn85026945_1915-07-03_ed-1_seq-1"
        assert doc1["text"] == "Front page headline text of the evening herald."
        assert doc1["newspaper_title"] == "The Evening Herald"
        assert doc1["city"] == "Omaha"
        assert doc1["state"] == "Nebraska"
        assert doc1["year"] == 1915
        assert doc1["sequence"] == 1

    # Test master index generation
    jsonl_cat, parquet_cat = pipeline.update_catalog_index()
    assert jsonl_cat.exists()
    if parquet_cat:
        assert parquet_cat.exists()

    with open(jsonl_cat, "r", encoding="utf-8") as f:
        cat_records = [json.loads(line) for line in f]
        assert len(cat_records) == 1
        assert cat_records[0]["lccn"] == "sn85026945"
        assert cat_records[0]["newspaper_title"] == "The Evening Herald"
        assert cat_records[0]["file_count"] == 1


def test_hf_dataset_card_generation(tmp_path):
    mgr = HuggingFaceDatasetManager(repo_id="tcondello/test-dataset")
    readme_path = tmp_path / "README.md"
    mgr.generate_readme(readme_path)
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "tcondello/test-dataset" in content
    assert "task_categories:" in content
    assert "Pinecone Document Schema" in content
