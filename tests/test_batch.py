"""Tests for Batch and archive streaming."""

import io
import tarfile
from pathlib import Path
from loc_chronicling_america.batch import Batch


def test_batch_object():
    batch = Batch("nbu_indescribablebeast_ver01")
    assert batch.name == "nbu_indescribablebeast_ver01"
    assert "nbu_indescribablebeast_ver01.tar.bz2" in batch.bulk_archive_url
    assert "nbu_indescribablebeast_ver01/" in batch.raw_batch_url


def test_batch_archive_stream(tmp_path: Path):
    """Test iterating over a synthesized .tar.bz2 archive without disk extraction."""
    archive_path = tmp_path / "test_batch_ver01.tar.bz2"

    txt_content = b"Headline: Great News Today In Omaha"
    xml_content = b"<alto><Page/></alto>"

    # Create synthetic archive with reel_id in path
    with tarfile.open(archive_path, mode="w:bz2") as tar:
        t_info = tarfile.TarInfo(name="00225879/00175032496/1915/07/03/ed-1/seq-1/ocr.txt")
        t_info.size = len(txt_content)
        tar.addfile(t_info, io.BytesIO(txt_content))

        x_info = tarfile.TarInfo(name="00225879/00175032496/1915/07/03/ed-1/seq-1/ocr.xml")
        x_info.size = len(xml_content)
        tar.addfile(x_info, io.BytesIO(xml_content))

    batch = Batch("test_batch_ver01")
    items = list(batch.iter_archive(archive_path, extract_xml=True))

    assert len(items) == 1
    item = items[0]
    assert item.lccn == "00225879"
    assert item.reel_id == "00175032496"
    assert item.date == "1915-07-03"
    assert item.edition == 1
    assert item.sequence == 1
    assert "Great News" in (item.text or "")
    assert "<alto>" in (item.alto_xml or "")
    assert item.layout_data is not None
    assert item.loc_url == "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?sp=1"
    assert item.pdf_url == "https://chroniclingamerica.loc.gov/lccn/00225879/1915-07-03/ed-1/seq-1.pdf"
    assert item.image_url == "https://chroniclingamerica.loc.gov/lccn/00225879/1915-07-03/ed-1/seq-1.jp2"
