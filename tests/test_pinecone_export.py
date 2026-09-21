"""Tests for Pinecone document schema export formatting."""

import json
from loc_chronicling_america.models import PageRecord, IssueRecord


def test_page_record_to_pinecone_document():
    page = PageRecord(
        sequence=1,
        title="Page 1 of The Monitor, July 3, 1915",
        lccn="00225879",
        date="1915-07-03",
        edition=1,
        width=3600,
        height=5200,
        pdf_url="https://example.com/page1.pdf",
    )
    # Set cache directly to avoid network calls during unit test
    page._text_cache = "Historic front page text of The Monitor in Omaha."

    doc = page.to_pinecone_document(
        text_field="text",
        extra_metadata={"newspaper_title": "The Monitor", "city": "Omaha", "state": "Nebraska"},
    )

    # 1. Check required _id
    assert "_id" in doc
    assert doc["_id"] == "00225879_1915-07-03_ed-1_seq-1"

    # 2. Check full-text search field
    assert "text" in doc
    assert doc["text"] == "Historic front page text of The Monitor in Omaha."

    # 3. Check metadata fields
    assert doc["newspaper_title"] == "The Monitor"
    assert doc["city"] == "Omaha"
    assert doc["state"] == "Nebraska"
    assert doc["year"] == 1915
    assert doc["month"] == 7
    assert doc["day"] == 3
    assert doc["sequence"] == 1
    assert doc["char_count"] == len("Historic front page text of The Monitor in Omaha.")
    assert doc["word_count"] == 9

    # 4. Check Pinecone compliance rules:
    # - No fields starting with _ other than _id
    # - No fields starting with $
    for key in doc:
        if key != "_id":
            assert not key.startswith("_"), f"Reserved key prefix '_' found: {key}"
            assert not key.startswith("$"), f"Reserved key prefix '$' found: {key}"

    # - Valid types: str, int, float, bool, list[str]
    for key, val in doc.items():
        assert val is not None, f"Null value not allowed for key {key}"
        assert isinstance(val, (str, int, float, bool, list)), f"Invalid type for key {key}: {type(val)}"
        if isinstance(val, list):
            assert all(isinstance(x, str) for x in val), f"List values must be strings: {key}"

    # 5. Check valid JSON serialization
    serialized = json.dumps(doc)
    parsed = json.loads(serialized)
    assert parsed["_id"] == doc["_id"]


def test_issue_record_to_pinecone_documents():
    page1 = PageRecord(
        sequence=1,
        title="Page 1",
        lccn="00225879",
        date="1915-07-03",
        edition=1,
    )
    page1._text_cache = "Text of page 1"

    page2 = PageRecord(
        sequence=2,
        title="Page 2",
        lccn="00225879",
        date="1915-07-03",
        edition=1,
    )
    page2._text_cache = "Text of page 2"

    issue = IssueRecord(
        title="The Monitor, July 3, 1915",
        newspaper_title="The Monitor",
        lccn="00225879",
        date="1915-07-03",
        edition=1,
        city="Omaha",
        state="Nebraska",
        batch_name="nbu_indescribablebeast_ver01",
        bulk_ocr_url="https://example.com/batch.tar.bz2",
        raw_batch_url="https://example.com/batch/",
        loc_item_url="https://www.loc.gov/item/00225879/1915-07-03/ed-1/",
        pages=[page1, page2],
    )

    docs = issue.to_pinecone_documents(text_field="text")
    assert len(docs) == 2
    assert docs[0]["_id"] == "00225879_1915-07-03_ed-1_seq-1"
    assert docs[1]["_id"] == "00225879_1915-07-03_ed-1_seq-2"
    assert docs[0]["newspaper_title"] == "The Monitor"
    assert docs[0]["batch_name"] == "nbu_indescribablebeast_ver01"
    assert docs[0]["page_count"] == 2
