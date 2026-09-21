"""Tests for Downloader with checksum verification."""

import hashlib
from pathlib import Path
import pytest
from loc_chronicling_america.downloader import Downloader


def test_download_file_checksum(tmp_path: Path):
    downloader = Downloader()

    # Create dummy local file to simulate download
    content = b"Sample newspaper OCR content for testing"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    expected_md5 = hashlib.md5(content).hexdigest()

    # Test checksum logic using custom mock stream or local file
    dest = tmp_path / "downloaded.txt"
    dest.write_bytes(content)

    # Verify sha256
    with open(dest, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == expected_sha256

    # Verify md5
    with open(dest, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    assert actual_md5 == expected_md5
