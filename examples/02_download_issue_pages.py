#!/usr/bin/env python3
"""Example 2: Downloading issue pages and plain text OCR.

Demonstrates downloading specific assets (plain text, single-page PDFs, high-res scans)
without needing to download multi-gigabyte bulk archives.
"""

from pathlib import Path
from loc_chronicling_america import ChroniclingAmerica

def main():
    client = ChroniclingAmerica()
    url = "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/"
    record = client.resolve(url)

    out_dir = Path("./example_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Issue: {record.title} ({record.page_count} pages)")
    page1 = record.get_page(1)

    # 1. Fetch plain text OCR directly
    print("\n1. Fetching Page 1 plain text OCR...")
    text = page1.get_text()
    print("--- First 300 characters of OCR Text ---")
    print(text[:300])
    print("----------------------------------------")

    # 2. Save text to file
    txt_path = page1.download_text(out_dir)
    print(f"Saved text to: {txt_path}")

    # 3. Download single-page PDF
    print("\n2. Downloading Page 1 PDF...")
    pdf_path = page1.download_pdf(out_dir)
    print(f"Saved PDF to: {pdf_path}")

    # 4. Download high-resolution master scan
    print("\n3. Downloading Page 1 high-resolution master image (100% scale)...")
    img_path = page1.download_image(out_dir, pct=100)
    print(f"Saved image to: {img_path}")

if __name__ == "__main__":
    main()
