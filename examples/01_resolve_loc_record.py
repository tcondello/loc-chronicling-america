#!/usr/bin/env python3
"""Example 1: Resolving a Library of Congress Chronicling America record.

Demonstrates how to take an LoC newspaper URL (e.g. from chroniclingamerica.loc.gov or loc.gov)
and instantly resolve its publication details, pages, and exact source batch.
"""

from loc_chronicling_america import ChroniclingAmerica

def main():
    client = ChroniclingAmerica()

    # User's example record: The Monitor (Omaha, Neb.), July 3, 1915 (Inaugural issue)
    url = "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/?st=gallery"
    print(f"Resolving LoC URL:\n  {url}\n")

    record = client.resolve(url)

    print("=" * 60)
    print(f"Publication:       {record.newspaper_title}")
    print(f"Issue Title:       {record.title}")
    print(f"Location:          {record.place_of_publication}")
    print(f"LCCN:              {record.lccn}")
    print(f"Publication Date:  {record.date} (Edition {record.edition})")
    print(f"Total Pages:       {record.page_count}")
    print("=" * 60)
    print(f"SOURCE BATCH:      {record.batch_name}")
    print(f"Bulk Archive (.tar.bz2): {record.bulk_ocr_url}")
    print(f"Raw BagIt Batch Dir:     {record.raw_batch_url}")
    print("=" * 60)

    print("\nPage Assets:")
    for page in record.pages:
        print(f"  Page {page.sequence} ({page.width}x{page.height}px):")
        print(f"    - PDF:   {page.pdf_url}")
        print(f"    - ALTO:  {page.alto_xml_url}")
        print(f"    - Image: {page.iiif_image_url(pct=100)}")

if __name__ == "__main__":
    main()
