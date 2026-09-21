#!/usr/bin/env python3
"""Example 3: Searching newspaper titles and batches using the embedded SQLite catalog."""

from loc_chronicling_america import ChroniclingAmerica

def main():
    client = ChroniclingAmerica()

    # 1. Search for a newspaper by name and state
    print("Searching for 'The Monitor' in Nebraska...")
    titles = client.search_titles("The Monitor", state="Nebraska")
    for t in titles:
        print(f"  - {t.name} (LCCN: {t.lccn})")
        print(f"    Location: {t.city}, {t.state} | Years: {t.formatted_years} | Total Issues: {t.issue_count}")
        print(f"    Demographic: {t.ethnicity or 'General'}")

    # 2. Search for Nebraska batches
    print("\nSearching for batches from Nebraska (awardee 'nbu')...")
    batches = client.search_batches(state="NE", limit=5)
    for b in batches:
        print(f"  - Batch: {b.name}")
        print(f"    Size: {b.size_mb} MB | Ingested: {b.ingested_date}")
        print(f"    Bulk Download: {b.archive_url}")

if __name__ == "__main__":
    main()
