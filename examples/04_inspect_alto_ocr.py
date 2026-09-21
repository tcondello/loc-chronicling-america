#!/usr/bin/env python3
"""Example 4: Inspecting ALTO XML layout, word bounding boxes, and OCR confidence."""

from loc_chronicling_america import ChroniclingAmerica

def main():
    client = ChroniclingAmerica()
    url = "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/"
    record = client.resolve(url)

    page1 = record.get_page(1)
    print(f"Fetching ALTO XML layout for Page 1 of '{record.title}'...")

    alto = page1.get_alto()

    print(f"\nPage Layout Information:")
    print(f"  Page Dimensions: {alto.page_width} x {alto.page_height} ({alto.measurement_unit})")
    print(f"  Total Text Blocks: {len(alto.blocks)}")
    print(f"  Total Recognized Lines: {len(alto.lines)}")
    print(f"  Total Recognized Words: {len(alto.words)}")

    print("\nSample 15 Recognized Words with Coordinates and Confidence Scores:")
    for w in alto.words[:15]:
        conf_pct = f"{w.confidence * 100:.1f}%" if w.confidence is not None else "N/A"
        print(f"  Word: {w.content:<16} | Box (x,y,w,h): {str(w.box):<20} | Confidence: {conf_pct}")

if __name__ == "__main__":
    main()
