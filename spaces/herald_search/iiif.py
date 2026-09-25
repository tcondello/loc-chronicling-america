"""Deterministic Library of Congress JP2 to IIIF JPEG URL Resolver.

Maps raw NDNP JPEG 2000 (.jp2) master scan URLs into live, CDN-backed
IIIF Image API endpoints on tile.loc.gov, delivering fast, browser-ready
JPEGs without local image decoding or file conversion.
"""

import re
from typing import Optional


def extract_iiif_identifier(jp2_url: str) -> Optional[str]:
    """Extract standard LoC IIIF service identifier from a JP2 URL.

    Handles formats:
    - https://chroniclingamerica.loc.gov/data/batches/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    - https://tile.loc.gov/storage-services/service/ndnp/{awardee}/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    """
    if not jp2_url or not isinstance(jp2_url, str):
        return None

    # Format 1: chroniclingamerica.loc.gov/data/batches/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    m1 = re.search(
        r"batches/([^/]+)/data/([^/]+)/(?:([^/]+)/)?([^/]+)/(\d+)\.jp2",
        jp2_url,
    )
    if m1:
        batch, lccn, reel, date_ed, frame = m1.groups()
        reel = reel or "00000000000"
        awardee = batch.split("_")[0]
        batch_folder = batch if batch.startswith("batch_") else f"batch_{batch}"
        return f"service:ndnp:{awardee}:{batch_folder}:data:{lccn}:{reel}:{date_ed}:{frame}"

    # Format 2: tile.loc.gov/storage-services/service/ndnp/{awardee}/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    m2 = re.search(
        r"ndnp/([^/]+)/([^/]+)/data/([^/]+)/([^/]+)/([^/]+)/(\d+)\.jp2",
        jp2_url,
    )
    if m2:
        awardee, batch, lccn, reel, date_ed, frame = m2.groups()
        batch_folder = batch if batch.startswith("batch_") else f"batch_{batch}"
        return f"service:ndnp:{awardee}:{batch_folder}:data:{lccn}:{reel}:{date_ed}:{frame}"

    return None


def jp2_to_iiif_url(
    jp2_url: str,
    region: str = "full",
    size: str = "600,",
    rotation: int = 0,
    quality: str = "default",
    format_ext: str = "jpg",
) -> str:
    """Transform an LoC JP2 master scan URL into a web-ready IIIF JPEG URL.

    Args:
        jp2_url: Raw JP2 link from dataset.
        region: 'full', 'x,y,w,h' pixel box, or 'pct:x,y,w,h'.
        size: '600,' (width 600px, auto aspect), 'pct:25', or 'full'.
        rotation: Degrees of rotation (0, 90, 180, 270).
        quality: Image quality ('default', 'gray', 'bitonal').
        format_ext: File format extension ('jpg', 'png').
    """
    iiif_id = extract_iiif_identifier(jp2_url)
    if not iiif_id:
        return jp2_url
    return f"https://tile.loc.gov/image-services/iiif/{iiif_id}/{region}/{size}/{rotation}/{quality}.{format_ext}"


def jp2_to_iiif_info(jp2_url: str) -> Optional[str]:
    """Return the IIIF info.json endpoint for deep-zoom viewers (e.g. OpenSeadragon)."""
    iiif_id = extract_iiif_identifier(jp2_url)
    if not iiif_id:
        return None
    return f"https://tile.loc.gov/image-services/iiif/{iiif_id}/info.json"


def jp2_to_iiif_region(
    jp2_url: str,
    x: int,
    y: int,
    w: int,
    h: int,
    size: str = "800,",
) -> str:
    """Return a cropped rectangular section from the high-res page scan."""
    region = f"{int(x)},{int(y)},{int(w)},{int(h)}"
    return jp2_to_iiif_url(jp2_url, region=region, size=size)
