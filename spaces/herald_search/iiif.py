"""Deterministic Library of Congress JP2 to IIIF JPEG URL Resolver.

Maps raw NDNP JPEG 2000 (.jp2) master scan URLs into live, CDN-backed
IIIF Image API endpoints on tile.loc.gov, delivering fast, browser-ready
JPEGs without local image decoding or file conversion.
"""

import re
from typing import Optional


def extract_iiif_identifier(jp2_url: str) -> Optional[str]:
    """Extract standard LoC IIIF service identifier from a JP2 URL or IIIF URL.

    Handles formats:
    - https://chroniclingamerica.loc.gov/data/batches/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    - https://chroniclingamerica.loc.gov/data/batches/{batch}/data/{lccn}/{issue_dir}/{frame}.jp2 (missing reel)
    - https://tile.loc.gov/storage-services/service/ndnp/{awardee}/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    - https://tile.loc.gov/image-services/iiif/service:ndnp:...
    """
    if not jp2_url or not isinstance(jp2_url, str):
        return None

    # Handle if already a IIIF URL with service:ndnp:...
    if "service:ndnp:" in jp2_url:
        m_svc = re.search(r"(service:ndnp:[^/]+)", jp2_url)
        if m_svc:
            svc_id = m_svc.group(1)
            # Fix batch_nn_hardin_ver01 placeholder if present
            if "batch_nn_hardin_ver01" in svc_id:
                m_frame = re.search(r"(191012\d{2}\d{2}):(\d{4})", svc_id)
                if m_frame:
                    date_ed, frame_str = m_frame.groups()
                    p = int(frame_str)
                    if date_ed.startswith("19101201"):
                        real_frame = f"{7 + p:04d}" if p <= 20 else frame_str
                    elif date_ed.startswith("19101202"):
                        real_frame = f"{27 + p:04d}" if p <= 24 else frame_str
                    elif date_ed.startswith("19101203"):
                        real_frame = f"{51 + p:04d}" if p <= 12 else frame_str
                    else:
                        real_frame = frame_str
                    svc_id = re.sub(
                        r"data:sn83030193:(?:00000000000:)?(191012\d{4}):\d{4}",
                        f"data:sn83030193:0028076582A:\\1:{real_frame}",
                        svc_id,
                    )
            return svc_id

    # Format 1: chroniclingamerica.loc.gov/data/batches/{batch}/data/{lccn}/{reel}/{issue_dir}/{frame}.jp2
    m1 = re.search(
        r"batches/([^/]+)/data/([^/]+)/(?:([^/]+)/)?([^/]+)/(\d+)\.jp2",
        jp2_url,
    )
    if m1:
        batch, lccn, reel, date_ed, frame = m1.groups()
        # Special case: The Evening World batch_nn_hardin_ver01 (Dec 1910)
        # where reel is omitted in upstream Parquet and frames are consecutive on reel 0028076582A
        if batch in ("nn_hardin_ver01", "batch_nn_hardin_ver01") and lccn == "sn83030193":
            reel = "0028076582A"
            p = int(frame)
            if date_ed.startswith("19101201"):
                frame = f"{7 + p:04d}" if p <= 20 else frame
            elif date_ed.startswith("19101202"):
                frame = f"{27 + p:04d}" if p <= 24 else frame
            elif date_ed.startswith("19101203"):
                frame = f"{51 + p:04d}" if p <= 12 else frame
        else:
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
    """Transform an LoC JP2 master scan URL into a web-ready IIIF JPEG URL."""
    iiif_id = extract_iiif_identifier(jp2_url)
    if not iiif_id:
        return jp2_url
    return f"https://tile.loc.gov/image-services/iiif/{iiif_id}/{region}/{size}/{rotation}/{quality}.{format_ext}"


def normalize_iiif_url(url_or_jp2: str, size: str = "600,") -> str:
    """Guarantee a valid, working LoC IIIF endpoint from any JP2 or legacy IIIF URL."""
    if not url_or_jp2 or not isinstance(url_or_jp2, str):
        return ""
    iiif_id = extract_iiif_identifier(url_or_jp2)
    if not iiif_id:
        return url_or_jp2
    return f"https://tile.loc.gov/image-services/iiif/{iiif_id}/full/{size}/0/default.jpg"


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
