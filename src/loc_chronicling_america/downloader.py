"""HTTP client and file downloader with streaming, checksum verification, and rate limiting."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Optional
import httpx
from tqdm import tqdm


DEFAULT_USER_AGENT = "loc-chronicling-america/0.1.0 (Research Tool; Python/3.14; curl/8.7.1)"


class Downloader:
    """Manages HTTP requests, streaming downloads, and file integrity verification."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        rate_limit_delay: float = 0.1,  # Delay between requests in seconds
        max_retries: int = 3,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time = 0.0

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
        }
        timeouts = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)
        self.client = httpx.Client(
            headers=headers,
            timeout=timeouts,
            follow_redirects=True,
            http2=False,  # httpcore h11 is rock-solid for LoC chunked endpoints
        )

    def _wait_for_rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, **kwargs) -> httpx.Response:
        """Execute a GET request with rate limiting and retry logic."""
        self._wait_for_rate_limit()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.get(url, **kwargs)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts: {last_error}") from last_error

    def fetch_text(self, url: str) -> str:
        """Fetch content as decoded UTF-8 string."""
        resp = self.get(url)
        return resp.text

    def fetch_json(self, url: str) -> dict:
        """Fetch content and parse as JSON."""
        resp = self.get(url)
        return resp.json()

    def download_file(
        self,
        url: str,
        dest_path: Path | str,
        expected_sha256: Optional[str] = None,
        expected_md5: Optional[str] = None,
        show_progress: bool = True,
        description: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Stream download a file with progress bar and optional checksum verification.

        Args:
            url: Remote file URL.
            dest_path: Local destination path.
            expected_sha256: Optional SHA-256 hash to verify against.
            expected_md5: Optional MD5 hash to verify against.
            show_progress: Whether to show a tqdm progress bar.
            description: Progress bar description.
            progress_callback: Optional callback receiving (bytes_read, total_bytes).

        Returns:
            Path to downloaded file.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        desc = description or dest.name
        temp_dest = dest.with_suffix(dest.suffix + ".part")

        last_err = None
        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()
            try:
                with self.client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total_size = int(resp.headers.get("content-length", 0))

                    sha256_hash = hashlib.sha256() if expected_sha256 else None
                    md5_hash = hashlib.md5() if expected_md5 else None
                    bytes_read = 0

                    with open(temp_dest, "wb") as f:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=desc,
                            disable=not show_progress,
                        ) as pbar:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    bytes_read += len(chunk)
                                    if sha256_hash:
                                        sha256_hash.update(chunk)
                                    if md5_hash:
                                        md5_hash.update(chunk)
                                    pbar.update(len(chunk))
                                    if progress_callback:
                                        progress_callback(bytes_read, total_size)

                # Verify Checksum
                if expected_sha256:
                    computed_sha256 = sha256_hash.hexdigest().lower()
                    if computed_sha256 != expected_sha256.lower():
                        temp_dest.unlink(missing_ok=True)
                        raise ValueError(
                            f"SHA-256 checksum mismatch for {dest.name}!\n"
                            f"Expected: {expected_sha256}\n"
                            f"Computed: {computed_sha256}"
                        )

                if expected_md5:
                    computed_md5 = md5_hash.hexdigest().lower()
                    if computed_md5 != expected_md5.lower():
                        temp_dest.unlink(missing_ok=True)
                        raise ValueError(
                            f"MD5 checksum mismatch for {dest.name}!\n"
                            f"Expected: {expected_md5}\n"
                            f"Computed: {computed_md5}"
                        )

                temp_dest.replace(dest)
                return dest

            except Exception as e:
                temp_dest.unlink(missing_ok=True)
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to download {url} after {self.max_retries} attempts: {last_err}") from last_err

    def close(self) -> None:
        """Close client sessions."""
        self.client.close()

    def __enter__(self) -> Downloader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


_DEFAULT_DOWNLOADER: Optional[Downloader] = None


def get_default_downloader() -> Downloader:
    """Return a shared singleton Downloader instance."""
    global _DEFAULT_DOWNLOADER
    if _DEFAULT_DOWNLOADER is None:
        _DEFAULT_DOWNLOADER = Downloader()
    return _DEFAULT_DOWNLOADER
