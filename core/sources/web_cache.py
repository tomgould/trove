"""Crawl cache for incremental mode — tracks ETags, Last-Modified, and content hashes.

Ported from ScrapePy/scrapepy/output/cache.py — unchanged core logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CACHE_FILENAME = ".trove_crawl_cache.json"
CACHE_VERSION = 1


def _get_header(headers: dict, name: str) -> str:
    """Case-insensitive header lookup. Returns empty string if not found."""
    val = headers.get(name.lower())
    if val:
        return val
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower:
            return v
    return ""


class CrawlCache:
    """Persists page metadata to skip unchanged resources on re-crawl."""

    def __init__(self, cache_dir: str):
        self.cache_path = os.path.join(cache_dir, CACHE_FILENAME)
        self._data: dict = {
            "version": CACHE_VERSION,
            "pages": {},
        }

    def load(self) -> None:
        """Load cache from disk."""
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == CACHE_VERSION:
                self._data = data
            else:
                logger.info(
                    "Cache version mismatch (expected %d, got %s), starting fresh",
                    CACHE_VERSION,
                    data.get("version"),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read cache (%s), starting fresh", e)

    def save(self) -> None:
        """Write cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, separators=(",", ":"))
        except (OSError, TypeError) as e:
            logger.warning("Could not save cache: %s", e)

    def get_page_headers(self, url_path: str) -> dict:
        """Return conditional request headers for a cached page, or empty dict."""
        entry = self._data["pages"].get(url_path)
        if not entry:
            return {}
        headers = {}
        if entry.get("etag"):
            headers["If-None-Match"] = entry["etag"]
        if entry.get("last_modified"):
            headers["If-Modified-Since"] = entry["last_modified"]
        return headers

    def update_page(
        self,
        url_path: str,
        headers: dict,
        content: str,
        links: list[str] | None = None,
    ) -> None:
        """Store cache entry for a page using response headers + content hash."""
        self._data["pages"][url_path] = {
            "etag": _get_header(headers, "etag"),
            "last_modified": _get_header(headers, "last-modified"),
            "content_hash": f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}",
            "links": links or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_page_links(self, url_path: str) -> list[str]:
        """Return cached links for a page, or empty list."""
        entry = self._data["pages"].get(url_path)
        if not entry:
            return []
        return entry.get("links", [])

    def is_page_unchanged(self, url_path: str, content: str) -> bool:
        """Check if page content matches the cached hash."""
        entry = self._data["pages"].get(url_path)
        if not entry or not entry.get("content_hash"):
            return False
        current_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
        return current_hash == entry["content_hash"]

    @property
    def page_count(self) -> int:
        return len(self._data["pages"])
