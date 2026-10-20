"""Web source connector for Trove.

Crawls websites and converts pages to Markdown for Trove indexing.
Supports incremental crawling via ETags/content hashing.

Ported from ScrapePy — simplified to synchronous httpx-based crawling
with BFS link following, scope control, and junk content filtering.
"""

from __future__ import annotations

import logging
import posixpath
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from core.sources.base import BaseSourceConnector, SyncResult
from core.sources.web_cache import CrawlCache
from core.sources.web_converter import extract_title, html_to_markdown

logger = logging.getLogger(__name__)

# File extensions to skip when crawling
SKIP_EXTENSIONS = re.compile(
    r"\.(pdf|doc|docx|jpg|jpeg|png|gif|svg|webp|ico|css|js|woff|woff2|"
    r"ttf|eot|mp4|mp3|wav|webm|zip|tar|gz|xml|json|txt)$",
    re.IGNORECASE,
)

# Non-content paths to skip
NON_CONTENT_PATHS = re.compile(
    r"/(?:login|log-in|signin|sign-in|signup|sign-up|register|logout)(?:/|$)"
    r"|/(?:settings|account|preferences|notifications)(?:/|$)"
    r"|/(?:cart|checkout|wishlist|order|orders)(?:/|$)"
    r"|/(?:admin|wp-admin|wp-login|dashboard)(?:/|$)"
    r"|/(?:share|embed|print)(?:/|$)"
    r"|^/(?:author|authors|topic|topics|tag|tags)/"
    r"|/(?:rss|feed|atom)(?:\.xml)?/?$"
    r"|^/(?:page|p)/\d+"
    r"|^/cdn-cgi/"
    r"|/(?:contact|about|privacy|cookie|gdpr|accessibility)(?:-|/|$)"
    r"|/(?:subscribe|subscriptions|newsletter|advertise)(?:-|/|$)"
    r"|/(?:search|sitemap|archive)(?:-|/|$|\.)"
    r"|/(?:terms|conditions)(?:-|/|$)",
    re.IGNORECASE,
)


@dataclass
class WebSourceConfig:
    """Configuration for a web source in client.yaml."""

    url: str = ""
    mode: str = "crawl"  # crawl, docs, swagger
    stay_in_path: bool = True
    max_pages: int = 500
    wait_time: float = 1.0
    timeout: int = 30

    @classmethod
    def from_dict(cls, data: dict) -> WebSourceConfig:
        return cls(
            url=data.get("url", ""),
            mode=data.get("mode", "crawl"),
            stay_in_path=data.get("stay_in_path", True),
            max_pages=data.get("max_pages", 500),
            wait_time=data.get("wait_time", 1.0),
            timeout=data.get("timeout", 30),
        )

    @property
    def domain(self) -> str:
        parsed = urlparse(self.url)
        return parsed.netloc.replace("www.", "")

    @property
    def base_url(self) -> str:
        parsed = urlparse(self.url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def starting_path(self) -> str:
        parsed = urlparse(self.url)
        return parsed.path or "/"


def normalize_link(link: str, current_path: str, base_url: str) -> str | None:
    """Normalize a link to an absolute path on the same domain.

    Returns None for external/skip links.
    """
    if not link:
        return None

    if link.startswith(("mailto:", "tel:", "javascript:", "data:", "blob:", "about:")):
        return None

    clean = link.split("?")[0].split("#")[0]
    if not clean:
        return None

    if SKIP_EXTENSIONS.search(clean):
        return None

    if clean.startswith(base_url):
        clean = clean[len(base_url):]
    elif clean.startswith("http://") or clean.startswith("https://"):
        return None  # external

    if not clean.startswith("/"):
        current_dir = current_path.rsplit("/", 1)[0] if "/" in current_path else "/"
        clean = posixpath.join(current_dir, clean)

    clean = posixpath.normpath(clean)
    if not clean.startswith("/"):
        clean = "/" + clean

    return clean


def _is_in_scope(url_path: str, config: WebSourceConfig) -> bool:
    """Check if a URL path is within the configured scope."""
    if not config.stay_in_path:
        return True
    starting = config.starting_path
    if not starting or starting == "/":
        return True
    normalised = url_path.split("?")[0].split("#")[0]
    return normalised == starting or normalised.startswith(starting + "/")


def _extract_links(html: str) -> list[str]:
    """Quick regex-based link extraction from HTML."""
    links = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
        links.add(m.group(1))
    return list(links)


def _url_to_filepath(url_path: str) -> str:
    """Convert a URL path to a safe filename."""
    path = url_path.strip("/")
    if not path:
        path = "index"
    # Replace path separators with underscores for flat storage
    path = path.replace("/", "_")
    # Remove unsafe characters
    path = re.sub(r'[<>:"|?*]', "_", path)
    if not path.endswith(".md"):
        path += ".md"
    return path


class WebSourceConnector(BaseSourceConnector):
    """Crawl a website and save pages as Markdown for Trove indexing."""

    def __init__(self, config: WebSourceConfig):
        self.config = config

    def sync(self, target_dir: str) -> SyncResult:
        """Crawl the website and save Markdown files to target_dir/web/{domain}/."""
        result = SyncResult(source_type="web")
        target = Path(target_dir)
        web_dir = target / "web" / self.config.domain
        web_dir.mkdir(parents=True, exist_ok=True)

        # Load cache for incremental crawling
        cache = CrawlCache(str(web_dir))
        cache.load()

        existing_files = set(f.name for f in web_dir.glob("*.md"))

        try:
            pages_crawled = self._crawl(web_dir, cache, result, existing_files)
        except (httpx.HTTPError, OSError) as e:
            result.errors.append(f"Crawl error: {e}")
            logger.exception("Error crawling %s", self.config.url)

        cache.save()
        return result

    def status(self) -> dict:
        return {
            "source_type": "web",
            "url": self.config.url,
            "domain": self.config.domain,
            "mode": self.config.mode,
            "stay_in_path": self.config.stay_in_path,
            "max_pages": self.config.max_pages,
        }

    def _crawl(
        self, output_dir: Path, cache: CrawlCache, result: SyncResult,
        existing_files: set[str] | None = None,
    ) -> int:
        """BFS crawl starting from config URL. Returns pages crawled."""
        from collections import deque

        visited: set[str] = set()
        queue: deque[str] = deque()
        pages_crawled = 0
        known_files = existing_files or set()

        # Seed with starting path
        starting_path = self.config.starting_path
        queue.append(starting_path)

        with httpx.Client(timeout=self.config.timeout, follow_redirects=True) as client:
            while queue and pages_crawled < self.config.max_pages:
                url_path = queue.popleft()

                if url_path in visited:
                    continue
                visited.add(url_path)

                # Skip non-content paths
                if NON_CONTENT_PATHS.search(url_path):
                    continue

                # Scope check
                if not _is_in_scope(url_path, self.config):
                    continue

                full_url = self.config.base_url + url_path

                # Fetch with conditional headers
                headers = cache.get_page_headers(url_path)
                try:
                    response = client.get(full_url, headers=headers)
                except httpx.HTTPError as e:
                    result.errors.append(f"{url_path}: {e}")
                    continue

                # 304 Not Modified — reuse cached links
                if response.status_code == 304:
                    for link in cache.get_page_links(url_path):
                        if link not in visited:
                            queue.append(link)
                    continue

                if response.status_code >= 400:
                    if response.status_code >= 500:
                        result.errors.append(
                            f"{url_path}: server error ({response.status_code})"
                        )
                    elif response.status_code == 429:
                        result.errors.append(f"{url_path}: rate limited (429)")
                        logger.warning("Rate limited while crawling %s", full_url)
                    else:
                        logger.debug(
                            "Skipping %s (HTTP %d)", url_path, response.status_code
                        )
                    continue

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue

                html = response.text

                # Check content hash for unchanged pages
                if cache.is_page_unchanged(url_path, html):
                    for link in cache.get_page_links(url_path):
                        if link not in visited:
                            queue.append(link)
                    continue

                # Extract and queue links
                raw_links = _extract_links(html)
                normalized_links = []
                for link in raw_links:
                    norm = normalize_link(link, url_path, self.config.base_url)
                    if norm and norm not in visited:
                        normalized_links.append(norm)
                        queue.append(norm)

                # Convert to Markdown
                md = html_to_markdown(html)
                if not md:
                    # Update cache even for pages with no content
                    cache.update_page(
                        url_path,
                        dict(response.headers),
                        html,
                        links=normalized_links,
                    )
                    continue

                # Add frontmatter
                title = extract_title(html)
                safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
                frontmatter = (
                    f"---\n"
                    f'title: "{safe_title}"\n'
                    f"url: {full_url}\n"
                    f"source: web_crawler\n"
                    f"---\n\n"
                )
                md_with_frontmatter = frontmatter + md

                # Write file
                filename = _url_to_filepath(url_path)
                filepath = output_dir / filename
                filepath.write_text(md_with_frontmatter, encoding="utf-8")

                # Update cache
                cache.update_page(
                    url_path,
                    dict(response.headers),
                    html,
                    links=normalized_links,
                )

                if filename in known_files:
                    result.files_updated += 1
                else:
                    result.files_added += 1
                    known_files.add(filename)
                pages_crawled += 1

                logger.info(
                    "Crawled %s (%d/%d)",
                    url_path,
                    pages_crawled,
                    self.config.max_pages,
                )

                # Politeness delay
                if self.config.wait_time > 0:
                    time.sleep(self.config.wait_time)

        return pages_crawled
