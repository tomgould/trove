"""HTML to Markdown conversion for web source connector.

Ported from ScrapePy/scrapepy/convert/to_markdown.py — simplified to
content extraction and junk removal. Drops: file I/O, batch processing,
section indexes, asset copying, platform-specific handlers.
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
import markdownify

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# CSS class keywords for junk element removal
_JUNK_CLASS_KEYWORDS = (
    "advert", "ad-slot", "ad-container", "ad-wrapper", "dfp-ad",
    "google-ad", "gpt-ad", "ad-unit", "ad-banner", "ad-break",
    "newsletter", "signup", "sign-up", "subscribe", "email-capture",
    "most-read", "most_read", "mostread", "trending", "top-stories",
    "popular-stories", "recommended", "related-articles", "related-stories",
    "more-stories", "more-news", "commercial-slot", "commercial-container",
    "commercial-wrapper", "sponsorship",
    "social-share", "share-buttons", "social-links",
    "cookie-banner", "cookie-consent", "consent-banner",
)

# Junk heading sections (regex removes entire section)
_JUNK_SECTION_RE = re.compile(
    r"^#{2,6}\s+(?:"
    r"Most Read|Subscribe\b|Sign up\b|In this article|More (?:News|Stories)|"
    r"Related (?:Articles?|Stories|Topics)|Trending|Popular|Recommended|"
    r"You May (?:Also )?Like|Also Read|Don't Miss|"
    r"Comments?\s*(?:\d+\s*)?$|Add (?:your |a )?comment"
    r").*?(?=^#{1,3}\s|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)

# Content selectors in priority order
_CONTENT_SELECTORS = [
    ".entry-content",
    ".post-content",
    ".article-text",
    ".article-body",
    "article .content",
    "[role='main'] article",
    "#content article",
    "article",
    "[role='main']",
    "main",
    "#main",
    "#content",
    ".content",
]


def _remove_junk_elements(soup: BeautifulSoup) -> None:
    """Remove non-content HTML elements before extraction."""
    for tag in soup.find_all(["script", "style", "noscript", "svg", "link"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "header", "footer"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()

    # Sidebars
    for tag in soup.find_all(["aside", "section"]):
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        if any(
            kw in classes.lower() or kw in tag_id.lower()
            for kw in ("sidebar", "widget", "secondary", "nav", "menu")
        ):
            tag.decompose()

    # Ad/newsletter/trending containers by CSS class
    for tag in list(soup.find_all(True)):
        if not tag.parent:
            continue
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        combined = (classes + " " + tag_id).lower()
        if any(kw in combined for kw in _JUNK_CLASS_KEYWORDS):
            tag.decompose()


def html_to_markdown(html: str) -> str | None:
    """Convert HTML to clean Markdown.

    Tries content selectors, then trafilatura fallback, then generic extraction.
    Returns None if no meaningful content found.
    """
    soup = BeautifulSoup(html, "lxml")
    _remove_junk_elements(soup)

    # Try content selectors
    content = None
    for sel in _CONTENT_SELECTORS:
        content = soup.select_one(sel)
        if content and len(content.get_text(strip=True).split()) > 50:
            break
    else:
        content = None

    if content:
        title = ""
        h1 = content.find("h1") or soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.find("title"):
            title = soup.find("title").get_text(strip=True)

        md = markdownify.markdownify(str(content), heading_style="ATX", bullets="-")
        if title and not md.strip().startswith("# "):
            md = f"# {title}\n\n{md}"

        md = clean_markdown(md)
        if md and len(md.strip()) > 100:
            return md

    # Fallback: trafilatura
    if HAS_TRAFILATURA:
        try:
            extracted = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                include_formatting=True,
                output_format="markdown",
            )
            if extracted and len(extracted.strip()) > 100:
                return clean_markdown(extracted)
        except (MemoryError, ImportError):
            raise
        except Exception as e:
            logger.warning("Trafilatura extraction failed, falling back to generic: %s", e)

    # Fallback: generic body extraction
    soup2 = BeautifulSoup(html, "lxml")
    for tag in soup2.find_all(
        ["script", "style", "nav", "header", "footer", "aside", "noscript"]
    ):
        tag.decompose()

    main = soup2.select_one(
        "main, article, [role='main'], #main, .content, #content, body"
    )
    if main:
        title = ""
        if soup2.find("title"):
            title = soup2.find("title").get_text().strip()
        if soup2.find("h1"):
            title = soup2.find("h1").get_text().strip()

        md = markdownify.markdownify(str(main), heading_style="ATX", bullets="-")
        if title and not md.strip().startswith("# "):
            md = f"# {title}\n\n{md}"

        md = clean_markdown(md)
        if md and len(md.strip()) > 100:
            return md

    return None


def extract_title(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)
    return ""


def clean_markdown(md: str) -> str | None:
    """Remove junk content from markdown text."""
    if not md:
        return None

    # Advertisement markers
    md = re.sub(
        r"^\s*Article continues below\s*(?:ADVERTISEMENT)?\s*$",
        "", md, flags=re.MULTILINE,
    )
    md = re.sub(
        r"^\s*Advertisement\s*[-\u2013\u2014]\s*Continue Reading Below\s*$",
        "", md, flags=re.MULTILINE | re.IGNORECASE,
    )
    md = re.sub(r"^\s*Advertisement\s*$", "", md, flags=re.MULTILINE)

    # Relative date strings
    md = re.sub(
        r"^\s*(?:just now|yesterday|"
        r"(?:Updated\s+)?(?:\d+|an?|one)\s+"
        r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago"
        r")\s*$",
        "", md, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Newsletter/subscribe standalone lines
    md = re.sub(
        r"^\s*(?:Sign up|Subscribe|Follow|Leave this field blank)\s*$",
        "", md, flags=re.MULTILINE,
    )

    # Junk heading sections
    md = _JUNK_SECTION_RE.sub("", md)

    # Empty links and images
    md = re.sub(r"\[]\([^)]*\)", "", md)
    md = re.sub(r"!\[]\([^)]*\)", "", md)

    # Collapse excessive blank lines
    md = re.sub(r"\n{4,}", "\n\n\n", md)

    return md.strip() if md.strip() else None
