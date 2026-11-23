"""Bounded discovery of approved HTML pages and linked PDF documents."""

from collections import deque
from dataclasses import dataclass
from urllib.parse import urlsplit

from backend.services.crawler import (
    CrawledPage,
    SafeCrawler,
    normalize_allowed_url,
)


@dataclass(frozen=True)
class DiscoveredCollection:
    """Pages and PDF links found during a bounded source discovery run."""

    pages: list[CrawledPage]
    pdf_urls: list[str]
    truncated: bool


def collect_source_links(
    source_url: str,
    crawler: SafeCrawler,
    *,
    max_depth: int = 1,
    max_pages: int = 100,
) -> DiscoveredCollection:
    """Discover same-domain pages and PDF links without downloading PDFs."""
    if max_depth < 0:
        raise ValueError("Maximum crawl depth cannot be negative.")
    if max_pages < 1:
        raise ValueError("Maximum page count must be at least one.")

    source_url = normalize_allowed_url(source_url, crawler.allowed_domain)
    pending = deque([(source_url, 0)])
    queued_urls = {source_url}
    visited_urls: set[str] = set()
    discovered_pages: list[CrawledPage] = []
    pdf_urls: set[str] = set()

    while pending and len(discovered_pages) < max_pages:
        current_url, depth = pending.popleft()
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)

        page = crawler.discover_links(current_url)
        discovered_pages.append(page)

        for link in page.links:
            if _is_pdf_url(link):
                pdf_urls.add(link)
            elif depth < max_depth and link not in queued_urls:
                pending.append((link, depth + 1))
                queued_urls.add(link)

    return DiscoveredCollection(
        pages=discovered_pages,
        pdf_urls=sorted(pdf_urls),
        truncated=bool(pending),
    )


def _is_pdf_url(url: str) -> bool:
    """Identify links whose URL path ends in a PDF extension."""
    return urlsplit(url).path.lower().endswith(".pdf")
