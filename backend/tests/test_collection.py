"""Tests for bounded, mocked source link collection."""

from unittest.mock import Mock

import pytest

from backend.services.collection import collect_source_links
from backend.services.crawler import CrawledPage


def make_page(url, *links):
    return CrawledPage(url=url, title=url, links=list(links))


def make_crawler(pages):
    pages_by_url = dict(pages)
    crawler = Mock()
    crawler.allowed_domain = "bkbck.edu.in"
    crawler.discover_links.side_effect = lambda url: pages_by_url[url]
    return crawler


def test_collection_discovers_pages_and_pdf_links_within_depth():
    crawler = make_crawler(
        [
            (
                "https://bkbck.edu.in/",
                make_page(
                    "https://bkbck.edu.in/",
                    "https://bkbck.edu.in/notices/",
                    "https://bkbck.edu.in/files/notice.PDF?download=1",
                ),
            ),
            (
                "https://bkbck.edu.in/notices/",
                make_page(
                    "https://bkbck.edu.in/notices/",
                    "https://bkbck.edu.in/notices/archive/",
                ),
            ),
        ]
    )

    collection = collect_source_links(
        "https://bkbck.edu.in/", crawler, max_depth=1
    )

    assert [page.url for page in collection.pages] == [
        "https://bkbck.edu.in/",
        "https://bkbck.edu.in/notices/",
    ]
    assert collection.pdf_urls == [
        "https://bkbck.edu.in/files/notice.PDF?download=1"
    ]
    assert collection.truncated is False
    assert crawler.discover_links.call_count == 2


def test_collection_does_not_repeat_pages_in_a_cycle():
    crawler = make_crawler(
        [
            (
                "https://bkbck.edu.in/",
                make_page("https://bkbck.edu.in/", "https://bkbck.edu.in/notices/"),
            ),
            (
                "https://bkbck.edu.in/notices/",
                make_page("https://bkbck.edu.in/notices/", "https://bkbck.edu.in/"),
            ),
        ]
    )

    collection = collect_source_links(
        "https://bkbck.edu.in/", crawler, max_depth=5
    )

    assert len(collection.pages) == 2
    assert crawler.discover_links.call_count == 2


def test_collection_normalizes_seed_url_before_tracking_cycles():
    crawler = make_crawler(
        [
            (
                "https://bkbck.edu.in/",
                make_page("https://bkbck.edu.in/", "https://bkbck.edu.in/"),
            ),
        ]
    )

    collection = collect_source_links(
        "https://bkbck.edu.in/#home", crawler, max_depth=3
    )

    assert len(collection.pages) == 1
    assert crawler.discover_links.call_args.args == ("https://bkbck.edu.in/",)


def test_collection_reports_when_page_limit_truncates_queue():
    crawler = make_crawler(
        [
            (
                "https://bkbck.edu.in/",
                make_page(
                    "https://bkbck.edu.in/",
                    "https://bkbck.edu.in/notices/",
                    "https://bkbck.edu.in/exams/",
                ),
            ),
            (
                "https://bkbck.edu.in/notices/",
                make_page("https://bkbck.edu.in/notices/"),
            ),
        ]
    )

    collection = collect_source_links(
        "https://bkbck.edu.in/", crawler, max_depth=2, max_pages=2
    )

    assert len(collection.pages) == 2
    assert collection.truncated is True


@pytest.mark.parametrize(
    ("max_depth", "max_pages"),
    [(-1, 1), (0, 0)],
)
def test_collection_rejects_invalid_limits(max_depth, max_pages):
    with pytest.raises(ValueError):
        collect_source_links(
            "https://bkbck.edu.in/",
            Mock(),
            max_depth=max_depth,
            max_pages=max_pages,
        )
