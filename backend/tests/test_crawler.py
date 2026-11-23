"""Tests for safe link discovery without live network requests."""

from unittest.mock import Mock

import pytest

from backend.services.crawler import CrawlError, SafeCrawler


def make_response(status_code, *, text="", headers=None, content=None):
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.headers = headers or {}
    response.content = content if content is not None else text.encode("utf-8")
    response.iter_content.side_effect = lambda chunk_size: iter(
        [response.content] if response.content else []
    )
    return response


def test_discover_links_keeps_only_approved_https_host(monkeypatch):
    responses = [
        make_response(200, text="User-agent: *\nAllow: /"),
        make_response(
            200,
            text=(
                "<html><title>Notices</title><body>"
                '<a href="/notices/new">New notice</a>'
                '<a href="http://bkbck.edu.in/insecure">HTTP</a>'
                '<a href="https://example.com/outside">External</a>'
                "</body></html>"
            ),
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    ]
    monkeypatch.setattr("backend.services.crawler.requests.get", lambda *a, **k: responses.pop(0))

    page = SafeCrawler("bkbck.edu.in", delay_seconds=0).discover_links(
        "https://bkbck.edu.in/"
    )

    assert page.title == "Notices"
    assert page.links == ["https://bkbck.edu.in/notices/new"]


def test_discover_links_honors_robots_disallow(monkeypatch):
    responses = [
        make_response(200, text="User-agent: *\nDisallow: /private"),
    ]
    monkeypatch.setattr("backend.services.crawler.requests.get", lambda *a, **k: responses.pop(0))

    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)

    with pytest.raises(CrawlError, match="robots.txt"):
        crawler.discover_links("https://bkbck.edu.in/private")


def test_missing_robots_file_allows_discovery(monkeypatch):
    responses = [
        make_response(404),
        make_response(
            200,
            text='<a href="/notices/">Notices</a>',
            headers={"Content-Type": "text/html"},
        ),
    ]
    monkeypatch.setattr("backend.services.crawler.requests.get", lambda *a, **k: responses.pop(0))

    page = SafeCrawler("bkbck.edu.in", delay_seconds=0).discover_links(
        "https://bkbck.edu.in/"
    )

    assert page.links == ["https://bkbck.edu.in/notices/"]


def test_redirect_outside_approved_host_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "backend.services.crawler.requests.get",
        lambda *a, **k: make_response(
            302,
            headers={"Location": "https://example.com/"},
        ),
    )
    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)

    with pytest.raises(CrawlError, match="approved college domain"):
        crawler.discover_links("https://bkbck.edu.in/")


def test_input_url_outside_approved_host_is_rejected():
    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)

    with pytest.raises(CrawlError, match="approved college domain"):
        crawler.discover_links("https://example.com/")


@pytest.mark.parametrize(
    "url",
    [
        "https://bkbck.edu.in:8443/",
        "https://bkbck.edu.in:not-a-port/",
    ],
)
def test_input_url_with_unapproved_port_is_rejected(url):
    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)

    with pytest.raises(CrawlError):
        crawler.discover_links(url)


def test_fetch_resource_rejects_oversized_content_length(monkeypatch):
    responses = [
        make_response(200, text="User-agent: *\nAllow: /"),
        make_response(
            200,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": "11",
            },
            content=b"%PDF-oversized",
        ),
    ]
    monkeypatch.setattr(
        "backend.services.crawler.requests.get",
        lambda *args, **kwargs: responses.pop(0),
    )
    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)

    with pytest.raises(CrawlError, match="size limit"):
        crawler.fetch_resource("https://bkbck.edu.in/notice.pdf", max_bytes=10)


def test_fetch_resource_rejects_oversized_stream_without_content_length(monkeypatch):
    response = make_response(
        200,
        headers={"Content-Type": "application/pdf"},
        content=b"%PDF-123456789",
    )
    monkeypatch.setattr(
        "backend.services.crawler.requests.get",
        lambda *args, **kwargs: response,
    )
    crawler = SafeCrawler("bkbck.edu.in", delay_seconds=0)
    monkeypatch.setattr(
        crawler,
        "_load_robots_policy",
        lambda: setattr(
            crawler,
            "_robots_parser",
            type("AllowAll", (), {"can_fetch": lambda *args: True})(),
        ),
    )

    with pytest.raises(CrawlError, match="size limit"):
        crawler.fetch_resource("https://bkbck.edu.in/notice.pdf", max_bytes=10)
