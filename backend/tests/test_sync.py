"""Tests for bounded, mocked source synchronization."""

from types import SimpleNamespace

import pytest

from backend.database.db import get_connection, initialize_database
from backend.database.models import (
    insert_source,
    list_chunks_for_embedding,
    list_documents,
)
from backend.services.crawler import CrawlError, CrawledPage, FetchedResource
from backend.services.embeddings import EmbeddingError
from backend.services.sync import sync_sources

ROOT_URL = "https://bkbck.edu.in/"
CHILD_URL = "https://bkbck.edu.in/notices/"


class FakeCrawler:
    def __init__(
        self,
        allowed_domain,
        *,
        timeout,
        delay_seconds,
        max_response_bytes,
        fail_discovery=False,
    ):
        self.allowed_domain = allowed_domain
        self.fail_discovery = fail_discovery
        self.options = timeout, delay_seconds, max_response_bytes

    def discover_links(self, url):
        if self.fail_discovery:
            raise CrawlError("robots.txt denied this request.")
        if url == ROOT_URL:
            return CrawledPage(ROOT_URL, "College", [CHILD_URL])
        return CrawledPage(CHILD_URL, "Notices", [])

    def fetch_resource(self, url, *, max_bytes):
        content = {
            ROOT_URL: (
                b"<html><head><title>College</title></head>"
                b"<body>Official student information and services.</body></html>"
            ),
            CHILD_URL: (
                b"<html><head><title>Notice</title></head>"
                b"<body>Examination notice and timetable information.</body></html>"
            ),
        }[url]
        assert len(content) <= max_bytes
        return FetchedResource(url, "text/html", content)


def create_database(database_path):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        source_id = insert_source(
            connection,
            name="College homepage",
            url=ROOT_URL,
            category="general",
            crawl_depth=1,
        )
    return source_id


def configure_sync(monkeypatch, database_path):
    monkeypatch.setenv("COLLEGE_DOMAIN", "bkbck.edu.in")
    monkeypatch.setenv("CRAWL_TIMEOUT", "15")
    monkeypatch.setenv("CRAWL_DELAY_SECONDS", "0")
    monkeypatch.setenv("MAX_DOCUMENT_BYTES", "10485760")
    monkeypatch.setenv("DATABASE_PATH", str(database_path))


def test_sync_collects_chunks_and_rebuilds_index_once_for_unchanged_sources(
    monkeypatch, tmp_path
):
    database_path = tmp_path / "sync.db"
    source_id = create_database(database_path)
    configure_sync(monkeypatch, database_path)
    index_is_current = False
    builds = []

    def build_index(**kwargs):
        nonlocal index_is_current
        builds.append(kwargs)
        index_is_current = True
        return SimpleNamespace(
            vector_count=2,
            dimensions=3,
            model_name="test-model",
            chunk_generation=2,
        )

    def check_index(**kwargs):
        assert kwargs["database_path"] == database_path
        return index_is_current

    monkeypatch.setattr(
        "backend.services.sync.is_vector_index_current",
        check_index,
    )
    monkeypatch.setattr("backend.services.sync.build_vector_index", build_index)

    first = sync_sources(
        source_ids=[source_id],
        database_path=database_path,
        storage_root=tmp_path / "raw",
        index_directory=tmp_path / "index",
        crawler_factory=FakeCrawler,
    )
    second = sync_sources(
        source_id=source_id,
        database_path=database_path,
        storage_root=tmp_path / "raw",
        index_directory=tmp_path / "index",
        crawler_factory=FakeCrawler,
    )

    with get_connection(database_path) as connection:
        documents = list_documents(connection)
        chunks = list_chunks_for_embedding(connection)
        source = connection.execute(
            "SELECT last_synced_at FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()

    assert first.status == "succeeded"
    assert first.sources[0].status == "succeeded"
    assert first.sources[0].pages_discovered == 2
    assert first.sources[0].documents_processed == 2
    assert first.sources[0].chunks_added == 2
    assert first.index_status == "rebuilt"
    assert second.status == "succeeded"
    assert second.sources[0].documents_unchanged == 2
    assert second.sources[0].chunks_added == 0
    assert second.index_status == "current"
    assert len(builds) == 1
    assert len(documents) == 2
    assert len(chunks) == 2
    assert source["last_synced_at"] is not None


def test_sync_reports_discovery_failure_without_suppressing_it(monkeypatch, tmp_path):
    database_path = tmp_path / "sync.db"
    source_id = create_database(database_path)
    configure_sync(monkeypatch, database_path)

    def failing_crawler(*args, **kwargs):
        return FakeCrawler(*args, **kwargs, fail_discovery=True)

    result = sync_sources(
        source_id=source_id,
        database_path=database_path,
        storage_root=tmp_path / "raw",
        crawler_factory=failing_crawler,
    )

    assert result.status == "failed"
    assert result.sources[0].status == "failed"
    assert result.sources[0].errors == [
        {
            "stage": "discovery",
            "url": ROOT_URL,
            "message": "robots.txt denied this request.",
        }
    ]


def test_sync_reports_index_failure_after_preserving_collected_chunks(
    monkeypatch, tmp_path
):
    database_path = tmp_path / "sync.db"
    source_id = create_database(database_path)
    configure_sync(monkeypatch, database_path)

    def check_missing_index(**kwargs):
        assert "database_path" in kwargs
        return False

    monkeypatch.setattr(
        "backend.services.sync.is_vector_index_current",
        check_missing_index,
    )

    def fail_index(**kwargs):
        assert "database_path" in kwargs
        raise EmbeddingError("Sentence Transformers is unavailable.")

    monkeypatch.setattr("backend.services.sync.build_vector_index", fail_index)
    result = sync_sources(
        source_id=source_id,
        database_path=database_path,
        storage_root=tmp_path / "raw",
        crawler_factory=FakeCrawler,
    )

    with get_connection(database_path) as connection:
        chunks = list_chunks_for_embedding(connection)

    assert result.status == "partial"
    assert result.sources[0].status == "succeeded"
    assert result.index_status == "failed"
    assert result.errors == [
        {
            "stage": "index",
            "message": "Sentence Transformers is unavailable.",
        }
    ]
    assert len(chunks) == 2


def test_sync_rejects_disabled_or_unknown_sources(tmp_path):
    database_path = tmp_path / "sync.db"
    source_id = create_database(database_path)

    with get_connection(database_path) as connection:
        connection.execute("UPDATE sources SET enabled = 0 WHERE id = ?", (source_id,))

    with pytest.raises(ValueError, match="disabled"):
        sync_sources(source_id=source_id, database_path=database_path)

    with pytest.raises(LookupError, match="not registered"):
        sync_sources(source_id=999, database_path=database_path)
