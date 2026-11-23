"""Bounded source synchronization and vector-index refresh orchestration."""

import math
import os
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database.db import get_connection, get_database_path, initialize_database
from backend.database.models import (
    get_latest_document_version,
    get_source,
    list_chunks_for_embedding,
    list_document_version_chunks,
    list_sources,
    update_source_last_synced,
)
from backend.services.chunker import chunk_document_version
from backend.services.collection import collect_source_links
from backend.services.crawler import CrawlError, SafeCrawler
from backend.services.document_fetcher import (
    DocumentExtractionError,
    fetch_and_store_document,
)
from backend.services.embeddings import EmbeddingError
from backend.services.vector_store import (
    VectorIndexError,
    build_vector_index,
    is_vector_index_current,
)

MAX_SYNC_PAGES_PER_SOURCE = 25
MAX_SYNC_DOCUMENTS_PER_SOURCE = 50


class SyncConfigurationError(RuntimeError):
    """Raised when source synchronization is not configured correctly."""


class NoEnabledSourcesError(ValueError):
    """Raised when a synchronization request has no enabled sources."""


class DisabledSourceError(ValueError):
    """Raised when an admin requests synchronization of a disabled source."""


@dataclass
class SourceSyncSummary:
    """Per-source progress and error details from one synchronization run."""

    source_id: int
    name: str
    url: str
    status: str = "pending"
    pages_discovered: int = 0
    documents_discovered: int = 0
    documents_processed: int = 0
    documents_unchanged: int = 0
    chunks_added: int = 0
    truncated: bool = False
    errors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SyncRunSummary:
    """Complete report for a bounded, synchronous synchronization run."""

    status: str
    started_at: str
    completed_at: str
    sources: list[SourceSyncSummary]
    index_status: str
    index: dict[str, Any] | None
    errors: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable run report."""
        return {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "sources": [asdict(source) for source in self.sources],
            "index_status": self.index_status,
            "index": self.index,
            "errors": self.errors,
        }


def sync_sources(
    *,
    source_id: int | None = None,
    source_ids: Sequence[int] | None = None,
    database_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    index_directory: str | Path | None = None,
    crawler_factory: Callable[..., SafeCrawler] | None = None,
    progress_callback: Callable[[SourceSyncSummary], None] | None = None,
    index_status_callback: Callable[[str], None] | None = None,
) -> SyncRunSummary:
    """Collect enabled approved sources, chunk new content, and refresh search."""
    if source_id is not None and source_ids is not None:
        raise ValueError("Provide either source_id or source_ids, not both.")
    if source_id is not None and (
        isinstance(source_id, bool) or not isinstance(source_id, int) or source_id < 1
    ):
        raise ValueError("source_id must be a positive integer.")
    if source_ids is not None and (
        isinstance(source_ids, (str, bytes))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in source_ids
        )
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("source_ids must contain unique positive integers.")

    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    sources = _get_sources(source_id, source_ids, resolved_database_path)
    if not sources:
        raise NoEnabledSourcesError("There are no enabled sources to synchronize.")

    domain, timeout, delay_seconds, max_bytes = _crawler_settings()
    make_crawler = crawler_factory or SafeCrawler
    started_at = _utc_now()
    summaries: list[SourceSyncSummary] = []
    chunks_changed = False

    for source in sources:
        summary, source_chunks_changed = _sync_one_source(
            source,
            domain=domain,
            timeout=timeout,
            delay_seconds=delay_seconds,
            max_bytes=max_bytes,
            database_path=resolved_database_path,
            storage_root=storage_root,
            crawler_factory=make_crawler,
            progress_callback=progress_callback,
        )
        summaries.append(summary)
        chunks_changed = chunks_changed or source_chunks_changed

    errors: list[dict[str, str]] = []
    index_status = "no_chunks"
    index_summary: dict[str, Any] | None = None
    # The student assistant is intentionally scoped to recent student-facing
    # records, including when its normal scheduled source sync rebuilds the index.
    from backend.services.official_site_snapshot import (
        _current_student_chunk,
        _current_student_document_ids,
    )

    selected_document_ids, _ = _current_student_document_ids(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        all_chunks = list_chunks_for_embedding(connection)
    is_college_corpus = len(all_chunks) > 20 and any(
        "bkbck.edu.in" in str(row["source_url"]).lower() for row in all_chunks
    )
    if is_college_corpus:
        chunks = [
            row for row in all_chunks
            if int(row["document_id"]) in selected_document_ids
            and _current_student_chunk(row)
        ]
    else:
        # Keep generic source-sync behavior for tests and other configured hosts.
        chunks = all_chunks

    if chunks:
        _emit_index_status(index_status_callback, "checking")
        try:
            index_is_current = is_vector_index_current(
                database_path=resolved_database_path,
                index_directory=index_directory,
            )
            if chunks_changed or not index_is_current:
                _emit_index_status(index_status_callback, "building")
                index = build_vector_index(
                    database_path=resolved_database_path,
                    index_directory=index_directory,
                    document_ids=selected_document_ids if is_college_corpus else None,
                    chunk_filter=_current_student_chunk if is_college_corpus else None,
                )
                index_status = "rebuilt"
                index_summary = {
                    "vector_count": index.vector_count,
                    "dimensions": index.dimensions,
                    "model_name": index.model_name,
                    "chunk_generation": index.chunk_generation,
                }
                _emit_index_status(index_status_callback, "current")
            else:
                index_status = "current"
                _emit_index_status(index_status_callback, index_status)
        except (EmbeddingError, VectorIndexError, ValueError, OSError, sqlite3.Error) as error:
            index_status = "failed"
            _emit_index_status(index_status_callback, index_status)
            errors.append({"stage": "index", "message": str(error)})
    else:
        _emit_index_status(index_status_callback, index_status)

    source_statuses = [summary.status for summary in summaries]
    if all(status == "failed" for status in source_statuses):
        run_status = "failed"
    elif (
        any(status in {"failed", "partial"} for status in source_statuses)
        or index_status == "failed"
    ):
        run_status = "partial"
    else:
        run_status = "succeeded"

    return SyncRunSummary(
        status=run_status,
        started_at=started_at,
        completed_at=_utc_now(),
        sources=summaries,
        index_status=index_status,
        index=index_summary,
        errors=errors,
    )


def _get_sources(
    source_id: int | None,
    source_ids: Sequence[int] | None,
    database_path: Path,
) -> list[sqlite3.Row]:
    with get_connection(database_path) as connection:
        if source_id is not None:
            source_ids = [source_id]
        if source_ids is not None:
            sources = []
            for requested_id in source_ids:
                source = get_source(connection, requested_id)
                if source is None:
                    raise LookupError(f"Source {requested_id} is not registered.")
                if not bool(source["enabled"]):
                    raise DisabledSourceError(f"Source {requested_id} is disabled.")
                sources.append(source)
            return sources
        return [
            source for source in list_sources(connection) if bool(source["enabled"])
        ]


def _crawler_settings() -> tuple[str, float, float, int]:
    domain = os.environ.get("COLLEGE_DOMAIN", "").strip()
    if not domain:
        raise SyncConfigurationError("COLLEGE_DOMAIN is not configured.")
    try:
        timeout = float(os.environ.get("CRAWL_TIMEOUT", "15"))
        delay_seconds = float(os.environ.get("CRAWL_DELAY_SECONDS", "1"))
        max_bytes = int(os.environ.get("MAX_DOCUMENT_BYTES", str(10 * 1024 * 1024)))
    except ValueError as error:
        raise SyncConfigurationError(
            "CRAWL_TIMEOUT, CRAWL_DELAY_SECONDS, and MAX_DOCUMENT_BYTES "
            "must contain valid numbers."
        ) from error
    if (
        not math.isfinite(timeout)
        or not math.isfinite(delay_seconds)
        or timeout <= 0
        or delay_seconds < 0
        or max_bytes < 1
    ):
        raise SyncConfigurationError("Crawler configuration values are out of range.")
    return domain, timeout, delay_seconds, max_bytes


def _sync_one_source(
    source: sqlite3.Row,
    *,
    domain: str,
    timeout: float,
    delay_seconds: float,
    max_bytes: int,
    database_path: Path,
    storage_root: str | Path | None,
    crawler_factory: Callable[..., SafeCrawler],
    progress_callback: Callable[[SourceSyncSummary], None] | None,
) -> tuple[SourceSyncSummary, bool]:
    summary = SourceSyncSummary(
        source_id=int(source["id"]),
        name=source["name"],
        url=source["url"],
        status="running",
    )
    _emit_progress(progress_callback, summary)
    crawler = crawler_factory(
        domain,
        timeout=timeout,
        delay_seconds=delay_seconds,
        max_response_bytes=max_bytes,
    )
    try:
        collection = collect_source_links(
            source["url"],
            crawler,
            max_depth=int(source["crawl_depth"]),
            max_pages=MAX_SYNC_PAGES_PER_SOURCE,
        )
    except (CrawlError, ValueError) as error:
        summary.status = "failed"
        summary.errors.append({"stage": "discovery", "url": source["url"], "message": str(error)})
        _emit_progress(progress_callback, summary)
        return summary, False

    summary.pages_discovered = len(collection.pages)
    document_urls = list(
        dict.fromkeys([page.url for page in collection.pages] + collection.pdf_urls)
    )
    summary.documents_discovered = len(document_urls)
    summary.truncated = collection.truncated
    _emit_progress(progress_callback, summary)
    if len(document_urls) > MAX_SYNC_DOCUMENTS_PER_SOURCE:
        document_urls = document_urls[:MAX_SYNC_DOCUMENTS_PER_SOURCE]
        summary.truncated = True
        summary.errors.append(
            {
                "stage": "limit",
                "url": source["url"],
                "message": (
                    f"Only the first {MAX_SYNC_DOCUMENTS_PER_SOURCE} discovered "
                    "documents were processed."
                ),
            }
        )
    elif collection.truncated:
        summary.errors.append(
            {
                "stage": "limit",
                "url": source["url"],
                "message": (
                    f"Page discovery reached its limit of "
                    f"{MAX_SYNC_PAGES_PER_SOURCE} pages."
                ),
            }
        )

    chunks_changed = False
    for url in document_urls:
        try:
            document = fetch_and_store_document(
                summary.source_id,
                url,
                crawler,
                category=source["category"],
                database_path=database_path,
                storage_root=storage_root,
                max_bytes=max_bytes,
            )
            summary.documents_processed += 1
            if document.status == "unchanged":
                summary.documents_unchanged += 1

            with get_connection(database_path) as connection:
                version = get_latest_document_version(connection, document.id)
                if version is None:
                    raise RuntimeError(
                        f"Stored document {document.id} has no extracted version."
                    )
                existing_chunks = list_document_version_chunks(
                    connection, int(version["id"])
                )

            if version["extraction_status"] != "extracted" or existing_chunks:
                continue
            created_chunks = chunk_document_version(
                int(version["id"]),
                database_path=database_path,
            )
            summary.chunks_added += len(created_chunks)
            chunks_changed = chunks_changed or bool(created_chunks)
        except (
            CrawlError,
            DocumentExtractionError,
            ValueError,
            LookupError,
            OSError,
            sqlite3.Error,
        ) as error:
            summary.errors.append(
                {"stage": "document", "url": url, "message": str(error)}
            )
        _emit_progress(progress_callback, summary)

    if summary.errors:
        summary.status = (
            "partial" if summary.documents_processed > 0 else "failed"
        )
    else:
        summary.status = "succeeded"

    if summary.status == "succeeded":
        synced_at = _utc_now()
        with get_connection(database_path) as connection:
            if not update_source_last_synced(
                connection, summary.source_id, synced_at
            ):
                raise RuntimeError(
                    f"Source {summary.source_id} disappeared during synchronization."
                )
    _emit_progress(progress_callback, summary)
    return summary, chunks_changed


def _emit_progress(
    callback: Callable[[SourceSyncSummary], None] | None,
    summary: SourceSyncSummary,
) -> None:
    if callback is not None:
        callback(summary)


def _emit_index_status(
    callback: Callable[[str], None] | None,
    status: str,
) -> None:
    if callback is not None:
        callback(status)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
