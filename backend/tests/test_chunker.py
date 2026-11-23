"""Tests for page-aware chunking and chunk persistence."""

from datetime import datetime, timezone

import pytest

from backend.database.db import get_connection, initialize_database
from backend.database.models import (
    insert_document_version,
    insert_source,
    list_document_version_chunks,
    upsert_document,
)
from backend.services.chunker import chunk_document_version, chunk_text


def create_document_version(database_path, text, document_type):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        source_id = insert_source(
            connection,
            name="College notices",
            url="https://bkbck.edu.in/notices/",
            category="notices",
        )
        document_id = upsert_document(
            connection,
            source_id=source_id,
            title="Test document",
            source_url="https://bkbck.edu.in/notices/test",
            document_type=document_type,
            category="notices",
            content_hash="test-hash",
            file_path="data/raw/test",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            status="fetched",
        )
        return insert_document_version(
            connection,
            document_id=document_id,
            version_number=1,
            content_hash="test-hash",
            file_path="data/raw/test",
            extracted_text=text,
            extraction_status="extracted" if text else "no_text",
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def test_chunk_text_uses_overlap_without_exceeding_limit():
    chunks = chunk_text(
        "one two three four five six",
        document_type="html",
        chunk_size=3,
        overlap=1,
    )

    assert [chunk.content for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]
    assert [chunk.token_count for chunk in chunks] == [3, 3, 2]
    assert all(chunk.page_number is None for chunk in chunks)


def test_pdf_chunks_preserve_original_page_numbers():
    chunks = chunk_text(
        "first page has four\f\fthird page text",
        document_type="pdf",
        chunk_size=2,
        overlap=0,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 1, 3, 3]
    assert [chunk.content for chunk in chunks] == [
        "first page",
        "has four",
        "third page",
        "text",
    ]


def test_empty_text_returns_no_chunks():
    assert chunk_text(" \n\t", document_type="html") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (3, -1), (3, 3)],
)
def test_chunk_text_rejects_invalid_size_and_overlap(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_type="html",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_document_version_persists_and_replaces_chunks(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    version_id = create_document_version(
        database_path,
        "one two three four five six",
        "html",
    )

    initial_chunks = chunk_document_version(
        version_id,
        chunk_size=3,
        overlap=1,
        database_path=database_path,
    )
    repeated_chunks = chunk_document_version(
        version_id,
        chunk_size=3,
        overlap=1,
        database_path=database_path,
    )

    assert [chunk.content for chunk in initial_chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]
    assert [chunk.content for chunk in repeated_chunks] == [
        chunk.content for chunk in initial_chunks
    ]
    with get_connection(database_path) as connection:
        rows = list_document_version_chunks(connection, version_id)
    assert len(rows) == 3
    assert [row["chunk_index"] for row in rows] == [0, 1, 2]
    assert all(row["document_version_id"] == version_id for row in rows)

    resized_chunks = chunk_document_version(
        version_id,
        chunk_size=2,
        overlap=0,
        database_path=database_path,
    )
    assert len(resized_chunks) == 3
    with get_connection(database_path) as connection:
        rows = list_document_version_chunks(connection, version_id)
    assert len(rows) == 3
    assert [row["content"] for row in rows] == ["one two", "three four", "five six"]


def test_chunk_document_version_keeps_pdf_page_numbers(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    version_id = create_document_version(
        database_path,
        "page one\fpage two",
        "pdf",
    )

    chunks = chunk_document_version(
        version_id,
        chunk_size=2,
        overlap=0,
        database_path=database_path,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.content for chunk in chunks] == ["page one", "page two"]


def test_chunking_empty_version_removes_previous_chunks(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    version_id = create_document_version(database_path, "some text", "html")
    chunk_document_version(version_id, database_path=database_path)

    with get_connection(database_path) as connection:
        connection.execute(
            "UPDATE document_versions SET extracted_text = '' WHERE id = ?",
            (version_id,),
        )

    chunks = chunk_document_version(version_id, database_path=database_path)

    assert chunks == []
    with get_connection(database_path) as connection:
        assert list_document_version_chunks(connection, version_id) == []


def test_missing_document_version_is_reported(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    initialize_database(database_path)

    with pytest.raises(LookupError, match="does not exist"):
        chunk_document_version(404, database_path=database_path)
