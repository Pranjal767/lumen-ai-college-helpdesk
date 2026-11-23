"""Tests for bounded document fetching and metadata persistence."""

import hashlib
from unittest.mock import Mock

import pymupdf
import pytest

from backend.database.db import get_connection, initialize_database
from backend.database.models import (
    insert_source,
    list_document_versions,
    list_documents,
)
from backend.services.crawler import FetchedResource
from backend.services.document_fetcher import (
    DocumentExtractionError,
    fetch_and_store_document,
)


def make_pdf(*page_texts):
    pdf_document = pymupdf.open()
    for text in page_texts:
        page = pdf_document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = pdf_document.tobytes()
    pdf_document.close()
    return content


def make_password_protected_pdf():
    pdf_document = pymupdf.open()
    pdf_document.new_page().insert_text((72, 72), "Protected text")
    content = pdf_document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    pdf_document.close()
    return content


def create_source(database_path):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        return insert_source(
            connection,
            name="College notices",
            url="https://bkbck.edu.in/notices/",
            category="notices",
        )


def make_crawler(url, content, content_type):
    crawler = Mock()
    crawler.allowed_domain = "bkbck.edu.in"
    crawler.fetch_resource.return_value = FetchedResource(
        url=url,
        content_type=content_type,
        content=content,
    )
    return crawler


def test_html_document_is_stored_with_hash_and_metadata(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    content = (
        b"<html><title>Exam timetable</title><body><p>Schedule</p>"
        b"<script>ignore this</script></body></html>"
    )
    crawler = make_crawler(
        "https://bkbck.edu.in/notices/exam.html",
        content,
        "text/html; charset=utf-8",
    )

    document = fetch_and_store_document(
        source_id,
        "https://bkbck.edu.in/notices/exam.html",
        crawler,
        database_path=database_path,
        storage_root=tmp_path / "raw",
    )

    assert document.title == "Exam timetable"
    assert document.document_type == "html"
    assert document.content_hash == hashlib.sha256(content).hexdigest()
    assert document.status == "fetched"
    assert document.extraction_status == "extracted"
    assert document.extracted_text == "Schedule"
    assert document.version_number == 1
    assert document.file_path.read_bytes() == content
    with get_connection(database_path) as connection:
        rows = list_documents(connection)
    assert len(rows) == 1
    assert rows[0]["source_id"] == source_id
    assert rows[0]["category"] == "notices"
    assert rows[0]["content_hash"] == document.content_hash
    with get_connection(database_path) as connection:
        versions = list_document_versions(connection, document.id)
    assert len(versions) == 1
    assert versions[0]["extracted_text"] == "Schedule"


def test_pdf_document_is_signature_checked_and_stored(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    content = make_pdf("First page text", "Second page text")
    crawler = make_crawler(
        "https://bkbck.edu.in/notices/notice.pdf",
        content,
        "application/pdf",
    )

    document = fetch_and_store_document(
        source_id,
        "https://bkbck.edu.in/notices/notice.pdf",
        crawler,
        database_path=database_path,
        storage_root=tmp_path / "raw",
    )

    assert document.document_type == "pdf"
    assert document.title == "notice.pdf"
    assert document.file_path.suffix == ".pdf"
    assert document.file_path.read_bytes() == content
    assert document.extracted_text == "First page text\fSecond page text"
    assert document.extraction_status == "extracted"
    assert document.version_number == 1


def test_unchanged_document_is_not_reextracted_or_duplicated(tmp_path, monkeypatch):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    url = "https://bkbck.edu.in/notices/exam.html"
    crawler = make_crawler(url, b"<title>Exam notice</title>", "text/html")
    storage_root = tmp_path / "raw"

    first = fetch_and_store_document(
        source_id,
        url,
        crawler,
        database_path=database_path,
        storage_root=storage_root,
    )
    monkeypatch.setattr(
        "backend.services.document_fetcher._extract_document",
        lambda *_: pytest.fail("unchanged document was extracted again"),
    )
    second = fetch_and_store_document(
        source_id,
        url,
        crawler,
        database_path=database_path,
        storage_root=storage_root,
    )

    assert first.id == second.id
    assert first.content_hash == second.content_hash
    assert second.status == "unchanged"
    assert second.version_number == 1
    assert len(list((storage_root / "html").glob("*.html"))) == 1
    with get_connection(database_path) as connection:
        assert len(list_documents(connection)) == 1
        assert len(list_document_versions(connection, first.id)) == 1


def test_changed_document_updates_metadata_and_keeps_content_addressed_file(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    url = "https://bkbck.edu.in/notices/exam.html"
    storage_root = tmp_path / "raw"
    crawler = make_crawler(
        url,
        b"<html><head><title>First notice</title></head>"
        b"<body>First notice body</body></html>",
        "text/html",
    )
    first = fetch_and_store_document(
        source_id,
        url,
        crawler,
        database_path=database_path,
        storage_root=storage_root,
    )
    crawler.fetch_resource.return_value = FetchedResource(
        url=url,
        content_type="text/html",
        content=b"<html><head><title>Updated notice</title></head>"
        b"<body>Updated notice body</body></html>",
    )

    second = fetch_and_store_document(
        source_id,
        url,
        crawler,
        database_path=database_path,
        storage_root=storage_root,
    )

    assert first.id == second.id
    assert second.title == "Updated notice"
    assert second.status == "fetched"
    assert second.version_number == 2
    assert first.content_hash != second.content_hash
    assert first.file_path.exists()
    assert second.file_path.exists()
    with get_connection(database_path) as connection:
        assert len(list_documents(connection)) == 1
        versions = list_document_versions(connection, first.id)
    assert [version["version_number"] for version in versions] == [1, 2]
    assert [version["extracted_text"] for version in versions] == [
        "First notice body",
        "Updated notice body",
    ]


@pytest.mark.parametrize(
    ("url", "content", "content_type"),
    [
        ("https://bkbck.edu.in/notice.pdf", b"not a PDF", "application/pdf"),
        ("https://bkbck.edu.in/notice.html", b"<html/>", "application/octet-stream"),
    ],
)
def test_unsupported_or_malformed_resource_is_not_stored(
    tmp_path, url, content, content_type
):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    crawler = make_crawler(url, content, content_type)

    with pytest.raises(ValueError):
        fetch_and_store_document(
            source_id,
            url,
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
        )

    with get_connection(database_path) as connection:
        assert list_documents(connection) == []
    assert not (tmp_path / "raw").exists()


def test_document_service_enforces_size_limit_even_if_crawler_does_not(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    crawler = make_crawler(
        "https://bkbck.edu.in/notice.html",
        b"<title>Too large</title>",
        "text/html",
    )

    with pytest.raises(ValueError, match="size limit"):
        fetch_and_store_document(
            source_id,
            "https://bkbck.edu.in/notice.html",
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
            max_bytes=5,
        )


def test_unknown_source_is_rejected_before_fetching(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    initialize_database(database_path)
    crawler = Mock()

    with pytest.raises(LookupError, match="not registered"):
        fetch_and_store_document(
            999,
            "https://bkbck.edu.in/notice.html",
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
        )

    crawler.fetch_resource.assert_not_called()


def test_max_document_bytes_environment_value_is_validated(tmp_path, monkeypatch):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    monkeypatch.setenv("MAX_DOCUMENT_BYTES", "not-a-number")
    crawler = Mock()

    with pytest.raises(ValueError, match="MAX_DOCUMENT_BYTES"):
        fetch_and_store_document(
            source_id,
            "https://bkbck.edu.in/notice.html",
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
        )

    crawler.fetch_resource.assert_not_called()


def test_pdf_without_embedded_text_is_recorded_as_no_text(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    url = "https://bkbck.edu.in/notices/scanned.pdf"
    crawler = make_crawler(url, make_pdf(""), "application/pdf")

    document = fetch_and_store_document(
        source_id,
        url,
        crawler,
        database_path=database_path,
        storage_root=tmp_path / "raw",
    )

    assert document.extracted_text == ""
    assert document.extraction_status == "no_text"
    with get_connection(database_path) as connection:
        version = list_document_versions(connection, document.id)[0]
    assert version["extraction_status"] == "no_text"


def test_unreadable_pdf_does_not_update_document_or_write_file(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    url = "https://bkbck.edu.in/notices/broken.pdf"
    crawler = make_crawler(url, b"%PDF-not-a-valid-document", "application/pdf")

    with pytest.raises(DocumentExtractionError, match="could not be opened"):
        fetch_and_store_document(
            source_id,
            url,
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
        )

    with get_connection(database_path) as connection:
        assert list_documents(connection) == []
    assert not (tmp_path / "raw").exists()


def test_password_protected_pdf_is_reported_without_persisting(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    source_id = create_source(database_path)
    url = "https://bkbck.edu.in/notices/protected.pdf"
    crawler = make_crawler(url, make_password_protected_pdf(), "application/pdf")

    with pytest.raises(DocumentExtractionError, match="password-protected"):
        fetch_and_store_document(
            source_id,
            url,
            crawler,
            database_path=database_path,
            storage_root=tmp_path / "raw",
        )

    with get_connection(database_path) as connection:
        assert list_documents(connection) == []
    assert not (tmp_path / "raw").exists()
