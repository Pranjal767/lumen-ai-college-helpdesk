"""Fetch, validate and persist approved HTML and PDF documents."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pymupdf
from bs4 import BeautifulSoup

from backend.database.db import (
    PROJECT_ROOT,
    get_connection,
    get_database_path,
    initialize_database,
)
from backend.database.models import (
    get_document_by_url,
    get_latest_document_version,
    get_source,
    insert_document_version,
    upsert_document,
)
from backend.services.crawler import (
    FetchedResource,
    SafeCrawler,
    normalize_allowed_url,
)

DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


class DocumentExtractionError(RuntimeError):
    """Raised when a supported document cannot be parsed for text."""


@dataclass(frozen=True)
class ExtractedDocument:
    """Extracted text, display title and extraction status."""

    title: str
    text: str
    status: str


@dataclass(frozen=True)
class StoredDocument:
    """Metadata for a successfully fetched and stored source document."""

    id: int
    title: str
    source_url: str
    document_type: str
    content_hash: str
    file_path: Path
    status: str
    version_number: int
    extraction_status: str
    extracted_text: str


def fetch_and_store_document(
    source_id: int,
    url: str,
    crawler: SafeCrawler,
    *,
    category: str | None = None,
    database_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    max_bytes: int | None = None,
    resource: FetchedResource | None = None,
    force_reextract: bool = False,
) -> StoredDocument:
    """Fetch one approved HTML/PDF URL and persist its bytes and metadata."""
    byte_limit = _resolve_byte_limit(max_bytes)
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)

    with get_connection(resolved_database_path) as connection:
        source = get_source(connection, source_id)
        if source is None:
            raise LookupError(f"Source {source_id} is not registered.")
        stored_category = category if category is not None else source["category"]

    resource = resource or crawler.fetch_resource(url, max_bytes=byte_limit)
    if len(resource.content) > byte_limit:
        raise ValueError("The fetched document exceeds the configured size limit.")
    safe_url = normalize_allowed_url(resource.url, crawler.allowed_domain)
    document_source_type = (
        "official"
        if urlsplit(safe_url).hostname == "bkbck.edu.in"
        else "external"
    )
    document_type = _validate_resource(safe_url, resource)
    content_hash = hashlib.sha256(resource.content).hexdigest()
    with get_connection(resolved_database_path) as connection:
        existing = get_document_by_url(connection, safe_url)
        previous_version = (
            get_latest_document_version(connection, int(existing["id"]))
            if existing is not None
            else None
        )
    unchanged = existing is not None and existing["content_hash"] == content_hash
    if (
        unchanged
        and previous_version is not None
        and previous_version["content_hash"] == content_hash
        and not force_reextract
    ):
        extracted_document = ExtractedDocument(
            title=existing["title"],
            text=previous_version["extracted_text"],
            status=previous_version["extraction_status"],
        )
    else:
        extracted_document = _extract_document(safe_url, resource.content, document_type)

    extension = "pdf" if document_type == "pdf" else "html"
    root = Path(storage_root) if storage_root is not None else PROJECT_ROOT / "data" / "raw"
    target_path = root / document_type / f"{content_hash}.{extension}"

    with get_connection(resolved_database_path) as connection:
        if not target_path.exists():
            _write_content_atomically(target_path, resource.content)

        fetched_at = datetime.now(timezone.utc).isoformat()
        document_id = upsert_document(
            connection,
            source_id=source_id,
            title=extracted_document.title,
            source_url=safe_url,
            document_type=document_type,
            category=stored_category,
            content_hash=content_hash,
            file_path=str(target_path.resolve()),
            fetched_at=fetched_at,
            status="unchanged" if unchanged else "fetched",
            source_type=document_source_type,
        )
        latest_version = get_latest_document_version(connection, document_id)
        if (
            latest_version is None
            or latest_version["content_hash"] != content_hash
            or force_reextract
        ):
            version_number = (
                1 if latest_version is None else latest_version["version_number"] + 1
            )
            insert_document_version(
                connection,
                document_id=document_id,
                version_number=version_number,
                content_hash=content_hash,
                file_path=str(target_path.resolve()),
                extracted_text=extracted_document.text,
                extraction_status=extracted_document.status,
                created_at=fetched_at,
            )
        else:
            version_number = int(latest_version["version_number"])

    return StoredDocument(
        id=document_id,
        title=extracted_document.title,
        source_url=safe_url,
        document_type=document_type,
        content_hash=content_hash,
        file_path=target_path.resolve(),
        status="unchanged" if unchanged else "fetched",
        version_number=version_number,
        extraction_status=extracted_document.status,
        extracted_text=extracted_document.text,
    )


def _resolve_byte_limit(max_bytes: int | None) -> int:
    if max_bytes is None:
        configured_limit = os.environ.get(
            "MAX_DOCUMENT_BYTES", str(DEFAULT_MAX_DOCUMENT_BYTES)
        )
        try:
            max_bytes = int(configured_limit)
        except ValueError as error:
            raise ValueError("MAX_DOCUMENT_BYTES must be an integer.") from error
    if max_bytes < 1:
        raise ValueError("Maximum document size must be at least one byte.")
    return max_bytes


def _validate_resource(url: str, resource: FetchedResource) -> str:
    content_type = resource.content_type.split(";", maxsplit=1)[0].strip().lower()
    is_pdf_url = urlsplit(url).path.lower().endswith(".pdf")

    if is_pdf_url:
        if content_type not in {"application/pdf", "application/octet-stream"}:
            raise ValueError("PDF URL did not return a PDF-compatible content type.")
        if not resource.content.startswith(b"%PDF-"):
            raise ValueError("PDF response is missing the PDF file signature.")
        return "pdf"

    if content_type != "text/html":
        raise ValueError("Document URL did not return an HTML page.")
    return "html"


def _extract_document(
    url: str, content: bytes, document_type: str
) -> ExtractedDocument:
    if document_type == "html":
        soup = BeautifulSoup(content, "html.parser")
        for element in soup(["script", "style", "noscript", "template"]):
            element.decompose()
        title_element = soup.find("title")
        title = title_element.get_text(" ", strip=True) if title_element else ""
        for element in soup.select(
            "header, footer, nav, aside, [role='navigation'], "
            "[aria-label*='Menu'], .menu, .main-menu, .nav-menu, .breadcrumb, "
            ".breadcrumbs, .elementor-location-header, .elementor-location-footer, "
            ".elementor-widget-nav-menu, .site-navigation, .main-navigation, "
            "#masthead, #colophon"
        ):
            element.decompose()
        if soup.head:
            soup.head.decompose()
        content_root = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body or soup
        text = "\n".join(
            line.strip()
            for line in content_root.get_text("\n").splitlines()
            if line.strip()
        )
        return ExtractedDocument(
            title=title or _filename_title(url),
            text=text,
            status="extracted" if text else "no_text",
        )

    try:
        pdf_document = pymupdf.open(stream=content, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
        raise DocumentExtractionError("The PDF could not be opened.") from error

    with pdf_document:
        if pdf_document.needs_pass:
            raise DocumentExtractionError(
                "The PDF is password-protected and cannot be extracted."
            )
        page_texts = [page.get_text("text").strip() for page in pdf_document]
        metadata = pdf_document.metadata or {}
        title = (metadata.get("title") or "").strip()
        text = "\f".join(page_texts)
        return ExtractedDocument(
            title=title or _filename_title(url),
            text=text,
            status="extracted" if any(page_texts) else "no_text",
        )


def _filename_title(url: str) -> str:
    filename = unquote(Path(urlsplit(url).path).name).strip()
    return filename or url


def _write_content_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".lumen-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
