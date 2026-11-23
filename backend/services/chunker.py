"""Split extracted document text into overlapping, page-aware chunks."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.database.db import get_connection, get_database_path, initialize_database
from backend.database.models import (
    get_document_version_for_chunking,
    replace_document_version_chunks,
)

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class TextChunk:
    """A chunk of document text and its source page when available."""

    chunk_index: int
    content: str
    page_number: int | None
    token_count: int


@dataclass(frozen=True)
class StoredChunk(TextChunk):
    """A persisted text chunk linked to a document version."""

    id: int
    document_version_id: int


def chunk_text(
    extracted_text: str,
    *,
    document_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split text by whitespace tokens, preserving PDF page boundaries."""
    if document_type not in {"html", "pdf"}:
        raise ValueError("Document type must be 'html' or 'pdf'.")
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least one token.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    pages = extracted_text.split("\f") if document_type == "pdf" else [extracted_text]
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page_index, page_text in enumerate(pages):
        words = page_text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=" ".join(chunk_words),
                    page_number=page_index + 1 if document_type == "pdf" else None,
                    token_count=len(chunk_words),
                )
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - overlap

    return chunks


def chunk_document_version(
    version_id: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    database_path: str | Path | None = None,
) -> list[StoredChunk]:
    """Chunk and persist one extracted version, replacing prior chunks."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)

    with get_connection(resolved_database_path) as connection:
        version = get_document_version_for_chunking(connection, version_id)
        if version is None:
            raise LookupError(f"Document version {version_id} does not exist.")

        text_chunks = chunk_text(
            version["extracted_text"],
            document_type=version["document_type"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        rows = replace_document_version_chunks(
            connection,
            version_id,
            [
                (
                    chunk.chunk_index,
                    chunk.content,
                    chunk.page_number,
                    chunk.token_count,
                )
                for chunk in text_chunks
            ],
            datetime.now(timezone.utc).isoformat(),
        )

    return [
        StoredChunk(
            id=row["id"],
            document_version_id=row["document_version_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            page_number=row["page_number"],
            token_count=row["token_count"],
        )
        for row in rows
    ]
