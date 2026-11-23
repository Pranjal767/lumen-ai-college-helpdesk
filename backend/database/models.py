"""Database operations for source records."""

import sqlite3
from collections.abc import Sequence


def insert_source(
    connection: sqlite3.Connection,
    *,
    name: str,
    url: str,
    source_type: str = "website",
    category: str | None = None,
    enabled: bool = True,
    crawl_depth: int = 1,
) -> int:
    """Insert a source and return its database ID."""
    cursor = connection.execute(
        """
        INSERT INTO sources (name, url, source_type, category, enabled, crawl_depth)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, url, source_type, category, int(enabled), crawl_depth),
    )
    return int(cursor.lastrowid)


def list_sources(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return sources ordered by their database ID."""
    return connection.execute(
        """
        SELECT id, name, url, source_type, category, enabled, crawl_depth,
               last_synced_at, created_at
        FROM sources
        ORDER BY id
        """
    ).fetchall()


def get_source(connection: sqlite3.Connection, source_id: int) -> sqlite3.Row | None:
    """Return one registered source, or None when it does not exist."""
    return connection.execute(
        """
        SELECT id, name, url, source_type, category, enabled, crawl_depth,
               last_synced_at, created_at
        FROM sources
        WHERE id = ?
        """,
        (source_id,),
    ).fetchone()


def update_source_enabled(
    connection: sqlite3.Connection,
    source_id: int,
    enabled: bool,
) -> bool:
    """Update a source's enabled state and report whether it exists."""
    cursor = connection.execute(
        "UPDATE sources SET enabled = ? WHERE id = ?",
        (int(enabled), source_id),
    )
    return cursor.rowcount > 0


def update_source_last_synced(
    connection: sqlite3.Connection,
    source_id: int,
    synced_at: str,
) -> bool:
    """Record a successful source synchronization and report whether it exists."""
    cursor = connection.execute(
        "UPDATE sources SET last_synced_at = ? WHERE id = ?",
        (synced_at, source_id),
    )
    return cursor.rowcount > 0


def get_document_by_url(
    connection: sqlite3.Connection, source_url: str
) -> sqlite3.Row | None:
    """Return the document currently registered for a source URL."""
    return connection.execute(
        """
        SELECT id, source_id, title, source_url, document_type, category,
               content_hash, file_path, fetched_at, status
        FROM documents
        WHERE source_url = ?
        """,
        (source_url,),
    ).fetchone()


def upsert_document(
    connection: sqlite3.Connection,
    *,
    source_id: int,
    title: str,
    source_url: str,
    document_type: str,
    category: str | None,
    content_hash: str,
    file_path: str,
    fetched_at: str,
    status: str,
    source_type: str = "official",
    record_id: str | None = None,
    metadata_json: str = "{}",
) -> int:
    """Insert or update metadata for a collected source URL."""
    connection.execute(
        """
        INSERT INTO documents (
            source_id, title, source_url, document_type, category,
            content_hash, file_path, fetched_at, status, source_type,
            record_id, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_url) DO UPDATE SET
            source_id = excluded.source_id,
            title = excluded.title,
            document_type = excluded.document_type,
            category = excluded.category,
            content_hash = excluded.content_hash,
            file_path = excluded.file_path,
            fetched_at = excluded.fetched_at,
            status = excluded.status,
            source_type = excluded.source_type,
            record_id = excluded.record_id,
            metadata_json = excluded.metadata_json
        """,
        (
            source_id,
            title,
            source_url,
            document_type,
            category,
            content_hash,
            file_path,
            fetched_at,
            status,
            source_type,
            record_id,
            metadata_json,
        ),
    )
    row = get_document_by_url(connection, source_url)
    if row is None:
        raise RuntimeError("Document metadata was not persisted.")
    return int(row["id"])


def list_documents(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return stored document metadata ordered by ID."""
    return connection.execute(
        """
        SELECT id, source_id, title, source_url, document_type, category,
               content_hash, file_path, fetched_at, status
        FROM documents
        ORDER BY id
        """
    ).fetchall()


def get_latest_document_version(
    connection: sqlite3.Connection, document_id: int
) -> sqlite3.Row | None:
    """Return the newest stored version for a document."""
    return connection.execute(
        """
        SELECT id, document_id, version_number, content_hash, file_path,
               extracted_text, extraction_status, created_at
        FROM document_versions
        WHERE document_id = ?
        ORDER BY version_number DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()


def insert_document_version(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    version_number: int,
    content_hash: str,
    file_path: str,
    extracted_text: str,
    extraction_status: str,
    created_at: str,
) -> int:
    """Persist an immutable extracted-text snapshot and return its ID."""
    cursor = connection.execute(
        """
        INSERT INTO document_versions (
            document_id, version_number, content_hash, file_path,
            extracted_text, extraction_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            version_number,
            content_hash,
            file_path,
            extracted_text,
            extraction_status,
            created_at,
        ),
    )
    return int(cursor.lastrowid)


def list_document_versions(
    connection: sqlite3.Connection, document_id: int
) -> list[sqlite3.Row]:
    """Return a document's immutable versions from oldest to newest."""
    return connection.execute(
        """
        SELECT id, document_id, version_number, content_hash, file_path,
               extracted_text, extraction_status, created_at
        FROM document_versions
        WHERE document_id = ?
        ORDER BY version_number
        """,
        (document_id,),
    ).fetchall()


def get_document_version_for_chunking(
    connection: sqlite3.Connection, version_id: int
) -> sqlite3.Row | None:
    """Return extracted text and document type for a version."""
    return connection.execute(
        """
        SELECT v.id, v.extracted_text, d.document_type
        FROM document_versions AS v
        JOIN documents AS d ON d.id = v.document_id
        WHERE v.id = ?
        """,
        (version_id,),
    ).fetchone()


def list_chunks_for_embedding(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return chunk text and source context for a full index rebuild."""
    return connection.execute(
        """
        SELECT c.id, c.content, d.id AS document_id, d.title, d.source_url,
               d.source_type, d.record_id, d.category, d.metadata_json
        FROM chunks AS c
        JOIN document_versions AS v ON v.id = c.document_version_id
        JOIN documents AS d ON d.id = v.document_id
        WHERE v.id = (
            SELECT MAX(latest.id)
            FROM document_versions AS latest
            WHERE latest.document_id = d.id
        )
        ORDER BY c.id
        """
    ).fetchall()


def get_chunks_by_ids(
    connection: sqlite3.Connection, chunk_ids: Sequence[int]
) -> list[sqlite3.Row]:
    """Return chunk text and citation metadata for the requested IDs."""
    if not chunk_ids:
        return []
    placeholders = ", ".join("?" for _ in chunk_ids)
    return connection.execute(
        f"""
        SELECT c.id, c.content, c.page_number, c.token_count,
               v.version_number, d.id AS document_id, d.title,
               d.source_url, d.document_type, d.source_type, d.record_id,
               d.category, d.metadata_json
        FROM chunks AS c
        JOIN document_versions AS v ON v.id = c.document_version_id
        JOIN documents AS d ON d.id = v.document_id
        WHERE c.id IN ({placeholders})
        """,
        tuple(chunk_ids),
    ).fetchall()


def get_chunk_generation(connection: sqlite3.Connection) -> int:
    """Return the generation that changes whenever chunks are replaced."""
    row = connection.execute(
        "SELECT generation FROM chunk_state WHERE id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("Chunk generation state has not been initialized.")
    return int(row["generation"])


def replace_document_version_chunks(
    connection: sqlite3.Connection,
    version_id: int,
    chunks: Sequence[tuple[int, str, int | None, int]],
    created_at: str,
) -> list[sqlite3.Row]:
    """Replace a version's chunks atomically and return them in index order."""
    connection.execute(
        "DELETE FROM chunks WHERE document_version_id = ?",
        (version_id,),
    )
    connection.executemany(
        """
        INSERT INTO chunks (
            document_version_id, chunk_index, content, page_number,
            token_count, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (version_id, chunk_index, content, page_number, token_count, created_at)
            for chunk_index, content, page_number, token_count in chunks
        ],
    )
    connection.execute(
        "UPDATE chunk_state SET generation = generation + 1 WHERE id = 1"
    )
    return connection.execute(
        """
        SELECT id, document_version_id, chunk_index, content,
               page_number, token_count, created_at
        FROM chunks
        WHERE document_version_id = ?
        ORDER BY chunk_index
        """,
        (version_id,),
    ).fetchall()


def list_document_version_chunks(
    connection: sqlite3.Connection, version_id: int
) -> list[sqlite3.Row]:
    """Return the stored chunks for one document version."""
    return connection.execute(
        """
        SELECT id, document_version_id, chunk_index, content,
               page_number, token_count, created_at
        FROM chunks
        WHERE document_version_id = ?
        ORDER BY chunk_index
        """,
        (version_id,),
    ).fetchall()


def insert_conversation(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    session_id: str,
    title: str,
) -> None:
    """Create a conversation for an anonymous browser session."""
    connection.execute(
        """
        INSERT INTO conversations (id, session_id, title)
        VALUES (?, ?, ?)
        """,
        (conversation_id, session_id, title),
    )


def get_conversation(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    session_id: str,
) -> sqlite3.Row | None:
    """Return a conversation only when it belongs to the given session."""
    return connection.execute(
        """
        SELECT id, session_id, title, created_at, updated_at
        FROM conversations
        WHERE id = ? AND session_id = ?
        """,
        (conversation_id, session_id),
    ).fetchone()


def list_conversations(
    connection: sqlite3.Connection, session_id: str
) -> list[sqlite3.Row]:
    """Return a session's conversations, newest activity first."""
    return connection.execute(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
        FROM conversations AS c
        LEFT JOIN messages AS m ON m.conversation_id = c.id
        WHERE c.session_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC, c.id
        """,
        (session_id,),
    ).fetchall()


def insert_message(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    """Store one user or assistant message."""
    connection.execute(
        """
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
        """,
        (conversation_id, role, content),
    )


def list_messages(
    connection: sqlite3.Connection, conversation_id: str
) -> list[sqlite3.Row]:
    """Return a conversation's messages in insertion order."""
    return connection.execute(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id
        """,
        (conversation_id,),
    ).fetchall()


def update_conversation_activity(
    connection: sqlite3.Connection, conversation_id: str
) -> None:
    """Update the activity timestamp after adding messages."""
    connection.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
