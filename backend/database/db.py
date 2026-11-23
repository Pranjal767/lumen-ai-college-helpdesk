"""SQLite connection and schema initialization helpers."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "lumen.db"


def get_database_path() -> Path:
    """Return the configured database path."""
    configured_path = os.environ.get("DATABASE_PATH")
    if configured_path:
        return Path(configured_path)
    return DEFAULT_DATABASE_PATH


@contextmanager
def get_connection(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that commits on success and rolls back on error."""
    path = Path(database_path) if database_path is not None else get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: str | Path | None = None) -> None:
    """Create the initial schema if it does not already exist."""
    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                category TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                crawl_depth INTEGER NOT NULL DEFAULT 1,
                last_synced_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES sources(id),
                title TEXT NOT NULL,
                source_url TEXT NOT NULL UNIQUE,
                document_type TEXT NOT NULL,
                category TEXT,
                content_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        document_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(documents)")
        }
        if "source_type" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN source_type TEXT NOT NULL DEFAULT 'official'"
            )
        if "record_id" not in document_columns:
            connection.execute("ALTER TABLE documents ADD COLUMN record_id TEXT")
        if "metadata_json" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                version_number INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                extraction_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (document_id, version_number)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_version_id INTEGER NOT NULL
                    REFERENCES document_versions(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                token_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (document_version_id, chunk_index)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_session_updated
            ON conversations (session_id, updated_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages (conversation_id, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                generation INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO chunk_state (id, generation) VALUES (1, 0)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_runs (
                id TEXT PRIMARY KEY,
                requested_source_id INTEGER REFERENCES sources(id),
                source_ids_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('queued', 'running', 'succeeded', 'partial', 'failed')
                ),
                worker_pid INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                pages_discovered INTEGER NOT NULL DEFAULT 0,
                documents_discovered INTEGER NOT NULL DEFAULT 0,
                documents_processed INTEGER NOT NULL DEFAULT 0,
                documents_unchanged INTEGER NOT NULL DEFAULT 0,
                chunks_added INTEGER NOT NULL DEFAULT 0,
                index_status TEXT,
                report_json TEXT,
                error_message TEXT
            )
            """
        )
        sync_run_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sync_runs)")
        }
        if "worker_pid" not in sync_run_columns:
            connection.execute("ALTER TABLE sync_runs ADD COLUMN worker_pid INTEGER")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_runs_created_at
            ON sync_runs (created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_run_locks (
                source_id INTEGER PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
                run_id TEXT NOT NULL REFERENCES sync_runs(id) ON DELETE CASCADE
            )
            """
        )
