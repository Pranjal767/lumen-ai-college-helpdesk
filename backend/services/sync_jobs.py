"""Persistent SQLite-backed synchronization jobs and overlap protection."""

import json
import logging
import os
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.database.db import get_connection, get_database_path, initialize_database
from backend.database.models import get_source, list_sources
from backend.services.sync import (
    DisabledSourceError,
    NoEnabledSourcesError,
    SourceSyncSummary,
    SyncRunSummary,
    sync_sources,
)

logger = logging.getLogger(__name__)
_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lumen-sync")
_SUBMITTED_RUNS: set[str] = set()
_SUBMITTED_RUNS_LOCK = Lock()


class ActiveSyncConflictError(RuntimeError):
    """Raised when one or more requested sources are already being synchronized."""


class SyncQueueError(RuntimeError):
    """Raised when a persistent job cannot be handed to the background executor."""


def enqueue_sync(
    *,
    source_id: int | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist a sync job, lock its sources, and submit it to a worker thread."""
    if source_id is not None and (
        isinstance(source_id, bool) or not isinstance(source_id, int) or source_id < 1
    ):
        raise ValueError("source_id must be a positive integer.")

    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    run_id = uuid4().hex
    source_ids: list[int] = []
    try:
        with get_connection(resolved_database_path) as connection:
            if source_id is None:
                source_ids = [
                    int(source["id"])
                    for source in list_sources(connection)
                    if bool(source["enabled"])
                ]
                if not source_ids:
                    raise NoEnabledSourcesError(
                        "There are no enabled sources to synchronize."
                    )
            else:
                source = get_source(connection, source_id)
                if source is None:
                    raise LookupError(f"Source {source_id} is not registered.")
                if not bool(source["enabled"]):
                    raise DisabledSourceError(f"Source {source_id} is disabled.")
                source_ids = [source_id]

            connection.execute(
                """
                INSERT INTO sync_runs (
                    id, requested_source_id, source_ids_json, status, created_at
                )
                VALUES (?, ?, ?, 'queued', ?)
                """,
                (
                    run_id,
                    source_id,
                    json.dumps(source_ids, separators=(",", ":")),
                    _utc_now(),
                ),
            )
            connection.executemany(
                "INSERT INTO sync_run_locks (source_id, run_id) VALUES (?, ?)",
                [(locked_source_id, run_id) for locked_source_id in source_ids],
            )
    except sqlite3.IntegrityError as error:
        raise ActiveSyncConflictError(
            "A sync job is already queued or running for one or more sources."
        ) from error

    try:
        _submit_sync_job(run_id, source_ids, resolved_database_path)
    except RuntimeError as error:
        _mark_failed(run_id, resolved_database_path, str(error))
        _release_locks(run_id, resolved_database_path)
        raise SyncQueueError("The sync job could not be started.") from error

    job = get_sync_run(run_id, database_path=resolved_database_path)
    if job is None:
        raise RuntimeError("The queued sync job could not be retrieved.")
    return job


def resume_pending_sync_jobs(
    *,
    database_path: str | Path | None = None,
) -> int:
    """Submit persisted queued jobs not currently active in this process."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    recover_interrupted_sync_jobs(database_path=resolved_database_path)
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, source_ids_json
            FROM sync_runs
            WHERE status = 'queued'
            ORDER BY created_at, id
            """
        ).fetchall()

    submitted = 0
    for row in rows:
        source_ids = json.loads(row["source_ids_json"])
        if _submit_sync_job(row["id"], source_ids, resolved_database_path):
            submitted += 1
    return submitted


def recover_interrupted_sync_jobs(
    *,
    database_path: str | Path | None = None,
) -> int:
    """Mark runs from exited worker processes failed and release their locks."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, worker_pid
            FROM sync_runs
            WHERE status = 'running'
            """
        ).fetchall()

    recovered = 0
    for row in rows:
        worker_pid = row["worker_pid"]
        if worker_pid is not None and _is_process_running(int(worker_pid)):
            continue
        message = "The worker process stopped before the sync job completed."
        report = json.dumps(
            {"status": "failed", "error": message},
            separators=(",", ":"),
        )
        with get_connection(resolved_database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE sync_runs
                SET status = 'failed', completed_at = ?, report_json = ?,
                    error_message = ?
                WHERE id = ? AND status = 'running' AND worker_pid IS ?
                """,
                (_utc_now(), report, message, row["id"], worker_pid),
            )
            if cursor.rowcount:
                connection.execute(
                    "DELETE FROM sync_run_locks WHERE run_id = ?",
                    (row["id"],),
                )
                recovered += 1
    return recovered


def get_sync_run(
    run_id: str,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one synchronization job's current persisted status."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        row = connection.execute(
            """
            SELECT id, requested_source_id, source_ids_json, status, created_at,
                   started_at, completed_at, pages_discovered,
                   documents_discovered, documents_processed,
                   documents_unchanged, chunks_added, index_status,
                   report_json, error_message
            FROM sync_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
    return _serialize_sync_run(row) if row is not None else None


def list_sync_runs(
    *,
    limit: int = 20,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the newest synchronization jobs, capped to 100 records."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, requested_source_id, source_ids_json, status, created_at,
                   started_at, completed_at, pages_discovered,
                   documents_discovered, documents_processed,
                   documents_unchanged, chunks_added, index_status,
                   report_json, error_message
            FROM sync_runs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_serialize_sync_run(row) for row in rows]


def _execute_sync_job(
    run_id: str,
    source_ids: list[int],
    database_path: Path,
) -> None:
    try:
        claimed = _mark_running(run_id, database_path)
    except Exception:
        logger.exception("Sync job %s could not be claimed.", run_id)
        raise
    if not claimed:
        return

    progress_by_source: dict[int, dict[str, Any]] = {}

    def save_progress(source: SourceSyncSummary) -> None:
        progress_by_source[source.source_id] = asdict(source)
        _persist_sync_progress(
            run_id,
            list(progress_by_source.values()),
            database_path,
        )

    def save_index_status(status: str) -> None:
        _persist_sync_progress(
            run_id,
            list(progress_by_source.values()),
            database_path,
            index_status=status,
        )

    try:
        report = sync_sources(
            source_ids=source_ids,
            database_path=database_path,
            progress_callback=save_progress,
            index_status_callback=save_index_status,
        )
        _mark_completed(run_id, report, database_path)
    except Exception as error:
        logger.exception("Sync job %s failed.", run_id)
        _mark_failed(run_id, database_path, str(error))
        raise
    finally:
        _release_locks(run_id, database_path)


def _submit_sync_job(
    run_id: str,
    source_ids: list[int],
    database_path: Path,
) -> bool:
    with _SUBMITTED_RUNS_LOCK:
        if run_id in _SUBMITTED_RUNS:
            return False
        _SUBMITTED_RUNS.add(run_id)
    try:
        future = _SYNC_EXECUTOR.submit(
            _execute_sync_job,
            run_id,
            source_ids,
            database_path,
        )
    except RuntimeError:
        with _SUBMITTED_RUNS_LOCK:
            _SUBMITTED_RUNS.discard(run_id)
        raise
    future.add_done_callback(partial(_forget_submitted_run, run_id))
    return True


def _forget_submitted_run(run_id: str, future: Future[None]) -> None:
    if future.cancelled():
        logger.warning("Queued sync job %s was cancelled.", run_id)
    with _SUBMITTED_RUNS_LOCK:
        _SUBMITTED_RUNS.discard(run_id)


def _mark_running(run_id: str, database_path: Path) -> bool:
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE sync_runs
            SET status = 'running', worker_pid = ?, started_at = ?,
                index_status = 'pending', report_json = ?
            WHERE id = ? AND status = 'queued'
            """,
            (
                os.getpid(),
                _utc_now(),
                json.dumps(
                    {"status": "running", "sources": [], "index_status": "pending"},
                    separators=(",", ":"),
                ),
                run_id,
            ),
        )
    return cursor.rowcount == 1


def _is_process_running(process_id: int) -> bool:
    if process_id < 1:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if getattr(error, "winerror", None) == 87:
            return False
        raise
    return True


def _persist_sync_progress(
    run_id: str,
    source_reports: list[dict[str, Any]],
    database_path: Path,
    *,
    index_status: str = "pending",
) -> None:
    totals = {
        "pages_discovered": sum(item["pages_discovered"] for item in source_reports),
        "documents_discovered": sum(
            item["documents_discovered"] for item in source_reports
        ),
        "documents_processed": sum(
            item["documents_processed"] for item in source_reports
        ),
        "documents_unchanged": sum(
            item["documents_unchanged"] for item in source_reports
        ),
        "chunks_added": sum(item["chunks_added"] for item in source_reports),
    }
    report = {
        "status": "running",
        "sources": source_reports,
        "index_status": index_status,
        "index": None,
        "errors": [],
    }
    with get_connection(database_path) as connection:
        connection.execute(
            """
            UPDATE sync_runs
            SET pages_discovered = ?, documents_discovered = ?,
                documents_processed = ?, documents_unchanged = ?,
                chunks_added = ?, index_status = ?, report_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                totals["pages_discovered"],
                totals["documents_discovered"],
                totals["documents_processed"],
                totals["documents_unchanged"],
                totals["chunks_added"],
                index_status,
                json.dumps(report, separators=(",", ":")),
                run_id,
            ),
        )


def _mark_completed(
    run_id: str,
    report: SyncRunSummary,
    database_path: Path,
) -> None:
    result = report.to_dict()
    source_reports = result["sources"]
    totals = {
        "pages_discovered": sum(item["pages_discovered"] for item in source_reports),
        "documents_discovered": sum(
            item["documents_discovered"] for item in source_reports
        ),
        "documents_processed": sum(
            item["documents_processed"] for item in source_reports
        ),
        "documents_unchanged": sum(
            item["documents_unchanged"] for item in source_reports
        ),
        "chunks_added": sum(item["chunks_added"] for item in source_reports),
    }
    with get_connection(database_path) as connection:
        connection.execute(
            """
            UPDATE sync_runs
            SET status = ?, completed_at = ?, pages_discovered = ?,
                documents_discovered = ?, documents_processed = ?,
                documents_unchanged = ?, chunks_added = ?, index_status = ?,
                report_json = ?, error_message = NULL
            WHERE id = ?
            """,
            (
                report.status,
                report.completed_at,
                totals["pages_discovered"],
                totals["documents_discovered"],
                totals["documents_processed"],
                totals["documents_unchanged"],
                totals["chunks_added"],
                report.index_status,
                json.dumps(result, separators=(",", ":")),
                run_id,
            ),
        )


def _mark_failed(run_id: str, database_path: Path, message: str) -> None:
    completed_at = _utc_now()
    report = json.dumps(
        {"status": "failed", "error": message},
        separators=(",", ":"),
    )
    with get_connection(database_path) as connection:
        connection.execute(
            """
            UPDATE sync_runs
            SET status = 'failed', completed_at = ?, report_json = ?,
                error_message = ?
            WHERE id = ?
            """,
            (completed_at, report, message, run_id),
        )


def _release_locks(run_id: str, database_path: Path) -> None:
    with get_connection(database_path) as connection:
        connection.execute(
            "DELETE FROM sync_run_locks WHERE run_id = ?",
            (run_id,),
        )


def _serialize_sync_run(row: sqlite3.Row) -> dict[str, Any]:
    report = json.loads(row["report_json"]) if row["report_json"] else None
    return {
        "id": row["id"],
        "requested_source_id": row["requested_source_id"],
        "source_ids": json.loads(row["source_ids_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "pages_discovered": row["pages_discovered"],
        "documents_discovered": row["documents_discovered"],
        "documents_processed": row["documents_processed"],
        "documents_unchanged": row["documents_unchanged"],
        "chunks_added": row["chunks_added"],
        "index_status": row["index_status"],
        "report": report,
        "error": row["error_message"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
