"""Tests for persistent asynchronous synchronization jobs."""

import threading
import time

import pytest

from backend.database.db import get_connection, initialize_database
from backend.database.models import insert_source
from backend.services.sync import SourceSyncSummary, SyncRunSummary
from backend.services.sync_jobs import (
    ActiveSyncConflictError,
    enqueue_sync,
    get_sync_run,
    list_sync_runs,
)


def create_sources(database_path, count=2):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        return [
            insert_source(
                connection,
                name=f"College source {index}",
                url=f"https://bkbck.edu.in/section-{index}/",
                category="notices",
            )
            for index in range(1, count + 1)
        ]


def make_report(source_ids):
    now = "2026-11-17T00:00:00+00:00"
    sources = [
        SourceSyncSummary(
            source_id=source_id,
            name=f"College source {source_id}",
            url=f"https://bkbck.edu.in/section-{source_id}/",
            status="succeeded",
            pages_discovered=2,
            documents_discovered=2,
            documents_processed=2,
            documents_unchanged=1,
            chunks_added=1,
        )
        for source_id in source_ids
    ]
    return SyncRunSummary(
        status="succeeded",
        started_at=now,
        completed_at=now,
        sources=sources,
        index_status="current",
        index=None,
        errors=[],
    )


def wait_for_terminal_status(database_path, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = get_sync_run(run_id, database_path=database_path)
        if job["status"] in {"succeeded", "partial", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Sync job {run_id} did not finish in time.")


def test_jobs_persist_progress_and_prevent_overlapping_source_runs(
    monkeypatch, tmp_path
):
    database_path = tmp_path / "sync-jobs.db"
    source_ids = create_sources(database_path)
    started_sources = set()
    started = threading.Event()
    release = threading.Event()

    def slow_sync(
        *, source_ids, database_path, progress_callback, index_status_callback
    ):
        assert database_path == tmp_path / "sync-jobs.db"
        started_sources.update(source_ids)
        for source_id in source_ids:
            progress_callback(
                SourceSyncSummary(
                    source_id=source_id,
                    name=f"College source {source_id}",
                    url=f"https://bkbck.edu.in/section-{source_id}/",
                    status="running",
                    pages_discovered=2,
                    documents_discovered=1,
                    documents_processed=1,
                )
            )
        index_status_callback("building")
        if len(started_sources) == 2:
            started.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release the sync worker.")
        return make_report(source_ids)

    monkeypatch.setattr("backend.services.sync_jobs.sync_sources", slow_sync)
    first = enqueue_sync(source_id=source_ids[0], database_path=database_path)
    second = enqueue_sync(source_id=source_ids[1], database_path=database_path)

    assert started.wait(2)
    assert first["id"] != second["id"]
    assert first["status"] in {"queued", "running", "succeeded"}
    first_progress = get_sync_run(first["id"], database_path=database_path)
    assert first_progress["status"] == "running"
    assert first_progress["pages_discovered"] == 2
    assert first_progress["documents_processed"] == 1
    assert first_progress["index_status"] == "building"
    with pytest.raises(ActiveSyncConflictError, match="already queued or running"):
        enqueue_sync(source_id=source_ids[0], database_path=database_path)
    with pytest.raises(ActiveSyncConflictError, match="already queued or running"):
        enqueue_sync(database_path=database_path)

    release.set()
    first_result = wait_for_terminal_status(database_path, first["id"])
    second_result = wait_for_terminal_status(database_path, second["id"])
    history = list_sync_runs(database_path=database_path)

    assert first_result["status"] == "succeeded"
    assert first_result["documents_processed"] == 2
    assert first_result["documents_unchanged"] == 1
    assert first_result["chunks_added"] == 1
    assert second_result["status"] == "succeeded"
    assert len(history) == 2


def test_enqueue_rejects_disabled_or_unknown_sources(tmp_path):
    database_path = tmp_path / "sync-jobs.db"
    source_id = create_sources(database_path, count=1)[0]
    with get_connection(database_path) as connection:
        connection.execute("UPDATE sources SET enabled = 0 WHERE id = ?", (source_id,))

    with pytest.raises(ValueError, match="disabled"):
        enqueue_sync(source_id=source_id, database_path=database_path)
    with pytest.raises(LookupError, match="not registered"):
        enqueue_sync(source_id=999, database_path=database_path)


def test_enqueue_rejects_runs_when_no_sources_are_enabled(tmp_path):
    database_path = tmp_path / "sync-jobs.db"
    source_id = create_sources(database_path, count=1)[0]
    with get_connection(database_path) as connection:
        connection.execute("UPDATE sources SET enabled = 0 WHERE id = ?", (source_id,))

    with pytest.raises(ValueError, match="no enabled sources"):
        enqueue_sync(database_path=database_path)


def test_sync_run_history_limit_is_validated(tmp_path):
    database_path = tmp_path / "sync-jobs.db"
    initialize_database(database_path)

    with pytest.raises(ValueError, match="between 1 and 100"):
        list_sync_runs(limit=0, database_path=database_path)


def test_scheduler_can_resume_a_persisted_queued_run(monkeypatch, tmp_path):
    database_path = tmp_path / "sync-jobs.db"
    source_id = create_sources(database_path, count=1)[0]
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sync_runs (
                id, requested_source_id, source_ids_json, status, created_at
            )
            VALUES ('queued-after-restart', ?, ?, 'queued', ?)
            """,
            (
                source_id,
                f"[{source_id}]",
                "2026-11-17T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO sync_run_locks (source_id, run_id) VALUES (?, ?)",
            (source_id, "queued-after-restart"),
        )

    def fake_sync(
        *, source_ids, database_path, progress_callback, index_status_callback
    ):
        assert database_path == tmp_path / "sync-jobs.db"
        progress_callback(
            SourceSyncSummary(
                source_id=source_ids[0],
                name=f"College source {source_ids[0]}",
                url=f"https://bkbck.edu.in/section-{source_ids[0]}/",
                status="running",
            )
        )
        index_status_callback("current")
        return make_report(source_ids)

    monkeypatch.setattr("backend.services.sync_jobs.sync_sources", fake_sync)
    from backend.services.sync_jobs import resume_pending_sync_jobs

    submitted = resume_pending_sync_jobs(database_path=database_path)
    result = wait_for_terminal_status(database_path, "queued-after-restart")

    assert submitted == 1
    assert result["status"] == "succeeded"


def test_dead_worker_runs_are_marked_failed_and_release_source_locks(
    monkeypatch, tmp_path
):
    database_path = tmp_path / "sync-jobs.db"
    source_id = create_sources(database_path, count=1)[0]
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sync_runs (
                id, requested_source_id, source_ids_json, status, worker_pid,
                created_at, started_at
            )
            VALUES ('interrupted-run', ?, ?, 'running', 999999,
                    '2026-11-17T00:00:00+00:00', '2026-11-17T00:00:00+00:00')
            """,
            (source_id, f"[{source_id}]"),
        )
        connection.execute(
            "INSERT INTO sync_run_locks (source_id, run_id) VALUES (?, ?)",
            (source_id, "interrupted-run"),
        )
    def report_dead_process(process_id):
        assert process_id == 999999
        return False

    monkeypatch.setattr(
        "backend.services.sync_jobs._is_process_running",
        report_dead_process,
    )
    from backend.services.sync_jobs import recover_interrupted_sync_jobs

    recovered = recover_interrupted_sync_jobs(database_path=database_path)
    job = get_sync_run("interrupted-run", database_path=database_path)
    with get_connection(database_path) as connection:
        locks = connection.execute("SELECT * FROM sync_run_locks").fetchall()

    assert recovered == 1
    assert job["status"] == "failed"
    assert job["error"] == "The worker process stopped before the sync job completed."
    assert locks == []


def test_process_liveness_treats_windows_invalid_pid_as_exited(monkeypatch):
    from backend.services.sync_jobs import _is_process_running

    invalid_process_error = OSError("process does not exist")
    invalid_process_error.winerror = 87

    def fail_kill(process_id, signal):
        assert process_id == 12345
        assert signal == 0
        raise invalid_process_error

    monkeypatch.setattr("backend.services.sync_jobs.os.kill", fail_kill)

    assert _is_process_running(12345) is False
