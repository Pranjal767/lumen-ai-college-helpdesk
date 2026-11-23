"""Tests for APScheduler configuration and startup."""

import pytest

from backend.scheduler import run_scheduler


class FakeScheduler:
    def __init__(self, *, timezone):
        self.timezone = timezone
        self.jobs = []
        self.started = False

    def add_job(self, function, **options):
        self.jobs.append((function, options))

    def start(self):
        self.started = True


def test_scheduler_registers_periodic_sync_and_queue_recovery(monkeypatch):
    scheduler = FakeScheduler(timezone="UTC")

    def create_scheduler(**options):
        assert options["timezone"] == "UTC"
        return scheduler

    monkeypatch.setattr("backend.scheduler.BlockingScheduler", create_scheduler)
    monkeypatch.setattr("backend.scheduler.load_dotenv", lambda: False)
    monkeypatch.setenv("SYNC_INTERVAL_HOURS", "12")

    run_scheduler()

    jobs_by_id = {options["id"]: options for _, options in scheduler.jobs}
    assert scheduler.started is True
    assert scheduler.timezone == "UTC"
    assert jobs_by_id["lumen-source-sync"]["hours"] == 12
    assert jobs_by_id["lumen-source-sync"]["max_instances"] == 1
    assert jobs_by_id["lumen-resume-queued-syncs"]["seconds"] == 15


@pytest.mark.parametrize("value", ["not-an-integer", "0", "-1"])
def test_scheduler_rejects_invalid_interval(monkeypatch, value):
    monkeypatch.setattr("backend.scheduler.load_dotenv", lambda: False)
    monkeypatch.setenv("SYNC_INTERVAL_HOURS", value)

    with pytest.raises(ValueError, match="SYNC_INTERVAL_HOURS"):
        run_scheduler()
