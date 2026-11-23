"""Dedicated APScheduler process for periodic source synchronization."""

import logging
import os
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from backend.services.sync import NoEnabledSourcesError
from backend.services.sync_jobs import (
    ActiveSyncConflictError,
    enqueue_sync,
    resume_pending_sync_jobs,
)

logger = logging.getLogger(__name__)


def _enqueue_scheduled_sync() -> None:
    try:
        job = enqueue_sync()
    except ActiveSyncConflictError:
        logger.info("Scheduled sync skipped because one or more sources are active.")
    except NoEnabledSourcesError:
        logger.warning("Scheduled sync skipped because there are no enabled sources.")
    else:
        logger.info("Scheduled sync queued with job ID %s.", job["id"])


def _resume_queued_syncs() -> None:
    resumed = resume_pending_sync_jobs()
    if resumed:
        logger.info("Resumed %s queued synchronization job(s).", resumed)


def run_scheduler() -> None:
    """Run the dedicated blocking scheduler process."""
    load_dotenv()
    raw_interval = os.environ.get("SYNC_INTERVAL_HOURS", "24")
    try:
        interval_hours = int(raw_interval)
    except ValueError as error:
        raise ValueError("SYNC_INTERVAL_HOURS must be an integer.") from error
    if interval_hours < 1:
        raise ValueError("SYNC_INTERVAL_HOURS must be at least one.")

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _enqueue_scheduled_sync,
        trigger="interval",
        hours=interval_hours,
        id="lumen-source-sync",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(hours=interval_hours),
    )
    scheduler.add_job(
        _resume_queued_syncs,
        trigger="interval",
        seconds=15,
        id="lumen-resume-queued-syncs",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    logger.info(
        "Source scheduler started; sync interval is %s hour(s).",
        interval_hours,
    )
    scheduler.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_scheduler()
