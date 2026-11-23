"""Tests for source registry persistence."""

from backend.database.db import get_connection, initialize_database
from backend.database.models import insert_source, list_sources
from backend.database.seed_sources import SEED_SOURCES


def test_database_initialization_is_repeatable(tmp_path):
    database_path = tmp_path / "nested" / "test.db"

    initialize_database(database_path)
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sources'"
        ).fetchall()

    assert len(tables) == 1


def test_source_can_be_inserted_and_retrieved(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        source_id = insert_source(
            connection,
            name="Exam notices",
            url="https://bkbck.edu.in/examination/circular-notifications/",
            category="examinations",
        )
        sources = list_sources(connection)

    assert len(sources) == 1
    assert sources[0]["id"] == source_id
    assert sources[0]["name"] == "Exam notices"
    assert sources[0]["enabled"] == 1


def test_seed_sources_are_inserted_only_once(tmp_path, monkeypatch):
    database_path = tmp_path / "seed.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))

    from backend.database.seed_sources import seed_sources

    seed_sources()
    seed_sources()

    with get_connection(database_path) as connection:
        sources = list_sources(connection)

    assert len(sources) == len(SEED_SOURCES)
    assert [source["url"] for source in sources] == [
        seed[1] for seed in SEED_SOURCES
    ]
