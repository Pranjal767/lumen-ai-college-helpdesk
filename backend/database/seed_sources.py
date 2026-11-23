"""Seed the approved college website entry points."""

from backend.database.db import get_connection, initialize_database
from backend.database.models import list_sources

SEED_SOURCES = (
    ("College homepage", "https://bkbck.edu.in/", "general"),
    ("College notices", "https://bkbck.edu.in/notices/", "notices"),
    (
        "Examination circulars",
        "https://bkbck.edu.in/examination/circular-notifications/",
        "examinations",
    ),
    (
        "Examination timetable",
        "https://bkbck.edu.in/examination/exam-timetable/",
        "examinations",
    ),
)


def seed_sources() -> None:
    """Insert the approved seed URLs without duplicating existing records."""
    initialize_database()
    with get_connection() as connection:
        for name, url, category in SEED_SOURCES:
            connection.execute(
                """
                INSERT OR IGNORE INTO sources (name, url, source_type, category)
                VALUES (?, ?, 'website', ?)
                """,
                (name, url, category),
            )
        sources = list_sources(connection)
    print("Source registry ready:")
    for source in sources:
        print(f"- {source['name']}: {source['url']}")


if __name__ == "__main__":
    seed_sources()
