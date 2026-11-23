"""Business logic for administering approved college sources."""

import os
import sqlite3
from pathlib import Path
from typing import Any

from backend.database.db import get_connection, get_database_path, initialize_database
from backend.database.models import (
    get_source,
    insert_source,
    list_sources,
    update_source_enabled,
)
from backend.services.crawler import CrawlError, normalize_allowed_url

MAX_SOURCE_NAME_LENGTH = 200
MAX_CATEGORY_LENGTH = 100
MAX_CRAWL_DEPTH = 5


class SourceInputError(ValueError):
    """Raised when an admin source payload is invalid."""


class DuplicateSourceError(ValueError):
    """Raised when an approved source URL is already registered."""


def list_registered_sources(
    *, database_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return source records in API-ready form."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        rows = list_sources(connection)
    return [_serialize_source(row) for row in rows]


def register_source(
    *,
    name: str,
    url: str,
    category: str | None,
    crawl_depth: int,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one source on the configured approved domain."""
    normalized_name = _validate_name(name)
    normalized_category = _validate_category(category)
    if isinstance(crawl_depth, bool) or not isinstance(crawl_depth, int):
        raise SourceInputError("crawl_depth must be an integer.")
    if not 0 <= crawl_depth <= MAX_CRAWL_DEPTH:
        raise SourceInputError(
            f"crawl_depth must be between 0 and {MAX_CRAWL_DEPTH}."
        )

    allowed_domain = os.environ.get("COLLEGE_DOMAIN", "").strip()
    if not allowed_domain:
        raise RuntimeError("COLLEGE_DOMAIN is not configured.")
    try:
        normalized_url = normalize_allowed_url(url, allowed_domain)
    except (AttributeError, CrawlError) as error:
        raise SourceInputError(
            "url must be an HTTPS address on the configured college domain."
        ) from error

    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    try:
        with get_connection(resolved_database_path) as connection:
            source_id = insert_source(
                connection,
                name=normalized_name,
                url=normalized_url,
                source_type="website",
                category=normalized_category,
                crawl_depth=crawl_depth,
            )
            row = get_source(connection, source_id)
            if row is None:
                raise RuntimeError("The new source could not be retrieved.")
    except sqlite3.IntegrityError as error:
        raise DuplicateSourceError("That source URL is already registered.") from error
    return _serialize_source(row)


def set_source_enabled(
    source_id: int,
    enabled: bool,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Enable or disable a registered source; return None if it is missing."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        if not update_source_enabled(connection, source_id, enabled):
            return None
        row = get_source(connection, source_id)
        if row is None:
            raise RuntimeError("The updated source could not be retrieved.")
    return _serialize_source(row)


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise SourceInputError("name must be a non-empty string.")
    normalized_name = name.strip()
    if len(normalized_name) > MAX_SOURCE_NAME_LENGTH:
        raise SourceInputError(
            f"name must be at most {MAX_SOURCE_NAME_LENGTH} characters."
        )
    return normalized_name


def _validate_category(category: str | None) -> str | None:
    if category is None:
        return None
    if not isinstance(category, str):
        raise SourceInputError("category must be a string or null.")
    normalized_category = category.strip()
    if len(normalized_category) > MAX_CATEGORY_LENGTH:
        raise SourceInputError(
            f"category must be at most {MAX_CATEGORY_LENGTH} characters."
        )
    return normalized_category or None


def _serialize_source(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "url": row["url"],
        "source_type": row["source_type"],
        "category": row["category"],
        "enabled": bool(row["enabled"]),
        "crawl_depth": int(row["crawl_depth"]),
        "last_synced_at": row["last_synced_at"],
        "created_at": row["created_at"],
    }
