"""Authenticated admin API for managing approved source records."""

import hmac
import os

from flask import Blueprint, jsonify, request

from backend.services.admin_sources import (
    DuplicateSourceError,
    SourceInputError,
    list_registered_sources,
    register_source,
    set_source_enabled,
)
from backend.services.sync import (
    DisabledSourceError,
    NoEnabledSourcesError,
)
from backend.services.sync_jobs import (
    ActiveSyncConflictError,
    SyncQueueError,
    enqueue_sync,
    get_sync_run,
    list_sync_runs,
)

admin_blueprint = Blueprint("admin", __name__)


@admin_blueprint.before_request
def require_admin_api_key():
    """Require a separately configured bearer key for every admin route."""
    expected_key = os.environ.get("ADMIN_API_KEY", "").strip()
    if not expected_key:
        return _error_response(
            "ADMIN_NOT_CONFIGURED",
            "Admin API access is not configured.",
            503,
        )

    authorization = request.headers.get("Authorization", "")
    scheme, separator, supplied_key = authorization.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not hmac.compare_digest(
            supplied_key.strip().encode("utf-8"),
            expected_key.encode("utf-8"),
        )
    ):
        return _error_response(
            "ADMIN_UNAUTHORIZED",
            "A valid bearer token is required.",
            401,
        )
    return None


@admin_blueprint.get("/admin/sources")
def get_admin_sources():
    """List configured sources without exposing local filesystem paths."""
    return jsonify({"success": True, "sources": list_registered_sources()})


@admin_blueprint.post("/admin/sources")
def post_admin_source():
    """Register a source only when it belongs to the configured HTTPS domain."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error_response(
            "INVALID_REQUEST",
            "Send a JSON object containing name and url.",
            400,
        )
    allowed_fields = {"name", "url", "category", "crawl_depth"}
    if set(payload) - allowed_fields:
        return _error_response(
            "INVALID_REQUEST",
            "The request contains unsupported fields.",
            400,
        )
    if "name" not in payload or "url" not in payload:
        return _error_response(
            "INVALID_REQUEST",
            "Both name and url are required.",
            400,
        )

    try:
        source = register_source(
            name=payload["name"],
            url=payload["url"],
            category=payload.get("category"),
            crawl_depth=payload.get("crawl_depth", 1),
        )
    except SourceInputError as error:
        return _error_response("INVALID_SOURCE", str(error), 400)
    except DuplicateSourceError as error:
        return _error_response("SOURCE_EXISTS", str(error), 409)
    except RuntimeError as error:
        return _error_response("ADMIN_UNAVAILABLE", str(error), 503)

    return jsonify({"success": True, "source": source}), 201


@admin_blueprint.patch("/admin/sources/<int:source_id>")
def patch_admin_source(source_id: int):
    """Update only a source's enabled state."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {"enabled"}:
        return _error_response(
            "INVALID_REQUEST",
            "Send a JSON object containing only a boolean enabled field.",
            400,
        )
    if not isinstance(payload["enabled"], bool):
        return _error_response(
            "INVALID_REQUEST",
            "enabled must be a boolean.",
            400,
        )

    source = set_source_enabled(source_id, payload["enabled"])
    if source is None:
        return _error_response("SOURCE_NOT_FOUND", "Source not found.", 404)
    return jsonify({"success": True, "source": source})


@admin_blueprint.post("/admin/sync")
def post_admin_sync():
    """Queue synchronization for enabled sources or one requested source."""
    payload = request.get_json(silent=True) if request.data else {}
    if not isinstance(payload, dict) or set(payload) - {"source_id"}:
        return _error_response(
            "INVALID_REQUEST",
            "Send an optional positive integer source_id.",
            400,
        )
    source_id = payload.get("source_id")
    if source_id is not None and (
        isinstance(source_id, bool)
        or not isinstance(source_id, int)
        or source_id < 1
    ):
        return _error_response(
            "INVALID_REQUEST",
            "source_id must be a positive integer.",
            400,
        )

    try:
        job = enqueue_sync(source_id=source_id)
    except LookupError as error:
        return _error_response("SOURCE_NOT_FOUND", str(error), 404)
    except (DisabledSourceError, NoEnabledSourcesError) as error:
        return _error_response("SYNC_NOT_AVAILABLE", str(error), 409)
    except ActiveSyncConflictError as error:
        return _error_response("SYNC_ALREADY_ACTIVE", str(error), 409)
    except SyncQueueError as error:
        return _error_response("SYNC_UNAVAILABLE", str(error), 503)
    except ValueError as error:
        return _error_response("INVALID_REQUEST", str(error), 400)

    return jsonify({"success": True, "job": job}), 202


@admin_blueprint.get("/admin/sync")
def get_admin_sync_runs():
    """List the most recent persistent synchronization jobs."""
    return jsonify({"success": True, "jobs": list_sync_runs()})


@admin_blueprint.get("/admin/sync/<run_id>")
def get_admin_sync_run(run_id: str):
    """Return one synchronization job's current status and final report."""
    job = get_sync_run(run_id)
    if job is None:
        return _error_response("SYNC_NOT_FOUND", "Sync job not found.", 404)
    return (
        jsonify({"success": True, "job": job}),
        200,
    )


def _error_response(code: str, message: str, status_code: int):
    return (
        jsonify(
            {
                "success": False,
                "error": {"code": code, "message": message},
            }
        ),
        status_code,
    )
