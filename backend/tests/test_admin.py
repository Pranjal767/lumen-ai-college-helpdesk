"""Tests for authenticated admin source-management endpoints."""

import pytest

from backend.app import create_app

ADMIN_HEADERS = {"Authorization": "Bearer test-admin-token"}


def create_admin_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-token")
    monkeypatch.setenv("COLLEGE_DOMAIN", "bkbck.edu.in")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "admin.db"))
    return create_app().test_client()


def test_admin_routes_require_configured_bearer_key(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    client = create_app().test_client()

    unconfigured = client.get("/api/v1/admin/sources")
    assert unconfigured.status_code == 503
    assert unconfigured.get_json()["error"]["code"] == "ADMIN_NOT_CONFIGURED"

    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-token")
    unauthorized = client.get("/api/v1/admin/sources")
    assert unauthorized.status_code == 401
    assert unauthorized.get_json()["error"]["code"] == "ADMIN_UNAUTHORIZED"

    wrong_key = client.get(
        "/api/v1/admin/sources",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert wrong_key.status_code == 401


def test_admin_can_list_and_register_approved_sources(monkeypatch, tmp_path):
    client = create_admin_client(monkeypatch, tmp_path)

    empty_response = client.get(
        "/api/v1/admin/sources",
        headers=ADMIN_HEADERS,
    )
    assert empty_response.status_code == 200
    assert empty_response.get_json() == {"success": True, "sources": []}

    response = client.post(
        "/api/v1/admin/sources",
        headers=ADMIN_HEADERS,
        json={
            "name": "Exam notices",
            "url": "https://bkbck.edu.in/notices/#current",
            "category": "examinations",
            "crawl_depth": 2,
        },
    )

    assert response.status_code == 201
    source = response.get_json()["source"]
    assert {key: value for key, value in source.items() if key != "created_at"} == {
        "id": 1,
        "name": "Exam notices",
        "url": "https://bkbck.edu.in/notices/",
        "source_type": "website",
        "category": "examinations",
        "enabled": True,
        "crawl_depth": 2,
        "last_synced_at": None,
    }
    assert isinstance(source["created_at"], str)


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        (None, 400),
        ({"name": "Notices"}, 400),
        (
            {"name": "External", "url": "https://example.com/"},
            400,
        ),
        (
            {"name": "Unsafe", "url": "http://bkbck.edu.in/"},
            400,
        ),
        (
            {"name": "Notices", "url": "https://bkbck.edu.in/", "crawl_depth": True},
            400,
        ),
        (
            {
                "name": "Notices",
                "url": "https://bkbck.edu.in/",
                "unexpected": "field",
            },
            400,
        ),
    ],
)
def test_admin_rejects_invalid_source_payloads(
    monkeypatch, tmp_path, payload, status_code
):
    client = create_admin_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/admin/sources",
        headers=ADMIN_HEADERS,
        json=payload,
    )

    assert response.status_code == status_code
    assert response.get_json()["success"] is False


def test_duplicate_source_url_returns_conflict(monkeypatch, tmp_path):
    client = create_admin_client(monkeypatch, tmp_path)
    payload = {"name": "Notices", "url": "https://bkbck.edu.in/notices/"}

    first = client.post(
        "/api/v1/admin/sources", headers=ADMIN_HEADERS, json=payload
    )
    duplicate = client.post(
        "/api/v1/admin/sources", headers=ADMIN_HEADERS, json=payload
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.get_json()["error"]["code"] == "SOURCE_EXISTS"


def test_admin_can_disable_and_reenable_a_source(monkeypatch, tmp_path):
    client = create_admin_client(monkeypatch, tmp_path)
    created = client.post(
        "/api/v1/admin/sources",
        headers=ADMIN_HEADERS,
        json={"name": "Notices", "url": "https://bkbck.edu.in/notices/"},
    )
    source_id = created.get_json()["source"]["id"]

    disabled = client.patch(
        f"/api/v1/admin/sources/{source_id}",
        headers=ADMIN_HEADERS,
        json={"enabled": False},
    )
    enabled = client.patch(
        f"/api/v1/admin/sources/{source_id}",
        headers=ADMIN_HEADERS,
        json={"enabled": True},
    )
    invalid = client.patch(
        f"/api/v1/admin/sources/{source_id}",
        headers=ADMIN_HEADERS,
        json={"enabled": 1},
    )
    missing = client.patch(
        "/api/v1/admin/sources/999",
        headers=ADMIN_HEADERS,
        json={"enabled": False},
    )

    assert disabled.status_code == 200
    assert disabled.get_json()["source"]["enabled"] is False
    assert enabled.status_code == 200
    assert enabled.get_json()["source"]["enabled"] is True
    assert invalid.status_code == 400
    assert missing.status_code == 404


def test_admin_sync_endpoint_queues_job_and_exposes_persisted_status(
    monkeypatch, tmp_path
):
    client = create_admin_client(monkeypatch, tmp_path)
    queued_job = {
        "id": "run-1",
        "status": "queued",
        "source_ids": [1],
    }
    completed_job = {
        "id": "run-1",
        "status": "succeeded",
        "source_ids": [1],
        "documents_processed": 1,
    }

    def fake_enqueue_sync(**kwargs):
        assert kwargs["source_id"] is None
        return queued_job

    def fake_get_sync_run(run_id):
        return completed_job if run_id == "run-1" else None

    monkeypatch.setattr("backend.routes.admin.enqueue_sync", fake_enqueue_sync)
    monkeypatch.setattr("backend.routes.admin.get_sync_run", fake_get_sync_run)
    monkeypatch.setattr("backend.routes.admin.list_sync_runs", lambda: [completed_job])

    response = client.post("/api/v1/admin/sync", headers=ADMIN_HEADERS, json={})
    status = client.get(
        "/api/v1/admin/sync/run-1",
        headers=ADMIN_HEADERS,
    )
    history = client.get("/api/v1/admin/sync", headers=ADMIN_HEADERS)
    invalid = client.post(
        "/api/v1/admin/sync",
        headers=ADMIN_HEADERS,
        json={"url": "https://example.com/"},
    )

    assert response.status_code == 202
    assert response.get_json() == {
        "success": True,
        "job": queued_job,
    }
    assert status.status_code == 200
    assert status.get_json()["job"] == completed_job
    assert history.status_code == 200
    assert history.get_json()["jobs"] == [completed_job]
    assert invalid.status_code == 400
