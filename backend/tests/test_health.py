"""Tests for the health endpoint."""

from backend.app import create_app


def test_health_returns_application_status():
    client = create_app().test_client()

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "application": "Lumen AI",
        "version": "1.0.0",
    }


def test_app_uses_secret_key_from_environment(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")

    app = create_app()

    assert app.config["SECRET_KEY"] == "test-secret-key"
