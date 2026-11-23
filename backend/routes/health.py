"""Health check endpoint."""

from flask import Blueprint, jsonify

health_blueprint = Blueprint("health", __name__)


@health_blueprint.get("/health")
def get_health():
    """Return basic application health information."""
    return jsonify(
        {
            "status": "healthy",
            "application": "Lumen AI",
            "version": "1.0.0",
        }
    )
