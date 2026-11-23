"""Flask application setup."""

import os
from pathlib import Path

from flask import Flask, render_template, send_from_directory

from backend.routes.admin import admin_blueprint
from backend.routes.chat import chat_blueprint
from backend.routes.health import health_blueprint


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

    @app.get("/")
    def index():
        built_frontend = Path(app.static_folder) / "lumen-dist"
        if (built_frontend / "index.html").is_file():
            return send_from_directory(built_frontend, "index.html")
        return render_template("index.html")

    app.register_blueprint(health_blueprint, url_prefix="/api/v1")
    app.register_blueprint(chat_blueprint, url_prefix="/api/v1")
    app.register_blueprint(admin_blueprint, url_prefix="/api/v1")
    return app
