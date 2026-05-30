from __future__ import annotations

from flask import Flask

from swingsight.config import load_config
from webapp.routes.dashboard import dashboard_bp


def create_app() -> Flask:
    """Create a local-first Flask app for SwingSight dashboard."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SWINGSIGHT_CONFIG"] = load_config()
    app.register_blueprint(dashboard_bp)
    return app
