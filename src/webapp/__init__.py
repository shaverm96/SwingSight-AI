from __future__ import annotations

from pathlib import Path

from flask import Flask

from backend.services.coaching_engine import CoachingEngine
from backend.services.model_manager import ModelManager
from swingsight.config import load_config
from webapp.routes.dashboard import dashboard_bp


def create_app() -> Flask:
    """Create a local-first Flask app for SwingSight dashboard."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    config = load_config()
    app.config["SWINGSIGHT_CONFIG"] = config

    project_root = Path(app.root_path).resolve().parent.parent
    model_manager = ModelManager(project_root)
    model_manager.load_models()
    model_manager.load_metadata()

    app.extensions["swing_runtime"] = {
        "model_manager": model_manager,
        "coaching_engine": CoachingEngine(),
    }

    app.register_blueprint(dashboard_bp)
    return app
