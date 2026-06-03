from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask

# Prefer the src/ package tree over the repo-root duplicate packages.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swingsight.config import load_config


def create_app() -> Flask:
    """Create a local-first Flask app for SwingSight dashboard."""
    from backend.services.coaching_engine import CoachingEngine
    from backend.services.model_manager import ModelManager
    from webapp.routes.dashboard import dashboard_bp

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    config = load_config()
    app.config["SWINGSIGHT_CONFIG"] = config

    project_root = Path(app.root_path).resolve().parent.parent
    runtime_config = {**config, "project_root": str(project_root)}
    model_manager = ModelManager(runtime_config)
    coaching_engine = CoachingEngine(runtime_config)

    app.extensions["swing_runtime"] = {
        "model_manager": model_manager,
        "coaching_engine": coaching_engine,
    }
    app.extensions["swing_model_manager"] = model_manager
    app.extensions["swing_coaching_engine"] = coaching_engine

    app.register_blueprint(dashboard_bp)
    return app
