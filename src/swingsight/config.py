from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "models_dir": "models",
        "uploads_dir": "uploads",
        "outputs_dir": "outputs",
        "reports_dir": "reports",
    }
}


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml if present, otherwise fall back to config.example.yaml."""
    config_path = Path(path)
    if config_path.exists():
        return _read_yaml(config_path)

    example_path = Path("config.example.yaml")
    if example_path.exists():
        return _read_yaml(example_path)

    return DEFAULT_CONFIG.copy()


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
