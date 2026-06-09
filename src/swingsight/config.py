from __future__ import annotations

import os
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


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned = value.strip().strip('"').strip("'")
        os.environ[key] = cleaned


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml if present, otherwise fall back to config.example.yaml."""
    load_dotenv()
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
