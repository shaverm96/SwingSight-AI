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
    },
    "club_localization": {
        # Crops toward the golfer's arms/hands (using yolov8n-pose.pt, already
        # bundled with the project) before the five-way classifier runs, since
        # that classifier is trained on isolated club-head photos rather than
        # full "person holding a club" frames. See swingsight.club_localization.
        "enabled": True,
        "pose_model_path": "yolov8n-pose.pt",
        "min_keypoint_confidence": 0.3,
        "padding_fraction": 0.45,
        "min_padding_scale": 0.6,
    },
    "club_recognition": {
        "confirm_threshold": 0.45,
        "five_way_cnn_model_path": "models/trained/club_type_5way.pt",
        "five_way_cnn_min_confidence": 0.6,
        "marking_ocr": {
            "backend": "rapidocr",
            "min_confidence": 0.7,
            "max_image_side": 1600,
            "enhance_contrast": True,
            "contrast_clip_limit": 2.0,
            "sharpen": False,
            "try_sideways_rotations_on_failure": True,
        },
    },
}


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
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
