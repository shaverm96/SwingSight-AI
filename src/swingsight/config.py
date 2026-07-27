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
        # Crops toward a detected hand (using hand_landmarker.task, already
        # bundled with the project) before the five-way classifier runs,
        # since that classifier is trained on isolated club-head photos
        # rather than full "person holding a club" frames. Unlike an earlier
        # pose-based version, this does not require a visible body or golf
        # stance -- a close-up of just a hand is enough. See
        # swingsight.club_localization.
        "enabled": True,
        "hand_model_path": "hand_landmarker.task",
        "min_detection_confidence": 0.3,
        "min_presence_confidence": 0.3,
        "crop_scale": 5.5,
        "max_crop_fraction": 0.75,
        "merge_ratio": 1.5,
    },
    "club_recognition": {
        "confirm_threshold": 0.45,
        "five_way_cnn_model_path": "models/trained/club_type_5way.pt",
        "five_way_cnn_min_confidence": 0.45,
        "marking_ocr": {
            "backend": "rapidocr",
            "min_confidence": 0.7,
            "max_image_side": 1600,
            "min_image_side": 900,
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override onto base, without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml if present, otherwise fall back to config.example.yaml.

    Whichever file is loaded is deep-merged over DEFAULT_CONFIG, so a partial
    config.yaml that only overrides a few settings still gets this module's
    documented defaults for everything else. Previously a loaded file fully
    replaced DEFAULT_CONFIG, so a key missing from config.yaml (e.g.
    club_recognition.confirm_threshold) silently fell back to whatever
    hardcoded literal each individual call site happened to use
    (settings.get("confirm_threshold", 0.6)) instead of this module's default.
    """
    load_dotenv()
    config_path = Path(path)
    if config_path.exists():
        return _deep_merge(DEFAULT_CONFIG, _read_yaml(config_path))

    example_path = Path("config.example.yaml")
    if example_path.exists():
        return _deep_merge(DEFAULT_CONFIG, _read_yaml(example_path))

    return DEFAULT_CONFIG.copy()


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
