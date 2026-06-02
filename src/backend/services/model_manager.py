from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from swingsight.club_recognition import recognize_club_from_frame
from swingsight.metrics import compute_swing_metrics
from swingsight.pose_estimation import run_pose_estimation
from swingsight.visualization import render_annotated_video
from webapp.inference.pipeline_components import check_body_visibility_from_frame
from webapp.utils.storage import ensure_dir

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


LOGGER = logging.getLogger(__name__)

MODEL_FILE_NAMES = {
    "club_classifier": ["club_classifier.pt", "club_classifier.joblib", "club_classifier.pkl"],
    "club_detector": ["club_detector.pt"],
    "swing_classifier": ["swing_classifier.pt", "swing_classifier.joblib", "swing_classifier.pkl"],
    "pose_model": ["pose_model.pt", "yolov8n-pose.pt"],
}


def _to_path(value: object, root: Path) -> Path:
    path = Path(str(value)) if value is not None and str(value) else Path()
    if not path.is_absolute():
        path = root / path
    return path


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.is_file():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Failed to load CSV %s: %s", path, exc)
    return pd.DataFrame()


def _first_existing_csv(paths: Iterable[Path]) -> pd.DataFrame:
    for path in paths:
        frame = _read_csv_if_exists(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def load_club_metadata(processed_dir: str | Path) -> pd.DataFrame:
    root = Path(processed_dir)
    return _first_existing_csv(
        [
            root / "club_metadata.csv",
            root / "golfdb_video_inventory.csv",
            root / "frame_metadata.csv",
        ]
    )


def load_swing_metadata(processed_dir: str | Path) -> pd.DataFrame:
    root = Path(processed_dir)
    return _first_existing_csv(
        [
            root / "swing_metadata.csv",
            root / "video_features.csv",
            root / "manual_swing_labels.csv",
            root / "pose_feature_table.csv",
        ]
    )


def load_player_metadata(processed_dir: str | Path) -> pd.DataFrame:
    root = Path(processed_dir)
    return _first_existing_csv(
        [
            root / "player_metadata.csv",
            root / "manual_phase_label_template.csv",
            root / "swing_phase_review_template.csv",
        ]
    )


def _load_json_if_exists(path: Path) -> dict:
    if path.exists() and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to load JSON %s: %s", path, exc)
    return {}


def _load_numpy_if_exists(path: Path) -> Optional[np.ndarray]:
    if path.exists() and path.is_file():
        try:
            return np.load(path, allow_pickle=True)
        except Exception as exc:
            LOGGER.warning("Failed to load NumPy artifact %s: %s", path, exc)
    return None


def parse_pose_experiment_outputs(experiments_dir: str | Path) -> dict:
    root = Path(experiments_dir)
    artifacts: dict[str, Any] = {
        "pose_landmarks": _read_csv_if_exists(root / "pose_landmarks.csv"),
        "pose_landmarks_sample": _read_csv_if_exists(root / "pose_landmarks_sample.csv"),
        "pose_feature_table": _read_csv_if_exists(root / "pose_feature_table.csv"),
        "motion_scores": _read_csv_if_exists(root / "motion_scores.csv"),
        "body_metrics": _load_json_if_exists(root / "body_metrics.json"),
        "pose_sequences": _load_numpy_if_exists(root / "pose_sequences.npy"),
        "artifact_manifest": _load_json_if_exists(root / "artifact_manifest.json"),
    }
    return artifacts


def _normalize_metric(value: Optional[float], scale: float = 1.0, offset: float = 0.0) -> Optional[float]:
    if value is None:
        return None
    try:
        score = (float(value) * scale) + offset
    except Exception:
        return None
    return float(max(0.0, min(100.0, score)))


def summarize_pose_artifacts(artifacts: dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "hip_rotation": 0.0,
        "shoulder_rotation": 0.0,
        "spine_angle": 0.0,
        "head_stability": 0.0,
        "weight_shift": 0.0,
    }

    feature_table = artifacts.get("pose_feature_table")
    if isinstance(feature_table, pd.DataFrame) and not feature_table.empty:
        if "hip_width" in feature_table.columns:
            hip_width = float(feature_table["hip_width"].fillna(0).mean())
            metrics["hip_rotation"] = _normalize_metric(hip_width, scale=18.0) or 0.0
        if "shoulder_width" in feature_table.columns:
            shoulder_width = float(feature_table["shoulder_width"].fillna(0).mean())
            metrics["shoulder_rotation"] = _normalize_metric(shoulder_width, scale=16.0) or 0.0
        if "head_to_hip_y" in feature_table.columns:
            head_to_hip = float(feature_table["head_to_hip_y"].fillna(0).abs().mean())
            metrics["spine_angle"] = _normalize_metric(head_to_hip, scale=90.0) or 0.0
            metrics["head_stability"] = _normalize_metric(100.0 - head_to_hip * 20.0) or 0.0

    landmarks_df = artifacts.get("pose_landmarks")
    if isinstance(landmarks_df, pd.DataFrame) and not landmarks_df.empty:
        if {"x", "y"}.issubset(landmarks_df.columns):
            head_rows = landmarks_df[landmarks_df.get("keypoint_name", "") == "nose"] if "keypoint_name" in landmarks_df.columns else pd.DataFrame()
            if not head_rows.empty:
                head_spread = float(np.hypot(head_rows["x"].std(ddof=0) or 0.0, head_rows["y"].std(ddof=0) or 0.0))
                metrics["head_stability"] = _normalize_metric(100.0 - head_spread * 100.0) or metrics["head_stability"]

    body_metrics = artifacts.get("body_metrics")
    if isinstance(body_metrics, dict):
        for key in metrics:
            value = body_metrics.get(key)
            if value is not None:
                normalized = _normalize_metric(float(value), scale=1.0)
                if normalized is not None:
                    metrics[key] = normalized

    motion_df = artifacts.get("motion_scores")
    if isinstance(motion_df, pd.DataFrame) and not motion_df.empty and "motion_score" in motion_df.columns:
        motion_mean = float(motion_df["motion_score"].fillna(0).mean())
        metrics["weight_shift"] = _normalize_metric(100.0 - motion_mean * 10.0) or metrics["weight_shift"]

    return {key: round(float(value), 3) for key, value in metrics.items()}


class ModelManager:
    """Load training artifacts and provide inference entry points for the dashboard."""

    def __init__(self, config: Dict | None = None) -> None:
        self.original_config = copy.deepcopy(config or {})
        self.config = copy.deepcopy(config or {})
        self.logger = LOGGER

        project_root_value = self.config.get("project_root") or Path.cwd()
        self.project_root = Path(project_root_value).resolve()

        paths = self.config.setdefault("paths", {})
        self.data_dir = ensure_dir(_to_path(paths.get("data_dir", "data"), self.project_root))
        self.models_dir = ensure_dir(_to_path(paths.get("models_dir", "models"), self.project_root))
        self.pretrained_models_dir = ensure_dir(_to_path(paths.get("pretrained_models_dir", self.models_dir / "pretrained"), self.project_root))
        self.trained_models_dir = ensure_dir(_to_path(paths.get("trained_models_dir", self.models_dir / "trained"), self.project_root))
        self.processed_dir = ensure_dir(_to_path(paths.get("processed_dir", self.data_dir / "processed"), self.project_root))
        self.frames_dir = ensure_dir(_to_path(paths.get("frames_dir", self.data_dir / "frames"), self.project_root))
        self.splits_dir = ensure_dir(_to_path(paths.get("splits_dir", self.data_dir / "splits"), self.project_root))
        self.outputs_dir = ensure_dir(_to_path(paths.get("outputs_dir", "outputs"), self.project_root))
        self.experiments_dir = ensure_dir(self.outputs_dir / "experiments")
        self.overlays_dir = ensure_dir(self.outputs_dir / "overlays")

        self.models: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, pd.DataFrame] = {}
        self.experiment_artifacts: Dict[str, Any] = {}
        self.pose_metrics: Dict[str, float] = {}

        self.load_models()
        self.load_metadata()
        self.load_experiment_artifacts()

    def _resolve_model_file(self, model_key: str) -> Optional[Path]:
        candidates = MODEL_FILE_NAMES.get(model_key, [f"{model_key}.pt"])
        search_dirs = [self.trained_models_dir, self.pretrained_models_dir]

        for directory in search_dirs:
            for candidate in candidates:
                path = directory / candidate
                if path.exists():
                    return path

        keywords = [part for part in model_key.replace("_", " ").split() if part]
        for directory in search_dirs:
            for path in sorted(directory.glob("*.pt")):
                stem = path.stem.lower()
                if all(keyword in stem for keyword in keywords):
                    return path
        return None

    def _load_model_artifact(self, model_key: str, path: Optional[Path]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": model_key,
            "path": str(path) if path else None,
            "available": False,
            "loader": "missing",
            "model": None,
        }
        if path is None:
            self.logger.warning("Missing model artifact for %s", model_key)
            return record

        suffix = path.suffix.lower()
        try:
            if suffix == ".joblib" and joblib is not None:
                record.update({"model": joblib.load(path), "available": True, "loader": "joblib"})
                return record
            if suffix == ".pkl" and joblib is not None:
                record.update({"model": joblib.load(path), "available": True, "loader": "joblib"})
                return record
            if suffix == ".pt":
                if YOLO is not None and model_key in {"club_detector", "pose_model"}:
                    record.update({"model": YOLO(str(path)), "available": True, "loader": "ultralytics"})
                    return record
                if torch is not None:
                    try:
                        record.update({"model": torch.load(path, map_location="cpu"), "available": True, "loader": "torch"})
                        return record
                    except Exception as exc:
                        self.logger.warning("Torch load failed for %s: %s", path, exc)
        except Exception as exc:
            self.logger.warning("Failed to load %s model from %s: %s", model_key, path, exc)

        self.logger.warning("Model artifact could not be loaded for %s at %s", model_key, path)
        return record

    def load_models(self) -> Dict[str, Dict[str, Any]]:
        self.models = {}
        for model_key in MODEL_FILE_NAMES:
            path = self._resolve_model_file(model_key)
            record = self._load_model_artifact(model_key, path)
            self.models[model_key] = record

        club_detector_path = self.models.get("club_detector", {}).get("path")
        if club_detector_path:
            self.config.setdefault("club_recognition", {})["yolo_model_path"] = club_detector_path
        else:
            self.logger.info("No custom club detector found. Falling back to the built-in club recognition path.")

        pose_model_path = self.models.get("pose_model", {}).get("path")
        if pose_model_path:
            pose_settings = self.config.setdefault("pose_estimation", {})
            pose_settings["backend"] = "yolov8_pose"
            pose_settings["model_path"] = pose_model_path

        return self.models

    def load_metadata(self) -> Dict[str, pd.DataFrame]:
        self.metadata = {
            "club": load_club_metadata(self.processed_dir),
            "swing": load_swing_metadata(self.processed_dir),
            "player": load_player_metadata(self.processed_dir),
        }
        return self.metadata

    def load_experiment_artifacts(self) -> Dict[str, Any]:
        self.experiment_artifacts = parse_pose_experiment_outputs(self.experiments_dir)
        self.pose_metrics = summarize_pose_artifacts(self.experiment_artifacts)
        return self.experiment_artifacts

    def load_metadata_files(self) -> Dict[str, pd.DataFrame]:
        return self.metadata or self.load_metadata()

    def _image_from_path(self, frame_path: str | Path) -> Image.Image:
        return Image.open(frame_path).convert("RGB")

    def _public_output_url(self, file_path: str | Path) -> str:
        path = Path(file_path)
        try:
            relative = path.relative_to(self.outputs_dir)
            return f"/outputs/{relative.as_posix()}"
        except Exception:
            return f"/outputs/{path.name}"

    def collect_overlay_files(self, token: Optional[str] = None) -> list[str]:
        search_roots = [self.overlays_dir, self.experiments_dir]
        overlay_suffixes = {".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png"}
        matches: list[str] = []
        token_lower = token.lower() if token else None

        for root in search_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in overlay_suffixes:
                    continue
                if token_lower and token_lower not in path.stem.lower() and token_lower not in path.as_posix().lower():
                    continue
                matches.append(self._public_output_url(path))

        if matches:
            return sorted(dict.fromkeys(matches))

        fallback: list[str] = []
        for root in search_roots:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in overlay_suffixes:
                    fallback.append(self._public_output_url(path))
        return sorted(dict.fromkeys(fallback[:5]))

    def detect_club(self, frame: str | Path) -> Dict[str, Any]:
        frame_path = Path(frame)
        result = recognize_club_from_frame(str(frame_path), self.config)
        predicted_club = result.get("predicted_club") or result.get("category") or "Unknown"
        return {
            "club": predicted_club,
            "detected_club": predicted_club,
            "confidence": float(result.get("confidence", 0.0)),
            "status": result.get("status", "confirmed" if result.get("confidence", 0.0) >= 0.6 else "uncertain"),
            "bbox": result.get("bbox"),
            "reasoning": result.get("reasoning"),
            "sources": result.get("sources", {}),
            "raw": result,
        }

    def check_body_visibility(self, frame: str | Path) -> Dict[str, Any]:
        return check_body_visibility_from_frame(str(frame), self.config)

    def analyze_pose(self, video_path: str | Path) -> Dict[str, Any]:
        video_path = str(video_path)
        pose_frames = run_pose_estimation(video_path, self.config)
        live_metrics = compute_swing_metrics(pose_frames, self.config)
        rendered = render_annotated_video(video_path, pose_frames, str(self.outputs_dir))
        body_tracking = {
            "tracked_parts": {
                "feet": ["left_ankle", "right_ankle"],
                "knees": ["left_knee", "right_knee"],
                "hips": ["left_hip", "right_hip"],
                "spine_angle": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
                "shoulders": ["left_shoulder", "right_shoulder"],
                "elbows": ["left_elbow", "right_elbow"],
                "hands": ["left_wrist", "right_wrist"],
                "neck": ["left_shoulder", "right_shoulder", "nose"],
                "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
            },
            "frame_count": len(pose_frames),
            "source": "yolov8_pose_stub" if not pose_frames else "yolov8_pose",
        }

        artifact_metrics = self.pose_metrics or summarize_pose_artifacts(self.experiment_artifacts)
        technical_metrics = self._merge_metrics(live_metrics, artifact_metrics)
        overlay_files = self.collect_overlay_files(Path(video_path).stem)
        if rendered:
            overlay_files = sorted(dict.fromkeys([*overlay_files, self._public_output_url(rendered)]))

        return {
            "video_path": video_path,
            "pose_frames": pose_frames,
            "pose_frame_count": len(pose_frames),
            "live_metrics": live_metrics,
            "experiment_metrics": artifact_metrics,
            "technical_metrics": technical_metrics,
            "body_tracking": body_tracking,
            "overlay_files": overlay_files,
            "rendered_outputs": {"annotated_video_path": rendered},
            "model_sources": {
                "pose_model": self.models.get("pose_model", {}).get("loader", "missing"),
            },
        }

    def analyze_swing(
        self,
        video_path: str | Path,
        club_image_path: Optional[str | Path] = None,
        manual_club_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        video_path = str(video_path)
        club_source = club_image_path or self._default_club_frame(video_path)
        club_detection: Dict[str, Any]
        if club_source:
            club_detection = self.detect_club(club_source)
        else:
            club_detection = {
                "club": manual_club_category or "Unknown",
                "detected_club": manual_club_category or "Unknown",
                "confidence": 0.0,
                "status": "uncertain",
                "bbox": None,
                "reasoning": None,
                "sources": {"detector": "fallback_default"},
                "raw": {},
            }

        pose_analysis = self.analyze_pose(video_path)
        final_club = manual_club_category or club_detection.get("club") or club_detection.get("detected_club") or "Unknown"

        return {
            "club": final_club,
            "club_detection": club_detection,
            "pose_analysis": pose_analysis,
            "advanced_metrics": pose_analysis.get("technical_metrics", {}),
            "overlay_files": pose_analysis.get("overlay_files", []),
            "metadata": self._metadata_summary(),
            "model_outputs": {
                "club_model": self.models.get("club_detector", {}).get("loader", "missing"),
                "pose_model": self.models.get("pose_model", {}).get("loader", "missing"),
                "club_classifier": self.models.get("club_classifier", {}).get("loader", "missing"),
                "swing_classifier": self.models.get("swing_classifier", {}).get("loader", "missing"),
            },
        }

    def _default_club_frame(self, video_path: str) -> Optional[str]:
        video_file = Path(video_path)
        if not video_file.exists():
            return None

        fallback_dir = ensure_dir(self.experiments_dir / "fallback_frames")
        fallback_path = fallback_dir / f"{video_file.stem}_first_frame.jpg"
        if fallback_path.exists():
            return str(fallback_path)

        capture = cv2.VideoCapture(str(video_file))
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            return None

        cv2.imwrite(str(fallback_path), frame)
        return str(fallback_path)

    def _metadata_summary(self) -> Dict[str, int]:
        return {
            "club_rows": int(len(self.metadata.get("club", pd.DataFrame()))),
            "swing_rows": int(len(self.metadata.get("swing", pd.DataFrame()))),
            "player_rows": int(len(self.metadata.get("player", pd.DataFrame()))),
        }

    @staticmethod
    def _merge_metrics(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for key in {"hip_rotation", "shoulder_rotation", "spine_angle", "head_stability", "weight_shift"}:
            primary_value = primary.get(key)
            secondary_value = secondary.get(key)
            chosen = primary_value if primary_value not in (None, 0, 0.0) else secondary_value
            if chosen is None:
                chosen = 0.0
            merged[key] = round(float(chosen), 3)
        return merged
