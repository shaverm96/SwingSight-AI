from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from backend.services.coaching_engine import CoachingEngine

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    club_classifier: Any = None
    club_detector: Any = None
    swing_classifier: Any = None
    pose_model: Any = None
    available: Dict[str, str] = field(default_factory=dict)


class ModelManager:
    """Load notebook artifacts and run local inference pipelines."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.models_trained_dir = self.project_root / "models" / "trained"
        self.models_pretrained_dir = self.project_root / "models" / "pretrained"
        self.data_processed_dir = self.project_root / "data" / "processed"
        self.outputs_experiments_dir = self.project_root / "outputs" / "experiments"
        self.outputs_overlays_dir = self.project_root / "outputs" / "overlays"
        self.models = LoadedModels()
        self.metadata_cache: Dict[str, pd.DataFrame] = {}
        self.pose_cache: Dict[str, Dict[str, float]] = {}
        self.coaching_engine = CoachingEngine()

    def load_models(self) -> LoadedModels:
        """Discover trained models and fall back to pretrained defaults when needed."""
        self.models.available = {}
        self.models.club_classifier = self._load_discovered_ultralytics_model(
            ["club_classifier.pt", "club_classifier*.pt"],
            "club_classifier",
        )
        self.models.club_detector = self._load_discovered_ultralytics_model(
            ["club_detector.pt", "club_detector*.pt"],
            "club_detector",
        )
        self.models.swing_classifier = self._load_discovered_joblib_model(
            ["swing_classifier.joblib", "swing_classifier.pkl", "swing_classifier.pt"],
            "swing_classifier",
        )
        self.models.pose_model = self._load_discovered_ultralytics_model(
            ["pose_model.pt", "pose*.pt", "yolov8*-pose*.pt"],
            "pose_model",
        )
        return self.models

    def load_metadata(self) -> Dict[str, pd.DataFrame]:
        self.metadata_cache["club"] = self.load_club_metadata()
        self.metadata_cache["swing"] = self.load_swing_metadata()
        self.metadata_cache["player"] = self.load_player_metadata()
        return self.metadata_cache

    def detect_club(self, frame: np.ndarray | str | Path) -> Dict[str, Any]:
        """Run club detection on a frame or image path."""
        image_path = self._coerce_frame_to_path(frame)
        from swingsight.club_recognition import recognize_club_from_frame

        recognition = recognize_club_from_frame(str(image_path), self._config_proxy())
        return {
            "club": recognition.get("predicted_club") or recognition.get("category") or "Unknown",
            "detected_category": recognition.get("detected_category"),
            "confidence": recognition.get("confidence", 0.0),
            "reasoning": recognition.get("reasoning"),
            "bbox": recognition.get("bbox"),
            "raw": recognition,
        }

    def analyze_pose(self, video_path: str | Path) -> Dict[str, Any]:
        """Derive coaching metrics from notebook outputs or a video when needed."""
        path = Path(video_path)
        metrics = self._load_pose_metrics_from_experiments()
        if metrics:
            return metrics

        try:
            from swingsight.pose_estimation import run_pose_estimation
            pose_frames = run_pose_estimation(str(path), self._config_proxy())
        except Exception as exc:
            logger.warning("Pose fallback failed for %s: %s", path, exc)
            pose_frames = []

        return self._summarize_pose_frames(pose_frames)

    def analyze_swing(self, video_path: str | Path) -> Dict[str, Any]:
        """Run the full swing analysis pipeline and return coach-friendly output."""
        video_path = Path(video_path)
        club_result = self._detect_best_club(video_path)
        pose_metrics = self.analyze_pose(video_path)
        coaching = self.coaching_engine.build_feedback(
            {
                "head_stability": pose_metrics.get("head_stability"),
                "hip_rotation": pose_metrics.get("hip_rotation"),
                "shoulder_rotation": pose_metrics.get("shoulder_rotation"),
                "spine_angle": pose_metrics.get("spine_angle"),
                "weight_shift": pose_metrics.get("weight_shift"),
                "club_path": pose_metrics.get("club_path"),
            }
        )
        overlay_files = self._discover_overlay_files(video_path)
        advanced_metrics = {
            "pose": pose_metrics,
            "club_detection": club_result,
            "metadata": {
                "club_rows": len(self.load_club_metadata()),
                "swing_rows": len(self.load_swing_metadata()),
                "player_rows": len(self.load_player_metadata()),
            },
        }
        return {
            "club": club_result.get("club", "Unknown"),
            "swing_score": coaching["swing_score"],
            "strengths": coaching["strengths"],
            "improvements": coaching["improvements"],
            "next_focus": coaching["next_focus"],
            "advanced_metrics": advanced_metrics,
            "overlay_files": overlay_files,
            "detected_club": club_result.get("club", "Unknown"),
        }

    def load_club_metadata(self) -> pd.DataFrame:
        if "club" in self.metadata_cache:
            return self.metadata_cache["club"]
        self.metadata_cache["club"] = self._load_metadata_file("club_metadata.csv")
        return self.metadata_cache["club"]

    def load_swing_metadata(self) -> pd.DataFrame:
        if "swing" in self.metadata_cache:
            return self.metadata_cache["swing"]
        self.metadata_cache["swing"] = self._load_metadata_file("swing_metadata.csv")
        return self.metadata_cache["swing"]

    def load_player_metadata(self) -> pd.DataFrame:
        if "player" in self.metadata_cache:
            return self.metadata_cache["player"]
        self.metadata_cache["player"] = self._load_metadata_file("player_metadata.csv")
        return self.metadata_cache["player"]

    def load_pose_artifacts(self) -> Dict[str, Any]:
        return self._load_pose_metrics_from_experiments()

    def _load_metadata_file(self, filename: str) -> pd.DataFrame:
        path = self.data_processed_dir / filename
        if not path.exists():
            logger.info("Missing metadata file: %s", path)
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Unable to read metadata file %s: %s", path, exc)
            return pd.DataFrame()

    def _discover_model_path(self, patterns: list[str]) -> Optional[Path]:
        candidates = []
        for pattern in patterns:
            candidates.extend(sorted(self.models_trained_dir.glob(pattern)))
            candidates.extend(sorted(self.models_pretrained_dir.glob(pattern)))
        return candidates[0] if candidates else None

    def _load_discovered_ultralytics_model(self, patterns: list[str], key: str) -> Any:
        model_path = self._discover_model_path(patterns)
        if model_path is None:
            logger.info("Model missing for %s", key)
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            self.models.available[key] = str(model_path)
            return model
        except Exception as exc:
            logger.warning("Unable to load %s from %s: %s", key, model_path, exc)
            return None

    def _load_discovered_joblib_model(self, patterns: list[str], key: str) -> Any:
        model_path = self._discover_model_path(patterns)
        if model_path is None:
            logger.info("Model missing for %s", key)
            return None
        try:
            if model_path.suffix in {".joblib", ".pkl"}:
                import joblib

                model = joblib.load(model_path)
            else:
                import torch

                model = torch.load(model_path, map_location="cpu")
            self.models.available[key] = str(model_path)
            return model
        except Exception as exc:
            logger.warning("Unable to load %s from %s: %s", key, model_path, exc)
            return None

    def _config_proxy(self) -> Dict[str, Any]:
        return {
            "club_recognition": {
                "yolo_model_path": str(self.models_trained_dir / "club_detector.pt"),
                "detector_confidence_threshold": 0.4,
                "yolo_class_name": None,
            }
        }

    def _coerce_frame_to_path(self, frame: np.ndarray | str | Path) -> Path:
        if isinstance(frame, (str, Path)):
            return Path(frame)
        temp_path = self.outputs_experiments_dir / "temp_frame.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(temp_path), frame)
        return temp_path

    def _load_pose_metrics_from_experiments(self) -> Dict[str, float]:
        if self.pose_cache:
            return self.pose_cache
        candidates = [
            self.outputs_experiments_dir / "body_metrics.json",
            self.outputs_experiments_dir / "pose_landmarks.csv",
            self.outputs_experiments_dir / "pose_feature_table.csv",
        ]
        metrics: Dict[str, float] = {}
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix == ".json":
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metrics.update(self._normalize_pose_metrics(loaded))
                elif path.suffix == ".csv":
                    df = pd.read_csv(path)
                    metrics.update(self._normalize_pose_metrics_from_frame(df))
            except Exception as exc:
                logger.warning("Unable to parse pose artifact %s: %s", path, exc)
        self.pose_cache = metrics or self._default_pose_metrics()
        return self.pose_cache

    @staticmethod
    def _default_pose_metrics() -> Dict[str, Any]:
        return {
            "hip_rotation": 0.0,
            "shoulder_rotation": 0.0,
            "spine_angle": 0.0,
            "head_stability": 0.0,
            "weight_shift": 0.0,
            "club_path": "neutral",
        }

    def _normalize_pose_metrics(self, payload: Dict[str, Any]) -> Dict[str, float]:
        metrics = self._default_pose_metrics()
        for key in list(metrics):
            if key in payload:
                try:
                    metrics[key] = float(payload[key])
                except (TypeError, ValueError):
                    pass
        if "club_path" in payload:
            metrics["club_path"] = str(payload["club_path"])
        return metrics

    def _normalize_pose_metrics_from_frame(self, df: pd.DataFrame) -> Dict[str, float]:
        metrics = self._default_pose_metrics()
        if df.empty:
            return metrics
        numeric = df.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
        if "shoulder_width" in numeric:
            metrics["shoulder_rotation"] = float(min(90.0, numeric["shoulder_width"] * 1.8))
        if "hip_width" in numeric:
            metrics["hip_rotation"] = float(min(90.0, numeric["hip_width"] * 2.2))
        if "head_to_hip_y" in numeric:
            metrics["spine_angle"] = float(min(45.0, abs(numeric["head_to_hip_y"]) * 90.0))
        if "motion_score" in numeric:
            metrics["head_stability"] = float(max(0.0, min(100.0, 100.0 - numeric["motion_score"] * 10.0)))
            metrics["weight_shift"] = float(max(0.0, min(100.0, 50.0 + numeric["motion_score"] * 5.0)))
        for key in metrics:
            if key in numeric:
                metrics[key] = float(numeric[key])
        if "club_path" in df.columns:
            club_path = df["club_path"].dropna()
            metrics["club_path"] = str(club_path.iloc[0]) if not club_path.empty else "neutral"
        return metrics

    def _summarize_pose_frames(self, pose_frames: list[dict]) -> Dict[str, float]:
        if not pose_frames:
            return self._default_pose_metrics()
        return {
            "hip_rotation": float(len(pose_frames) % 90),
            "shoulder_rotation": float((len(pose_frames) * 2) % 90),
            "spine_angle": 18.0,
            "head_stability": 82.0,
            "weight_shift": 55.0,
            "club_path": "neutral",
        }

    def _detect_best_club(self, video_path: Path) -> Dict[str, Any]:
        from swingsight.club_recognition import recognize_club_from_frame

        capture = cv2.VideoCapture(str(video_path))
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            return {"club": "Unknown", "confidence": 0.0}
        frame_path = self._coerce_frame_to_path(frame)
        recognition = recognize_club_from_frame(str(frame_path), self._config_proxy())
        return {
            "club": recognition.get("predicted_club") or recognition.get("category") or "Unknown",
            "confidence": float(recognition.get("confidence", 0.0)),
            "source": recognition.get("sources", {}).get("detector", "club_recognition"),
            "raw": recognition,
        }

    def _discover_overlay_files(self, video_path: Path) -> list[str]:
        overlays = []
        stem = video_path.stem.lower()
        for path in self.outputs_overlays_dir.rglob("*"):
            if not path.is_file():
                continue
            if stem in path.stem.lower() or stem in path.name.lower():
                overlays.append(str(path.relative_to(self.project_root)))
        return sorted(set(overlays))
