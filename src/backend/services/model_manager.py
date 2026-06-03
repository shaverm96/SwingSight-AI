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
from backend.services.overlay_generator import generate_pose_overlay
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

    def validate_video(self, video_path: str | Path) -> Dict[str, Any]:
        video_file = Path(video_path)
        metadata: Dict[str, Any] = {
            "video_path": str(video_file),
            "exists": video_file.exists(),
            "readable": False,
            "frame_count": None,
            "fps": None,
            "width": None,
            "height": None,
            "duration_seconds": None,
        }

        if not video_file.exists() or not video_file.is_file():
            metadata["error"] = "Uploaded video file was not found."
            return metadata

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            capture.release()
            metadata["error"] = "OpenCV could not open the uploaded video."
            return metadata

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()

        metadata.update(
            {
                "readable": frame_count > 0 and width > 0 and height > 0,
                "frame_count": frame_count or None,
                "fps": fps or None,
                "width": width or None,
                "height": height or None,
                "duration_seconds": (frame_count / fps) if frame_count > 0 and fps > 0 else None,
            }
        )

        if not metadata["readable"]:
            metadata["error"] = "The uploaded video could not be read by OpenCV."
        return metadata

    def extract_fallback_frames(self, video_path: str | Path, video_metadata: Optional[Dict[str, Any]] = None, limit: int = 8) -> list[Dict[str, Any]]:
        video_file = Path(video_path)
        if not video_file.exists():
            return []

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            capture.release()
            return []

        fps = float((video_metadata or {}).get("fps") or capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_step = max(1, int(round(fps / 2.0))) if fps > 0 else 1
        fallback_dir = ensure_dir(self.experiments_dir / "fallback_frames" / video_file.stem)
        rows: list[Dict[str, Any]] = []
        frame_index = 0
        saved_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                frame_path = fallback_dir / f"frame_{saved_index:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                rows.append(
                    {
                        "frame_index": frame_index,
                        "frame_path": str(frame_path),
                        "timestamp_seconds": frame_index / fps if fps > 0 else None,
                    }
                )
                saved_index += 1
                if saved_index >= limit:
                    break
            frame_index += 1

        capture.release()
        return rows

    def _pose_model_runtime(self) -> Dict[str, Any]:
        record = self.models.get("pose_model", {})
        model = record.get("model")
        if model is not None and record.get("loader") == "ultralytics":
            return {"loaded": True, "loader": "ultralytics", "path": record.get("path"), "model": model, "source": "custom"}

        if YOLO is None:
            return {"loaded": False, "loader": "missing", "path": None, "model": None, "source": "unavailable"}

        pose_settings = self.config.get("pose_estimation", {})
        candidate_paths = []
        configured_path = pose_settings.get("model_path")
        if configured_path:
            candidate_paths.append(configured_path)
        candidate_paths.append(self.models_dir / "pretrained" / "yolov8n-pose.pt")
        candidate_paths.append("yolov8n-pose.pt")

        for candidate in candidate_paths:
            try:
                model = YOLO(str(candidate))
                return {"loaded": True, "loader": "ultralytics", "path": str(candidate), "model": model, "source": "pretrained"}
            except Exception as exc:
                self.logger.info("Pose model candidate unavailable (%s): %s", candidate, exc)

        return {"loaded": False, "loader": "missing", "path": None, "model": None, "source": "unavailable"}

    def _save_pose_landmark_rows(self, video_path: str | Path, pose_frames: list[Dict[str, Any]]) -> Optional[str]:
        if not pose_frames:
            return None

        rows: list[Dict[str, Any]] = []
        video_file = Path(video_path)
        for frame in pose_frames:
            frame_index = frame.get("frame_index")
            timestamp_sec = frame.get("timestamp_sec")
            landmarks = frame.get("landmarks", {}) or {}
            for name, landmark in landmarks.items():
                rows.append(
                    {
                        "video_id": video_file.stem,
                        "frame_number": frame_index,
                        "timestamp": timestamp_sec,
                        "club_type": "unknown",
                        "keypoint_name": name,
                        "x": landmark.get("x"),
                        "y": landmark.get("y"),
                        "confidence": landmark.get("visibility"),
                    }
                )

        if not rows:
            return None

        output_path = self.experiments_dir / "pose_landmarks.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return str(output_path)

    def _fallback_score_label(self, metrics: Dict[str, Optional[float]], video_processed: bool, pose_frames_detected: int) -> str:
        metric_values = [value for value in metrics.values() if value is not None]
        if not video_processed:
            return "Analysis incomplete"
        if pose_frames_detected == 0 or not metric_values:
            return "Needs clearer video"
        if len(metric_values) < 3:
            return "Needs clearer video"
        average = float(sum(metric_values) / len(metric_values))
        if average >= 80:
            return "Strong swing"
        if average >= 65:
            return "Improving"
        return "Needs clearer video"

    def _fallback_club_note(self) -> str:
        return "Club detection needs a closer view of the club head or club end."

    def _image_from_path(self, frame_path: str | Path) -> Image.Image:
        return Image.open(frame_path).convert("RGB")

    def _public_output_url(self, file_path: str | Path) -> str:
        path = Path(file_path)
        try:
            relative = path.relative_to(self.outputs_dir)
            return f"/outputs/{relative.as_posix()}"
        except Exception:
            return f"/outputs/{path.name}"

    def _public_upload_url(self, file_path: str | Path) -> str:
        return f"/uploads/{Path(file_path).name}"

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
        if not self.models.get("club_detector", {}).get("available", False):
            return {
                "club": "Not detected",
                "detected_club": "Not detected",
                "confidence": 0.0,
                "status": "not_detected",
                "bbox": None,
                "reasoning": self._fallback_club_note(),
                "sources": {"detector": "missing"},
                "raw": {},
            }

        frame_path = Path(frame)
        result = recognize_club_from_frame(str(frame_path), self.config)
        predicted_club = result.get("predicted_club") or result.get("category")
        confidence = float(result.get("confidence", 0.0))
        threshold = float(self.config.get("club_recognition", {}).get("detector_confidence_threshold", 0.6))
        if not predicted_club or confidence < threshold:
            predicted_club = "Not detected"
        return {
            "club": predicted_club,
            "detected_club": predicted_club,
            "confidence": confidence,
            "status": result.get("status", "confirmed" if confidence >= threshold else "not_detected"),
            "bbox": result.get("bbox"),
            "reasoning": result.get("reasoning"),
            "sources": result.get("sources", {}),
            "raw": result,
        }

    def check_body_visibility(self, frame: str | Path) -> Dict[str, Any]:
        return check_body_visibility_from_frame(str(frame), self.config)

    def analyze_pose(self, video_path: str | Path) -> Dict[str, Any]:
        video_file = Path(video_path)
        validation = self.validate_video(video_file)
        if validation.get("error"):
            raise ValueError(validation["error"])

        fallback_frames = self.extract_fallback_frames(video_file, validation)
        pose_runtime = self._pose_model_runtime()
        pose_frames = run_pose_estimation(str(video_file), self.config) if pose_runtime["loaded"] else []
        pose_landmarks_csv = self._save_pose_landmark_rows(video_file, pose_frames)
        technical_metrics = compute_swing_metrics(pose_frames, self.config) if pose_frames else {
            "hip_rotation": None,
            "shoulder_rotation": None,
            "spine_angle": None,
            "head_stability": None,
            "weight_shift": None,
        }

        pose_frames_detected = sum(1 for frame in pose_frames if frame.get("landmarks"))
        pose_frames_processed = len(pose_frames)

        overlay_result = generate_pose_overlay(
            video_file,
            pose_frames,
            self.overlays_dir,
            pose_landmarks_path=pose_landmarks_csv,
        )
        overlay_video_path = overlay_result.get("overlay_video_path")
        overlay_video_url = overlay_result.get("overlay_video_url")
        overlay_files = self.collect_overlay_files(video_file.stem)
        if overlay_video_url:
            overlay_files = sorted(dict.fromkeys([overlay_video_url, *overlay_files]))
        elif overlay_video_path:
            overlay_files = sorted(dict.fromkeys([self._public_output_url(overlay_video_path), *overlay_files]))

        club_source = fallback_frames[0]["frame_path"] if fallback_frames else self._default_club_frame(str(video_file))
        club_detection = self.detect_club(club_source) if club_source else {
            "club": "Not detected",
            "detected_club": "Not detected",
            "confidence": 0.0,
            "status": "not_detected",
            "bbox": None,
            "reasoning": None,
            "sources": {"detector": "fallback_default"},
            "raw": {},
        }

        if club_detection.get("club") in {None, "Unknown", "", "Driver", "Iron", "Hybrid", "Wood", "Wedge"} and not self.models.get("club_detector", {}).get("available", False):
            club_detection["club"] = "Not detected"
            club_detection["detected_club"] = "Not detected"

        warnings: list[str] = []
        if pose_frames_processed == 0:
            warnings.append("Pose tracking did not detect enough landmarks in the uploaded video.")
        if not pose_runtime["loaded"]:
            warnings.append("Using pretrained pose fallback or frame-level fallback analysis.")
        if not overlay_result.get("success", False):
            warnings.extend(overlay_result.get("warnings", []))
            warnings.append(overlay_result.get("message", "Pose tracking could not be generated from this video."))
        if club_detection.get("club") == "Not detected":
            warnings.append(self._fallback_club_note())

        self.logger.info("Video received: %s", video_file)
        self.logger.info("Video readable: %s", validation)
        self.logger.info("Frames extracted: %s", len(fallback_frames))
        self.logger.info("Pose model loaded: %s", pose_runtime["loaded"])
        self.logger.info("Pose frames processed: %s", pose_frames_processed)
        self.logger.info("Metrics generated: %s", technical_metrics)
        self.logger.info("Club detection result: %s", club_detection)

        fallback_used = not self.models.get("pose_model", {}).get("available", False) or not self.models.get("club_detector", {}).get("available", False)

        model_outputs = {
            "pose_model_loaded": bool(pose_runtime["loaded"]),
            "club_model_loaded": bool(self.models.get("club_detector", {}).get("available", False)),
            "fallback_used": fallback_used,
            "pose_model_source": pose_runtime.get("source"),
            "club_model_source": self.models.get("club_detector", {}).get("loader", "missing"),
        }

        tracking = {
            "pose_frames_processed": pose_frames_processed,
            "pose_frames_detected": pose_frames_detected,
            "video_fps": validation.get("fps"),
            "frame_count": validation.get("frame_count"),
            "duration_seconds": validation.get("duration_seconds"),
            "detection_rate": overlay_result.get("detection_rate"),
            "average_confidence": overlay_result.get("average_confidence"),
        }

        debug = {
            "video_metadata": validation,
            "model_load_status": model_outputs,
            "fallback_status": {
                "used": model_outputs["fallback_used"],
                "fallback_frames_dir": str((self.experiments_dir / "fallback_frames" / video_file.stem).resolve()),
                "pose_landmarks_csv": pose_landmarks_csv,
                "overlay_video_path": overlay_video_path,
            },
            "file_paths": {
                "video_path": str(video_file),
                "pose_landmarks_csv": pose_landmarks_csv,
                "overlay_video_path": overlay_video_path,
                "fallback_frames_dir": str((self.experiments_dir / "fallback_frames" / video_file.stem).resolve()),
            },
        }

        return {
            "status": "success",
            "video_processed": True,
            "video_metadata": validation,
            "club": club_detection.get("club", "Not detected"),
            "club_note": self._fallback_club_note() if club_detection.get("club") == "Not detected" else None,
            "club_detection": club_detection,
            "advanced_metrics": technical_metrics,
            "tracking": tracking,
            "model_outputs": model_outputs,
            "overlay_files": overlay_files,
            "visualization": {
                "original_video_url": self._public_upload_url(video_file),
                "overlay_video_url": overlay_video_url,
                "overlay_video_path": overlay_video_path,
                "overlay_available": bool(overlay_video_url),
            },
            "fallback_used": model_outputs["fallback_used"],
            "warnings": warnings,
            "debug": debug,
        }

    def analyze_swing(
        self,
        video_path: str | Path,
        club_image_path: Optional[str | Path] = None,
        manual_club_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = club_image_path
        analysis = self.analyze_pose(video_path)
        if manual_club_category:
            analysis["club"] = manual_club_category
            analysis["club_detection"] = {
                **analysis.get("club_detection", {}),
                "club": manual_club_category,
                "detected_club": manual_club_category,
                "status": "confirmed",
            }
            analysis["club_note"] = None
        return analysis

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
