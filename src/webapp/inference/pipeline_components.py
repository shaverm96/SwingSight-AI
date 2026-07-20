from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from swingsight.club_detection import CLUB_CATEGORIES
from swingsight.club_recognition import predicted_to_category, recognize_club_from_frame
from swingsight.feedback import generate_feedback_sections
from swingsight.metrics import compute_swing_metrics
from swingsight.pose_estimation import run_pose_estimation
from swingsight.visualization import render_annotated_video


def _classify_club_recognition(recognition: Dict) -> Dict:
    detected_category = str(recognition.get("detected_category", "Iron/Wedge"))
    normalized = detected_category.lower()
    if normalized.startswith("wood"):
        broad_category = "wood"
    elif normalized.startswith("iron"):
        broad_category = "iron_wedge"
    else:
        broad_category = "unknown"
    recognition["category"] = predicted_to_category(recognition.get("predicted_club"), broad_category)
    return recognition


def run_club_detection(club_image_path: Optional[str], config: Dict) -> Dict:
    """Detect club/head region and broad category with YOLOv8-compatible interface."""
    if not club_image_path:
        return {"category": CLUB_CATEGORIES[0], "confidence": 0.0, "source": "manual_default"}

    recognition = _classify_club_recognition(recognize_club_from_frame(club_image_path, config))

    return {
        "category": recognition["category"],
        "confidence": float(recognition.get("confidence", 0.0)),
        "bbox": recognition.get("bbox"),
        "source": recognition.get("sources", {}).get("detector", "club_recognition"),
        "recognition": recognition,
    }


def detect_club_from_frame(frame_path: str, config: Dict) -> Dict:
    """Run the complete staged club-recognition pipeline on one frame."""
    return _classify_club_recognition(recognize_club_from_frame(frame_path, config))


def check_body_visibility_from_frame(frame_path: str, config: Dict) -> Dict:
    """Check whether key body landmarks are visible in a single frame.

    Returns a dict: {visible: bool, missing: [parts], landmarks: {...}}

    NOTE: This is a placeholder. Replace with image-level pose estimator (MediaPipe or
    YOLOv8-pose image inference) that returns real keypoints.
    """
    required = [
        "left_ankle",
        "right_ankle",
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        "nose",
    ]

    # Heuristic placeholder: if file exists and size > 2000 bytes, assume visible.
    p = Path(frame_path)
    visible = p.exists() and p.stat().st_size > 2000
    missing = [] if visible else required

    # Fake landmarks structure for UI plumbing
    landmarks = {}
    if visible:
        for name in required:
            landmarks[name] = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}

    return {"visible": bool(visible), "missing": missing, "landmarks": landmarks}


def classify_club_category(club_image_path: Optional[str], detected_category: str, config: Dict) -> Dict:
    """Return the staged CNN category and probabilities for a club image."""

    _ = detected_category  # The stage-one CNN owns the iron/wood decision.
    if not club_image_path:
        return {
            "category": "iron_wedge",
            "probabilities": {},
            "source": "cnn_missing_image",
        }

    recognition = _classify_club_recognition(recognize_club_from_frame(club_image_path, config))
    return {
        "category": recognition["category"],
        "probabilities": recognition.get("probabilities", {}),
        "source": recognition.get("sources", {}).get("broad_classifier", "cnn"),
    }


def recognize_loft_or_number(club_image_path: Optional[str], config: Dict) -> Dict:
    """Return the number-stage CNN result when the submitted club is an iron."""

    if not club_image_path:
        return {
            "recognized_text": None,
            "normalized_loft_or_number": None,
            "confidence": 0.0,
            "source": "cnn_missing_image",
        }

    recognition = recognize_club_from_frame(club_image_path, config)
    number = recognition.get("ocr") or {}
    return {
        "recognized_text": number.get("text"),
        "normalized_loft_or_number": number.get("text"),
        "confidence": float(number.get("confidence", 0.0)),
        "source": number.get("source", "not_applicable"),
    }


def estimate_pose(video_path: str, config: Dict) -> List[Dict]:
    """Run YOLOv8-pose-compatible extraction over video frames."""
    # TODO: Route through real YOLOv8-pose backend in swingsight.pose_estimation.
    return run_pose_estimation(video_path, config)


def track_body_landmarks(pose_frames: List[Dict]) -> Dict:
    """Build body-part tracking payload for downstream metrics and UI charts."""
    tracked = {
        "feet": ["left_ankle", "right_ankle"],
        "knees": ["left_knee", "right_knee"],
        "hips": ["left_hip", "right_hip"],
        "spine_angle": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
        "shoulders": ["left_shoulder", "right_shoulder"],
        "elbows": ["left_elbow", "right_elbow"],
        "hands": ["left_wrist", "right_wrist"],
        "neck": ["left_shoulder", "right_shoulder", "nose"],
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    }

    frame_count = len(pose_frames)
    return {
        "tracked_parts": tracked,
        "frame_count": frame_count,
        "source": "yolov8_pose_stub" if frame_count == 0 else "yolov8_pose",
    }


def extract_swing_metrics(pose_frames: List[Dict], config: Dict) -> Dict[str, float]:
    """Compute swing metrics using NumPy/pandas-friendly output."""
    raw_metrics = compute_swing_metrics(pose_frames, config)
    metrics_series = pd.Series(raw_metrics, dtype=float)
    return metrics_series.to_dict()


def score_swing(metrics: Dict[str, float], club_category: str, config: Dict) -> Dict:
    """Return an aggregate score with hooks for sklearn model scoring."""
    _ = config
    values = np.array(list(metrics.values()), dtype=float)
    normalized = 100.0 - min(100.0, float(np.nanmean(np.abs(values)) * 10.0)) if values.size else 0.0

    # TODO: Replace heuristic with a trained scikit-learn model per club category.
    return {
        "overall_score": round(normalized, 2),
        "club_category": club_category,
        "model": "sklearn_placeholder",
    }


def build_feedback(metrics: Dict[str, float], club_category: str, config: Dict) -> Dict[str, List[str]]:
    """Create structured coaching feedback from metric outputs."""
    return generate_feedback_sections(metrics, club_category, config)


def render_outputs(video_path: str, pose_frames: List[Dict], outputs_dir: str) -> Dict:
    """Render annotated assets through OpenCV-compatible visualization layer."""
    annotated_video_path = render_annotated_video(video_path, pose_frames, outputs_dir)
    return {
        "annotated_video_path": annotated_video_path,
        "preview_images": [],
    }
