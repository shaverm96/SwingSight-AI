from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from swingsight.club_detection import CLUB_CATEGORIES, detect_club_category
from swingsight.feedback import generate_club_feedback
from swingsight.metrics import compute_swing_metrics
from swingsight.pose_estimation import run_pose_estimation
from swingsight.visualization import render_annotated_video


def run_club_detection(club_image_path: Optional[str], config: Dict) -> Dict:
    """Detect club/head region and broad category with YOLOv8-compatible interface."""
    if not club_image_path:
        return {"category": CLUB_CATEGORIES[0], "confidence": 0.0, "source": "manual_default"}

    with Path(club_image_path).open("rb") as image_file:
        category, confidence = detect_club_category(image_file, config)

    # TODO: Replace with real YOLOv8 bbox output and confidence from detector inference.
    return {
        "category": category,
        "confidence": float(confidence),
        "bbox": None,
        "source": "yolov8_stub",
    }


def classify_club_category(club_image_path: Optional[str], detected_category: str, config: Dict) -> Dict:
    """Placeholder CNN classifier for finer club category classification."""
    _ = club_image_path
    _ = config
    # TODO: Connect a trained CNN classifier for exact club-type probabilities.
    return {
        "category": detected_category,
        "probabilities": {
            "driver_wood": 0.2,
            "hybrid": 0.2,
            "iron_wedge": 0.6,
        },
        "source": "cnn_stub",
    }


def recognize_loft_or_number(club_image_path: Optional[str], config: Dict) -> Dict:
    """OCR/custom-classifier placeholder for club number or loft extraction."""
    _ = club_image_path
    _ = config
    # TODO: Integrate OCR (e.g., EasyOCR/Tesseract) or custom image classifier here.
    return {
        "recognized_text": None,
        "normalized_loft_or_number": None,
        "confidence": 0.0,
        "source": "ocr_stub",
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


def build_feedback(metrics: Dict[str, float], club_category: str, config: Dict) -> List[str]:
    """Create human-readable coaching feedback from metric outputs."""
    return generate_club_feedback(metrics, club_category, config)


def render_outputs(video_path: str, pose_frames: List[Dict], outputs_dir: str) -> Dict:
    """Render annotated assets through OpenCV-compatible visualization layer."""
    annotated_video_path = render_annotated_video(video_path, pose_frames, outputs_dir)
    return {
        "annotated_video_path": annotated_video_path,
        "preview_images": [],
    }
