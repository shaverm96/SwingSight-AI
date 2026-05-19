from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import math


@dataclass(frozen=True)
class SwingMetrics:
    spine_angle_deg: float
    head_movement_cm: float
    knee_flex_deg: float
    shoulder_turn_proxy: float
    hip_turn_proxy: float
    balance_score: float
    tempo_estimate: float

    def to_dict(self) -> Dict[str, float]:
        return dict(asdict(self))


def _compute_from_landmarks(landmarks: List[Dict], config: Dict) -> SwingMetrics:
    """Compute metrics from per-frame landmark dicts.

    Replace with geometry-based calculations when pose inference is ready.
    """
    _ = landmarks
    _ = config
    return SwingMetrics(
        spine_angle_deg=0.0,
        head_movement_cm=0.0,
        knee_flex_deg=0.0,
        shoulder_turn_proxy=0.0,
        hip_turn_proxy=0.0,
        balance_score=0.0,
        tempo_estimate=0.0,
    )


def angle_between(a: List[float], b: List[float], c: List[float]) -> float:
    """Angle at point b for triangle (a, b, c)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba == 0.0 or mag_bc == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def line_angle(a: List[float], b: List[float]) -> float:
    """Angle of line from a to b relative to horizontal (degrees)."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def mid_point(a: List[float], b: List[float]) -> List[float]:
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def get_point(landmarks: Dict[str, Dict], name: str) -> Optional[List[float]]:
    item = landmarks.get(name)
    if not item:
        return None
    return [float(item.get("x", 0.0)), float(item.get("y", 0.0))]


def compute_frame_metrics(landmarks: Dict[str, Dict]) -> Dict[str, float]:
    """Compute metrics for a single frame based on named landmarks."""
    left_ankle = get_point(landmarks, "left_ankle")
    right_ankle = get_point(landmarks, "right_ankle")
    left_knee = get_point(landmarks, "left_knee")
    right_knee = get_point(landmarks, "right_knee")
    left_hip = get_point(landmarks, "left_hip")
    right_hip = get_point(landmarks, "right_hip")
    left_shoulder = get_point(landmarks, "left_shoulder")
    right_shoulder = get_point(landmarks, "right_shoulder")
    nose = get_point(landmarks, "nose")

    stance_width = 0.0
    if left_ankle and right_ankle:
        stance_width = math.hypot(left_ankle[0] - right_ankle[0], left_ankle[1] - right_ankle[1])

    knee_flex_left = 0.0
    if left_hip and left_knee and left_ankle:
        knee_flex_left = angle_between(left_hip, left_knee, left_ankle)

    knee_flex_right = 0.0
    if right_hip and right_knee and right_ankle:
        knee_flex_right = angle_between(right_hip, right_knee, right_ankle)

    hip_tilt = 0.0
    if left_hip and right_hip:
        hip_tilt = line_angle(left_hip, right_hip)

    shoulder_tilt = 0.0
    if left_shoulder and right_shoulder:
        shoulder_tilt = line_angle(left_shoulder, right_shoulder)

    spine_angle = 0.0
    if left_hip and right_hip and left_shoulder and right_shoulder:
        mid_hip = mid_point(left_hip, right_hip)
        mid_shoulder = mid_point(left_shoulder, right_shoulder)
        spine_angle = line_angle(mid_hip, mid_shoulder)

    head_x = nose[0] if nose else 0.0
    head_y = nose[1] if nose else 0.0

    return {
        "stance_width": stance_width,
        "knee_flex_left": knee_flex_left,
        "knee_flex_right": knee_flex_right,
        "hip_tilt": hip_tilt,
        "shoulder_tilt": shoulder_tilt,
        "spine_angle": spine_angle,
        "head_x": head_x,
        "head_y": head_y,
    }


def compute_swing_metrics(landmarks: List[Dict], config: Dict) -> Dict[str, float]:
    """Compute basic swing metrics from landmarks.

    Returns a dict for easy JSON serialization.
    """
    metrics = _compute_from_landmarks(landmarks, config)
    return metrics.to_dict()
