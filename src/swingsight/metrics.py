from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import math


@dataclass(frozen=True)
class SwingMetrics:
    spine_angle_deg: Optional[float]
    head_movement_cm: Optional[float]
    knee_flex_deg: Optional[float]
    shoulder_turn_proxy: Optional[float]
    hip_turn_proxy: Optional[float]
    balance_score: Optional[float]
    tempo_estimate: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        return dict(asdict(self))


def _compute_from_landmarks(landmarks: List[Dict], config: Dict) -> SwingMetrics:
    _ = config

    if not landmarks:
        return SwingMetrics(None, None, None, None, None, None, None)

    frame_metrics = [compute_frame_metrics(frame.get("landmarks", {})) for frame in landmarks if frame.get("landmarks")]
    if not frame_metrics:
        return SwingMetrics(None, None, None, None, None, None, None)

    shoulder_tilts = [abs(metric["shoulder_tilt"]) for metric in frame_metrics if metric.get("shoulder_tilt") is not None]
    hip_tilts = [abs(metric["hip_tilt"]) for metric in frame_metrics if metric.get("hip_tilt") is not None]
    spine_angles = [abs(90.0 - abs(metric["spine_angle"])) for metric in frame_metrics if metric.get("spine_angle") is not None]
    head_x = [metric["head_x"] for metric in frame_metrics if metric.get("head_x") is not None]
    head_y = [metric["head_y"] for metric in frame_metrics if metric.get("head_y") is not None]
    stance_widths = [metric["stance_width"] for metric in frame_metrics if metric.get("stance_width") is not None]
    knee_flex_values = [
        value
        for metric in frame_metrics
        for value in (metric.get("knee_flex_left"), metric.get("knee_flex_right"))
        if value is not None and value > 0
    ]

    spine_angle_deg = float(np.mean(spine_angles)) if spine_angles else None
    shoulder_turn_proxy = float(max(0.0, min(100.0, 100.0 - np.mean(shoulder_tilts) * 2.0))) if shoulder_tilts else None
    hip_turn_proxy = float(max(0.0, min(100.0, 100.0 - np.mean(hip_tilts) * 2.0))) if hip_tilts else None

    if head_x and head_y:
        head_motion = float(np.hypot(np.std(head_x, ddof=0), np.std(head_y, ddof=0)))
        head_movement_cm = float(max(0.0, min(100.0, 100.0 - head_motion * 180.0)))
    else:
        head_movement_cm = None

    if len(head_x) > 1:
        head_path = np.abs(np.diff(head_x))
        head_variation = float(np.mean(head_path)) if len(head_path) else 0.0
    else:
        head_variation = 0.0

    if stance_widths:
        width_variation = float(np.std(stance_widths, ddof=0))
        balance_score = float(max(0.0, min(100.0, 100.0 - width_variation * 200.0)))
    else:
        balance_score = None

    if head_x:
        tempo_estimate = float(max(0.0, min(100.0, 100.0 - head_variation * 120.0)))
    else:
        tempo_estimate = None

    knee_flex_deg = float(np.mean(knee_flex_values)) if knee_flex_values else None

    return SwingMetrics(
        spine_angle_deg=spine_angle_deg,
        head_movement_cm=head_movement_cm,
        knee_flex_deg=knee_flex_deg,
        shoulder_turn_proxy=shoulder_turn_proxy,
        hip_turn_proxy=hip_turn_proxy,
        balance_score=balance_score,
        tempo_estimate=tempo_estimate,
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

    stance_width = None
    if left_ankle and right_ankle:
        stance_width = math.hypot(left_ankle[0] - right_ankle[0], left_ankle[1] - right_ankle[1])

    knee_flex_left = None
    if left_hip and left_knee and left_ankle:
        knee_flex_left = angle_between(left_hip, left_knee, left_ankle)

    knee_flex_right = None
    if right_hip and right_knee and right_ankle:
        knee_flex_right = angle_between(right_hip, right_knee, right_ankle)

    hip_tilt = None
    if left_hip and right_hip:
        hip_tilt = line_angle(left_hip, right_hip)

    shoulder_tilt = None
    if left_shoulder and right_shoulder:
        shoulder_tilt = line_angle(left_shoulder, right_shoulder)

    spine_angle = None
    if left_hip and right_hip and left_shoulder and right_shoulder:
        mid_hip = mid_point(left_hip, right_hip)
        mid_shoulder = mid_point(left_shoulder, right_shoulder)
        spine_angle = line_angle(mid_hip, mid_shoulder)

    head_x = nose[0] if nose else None
    head_y = nose[1] if nose else None

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


def _point_from_frame(frame: Dict, name: str) -> Optional[List[float]]:
    landmarks = frame.get("landmarks", {}) or {}
    point = get_point(landmarks, name)
    return point


def _mean_distance(points: List[Optional[List[float]]]) -> Optional[float]:
    valid_points = [point for point in points if point is not None]
    if len(valid_points) < 2:
        return None
    reference = valid_points[0]
    distances = [math.hypot(point[0] - reference[0], point[1] - reference[1]) for point in valid_points[1:]]
    return float(np.mean(distances)) if distances else None


def _path_length(points: List[Optional[List[float]]]) -> Optional[float]:
    valid_points = [point for point in points if point is not None]
    if len(valid_points) < 2:
        return None
    steps = [math.hypot(curr[0] - prev[0], curr[1] - prev[1]) for prev, curr in zip(valid_points, valid_points[1:])]
    return float(sum(steps)) if steps else None


def _mean_angle_deviation(angles: List[Optional[float]]) -> Optional[float]:
    valid_angles = [angle for angle in angles if angle is not None]
    if len(valid_angles) < 2:
        return None
    reference = valid_angles[0]
    return float(np.mean([abs(angle - reference) for angle in valid_angles[1:]]))


def compute_coordinate_movement_metrics(frames: List[Dict]) -> Dict[str, float]:
    if not frames:
        return {
            "head_movement_px": None,
            "hip_movement_px": None,
            "shoulder_turn_deg": None,
            "knee_movement_px": None,
            "foot_stability_px": None,
            "hand_path_px": None,
            "spine_angle_deg": None,
            "posture_changes": None,
        }

    head_points = [_point_from_frame(frame, "nose") for frame in frames]
    neck_points = [_point_from_frame(frame, "neck") for frame in frames]
    left_shoulder_points = [_point_from_frame(frame, "left_shoulder") for frame in frames]
    right_shoulder_points = [_point_from_frame(frame, "right_shoulder") for frame in frames]
    left_hip_points = [_point_from_frame(frame, "left_hip") for frame in frames]
    right_hip_points = [_point_from_frame(frame, "right_hip") for frame in frames]
    left_knee_points = [_point_from_frame(frame, "left_knee") for frame in frames]
    right_knee_points = [_point_from_frame(frame, "right_knee") for frame in frames]
    left_ankle_points = [_point_from_frame(frame, "left_ankle") for frame in frames]
    right_ankle_points = [_point_from_frame(frame, "right_ankle") for frame in frames]
    left_wrist_points = [_point_from_frame(frame, "left_wrist") for frame in frames]
    right_wrist_points = [_point_from_frame(frame, "right_wrist") for frame in frames]

    spine_angles = []
    shoulder_turns = []
    posture_changes = 0
    previous_spine_angle: Optional[float] = None

    for frame in frames:
        landmarks = frame.get("landmarks", {}) or {}
        left_hip = get_point(landmarks, "left_hip")
        right_hip = get_point(landmarks, "right_hip")
        left_shoulder = get_point(landmarks, "left_shoulder")
        right_shoulder = get_point(landmarks, "right_shoulder")
        if left_hip and right_hip and left_shoulder and right_shoulder:
            mid_hip = mid_point(left_hip, right_hip)
            mid_shoulder = mid_point(left_shoulder, right_shoulder)
            spine_angle = line_angle(mid_hip, mid_shoulder)
            spine_angles.append(spine_angle)
            if previous_spine_angle is not None and abs(spine_angle - previous_spine_angle) >= 5.0:
                posture_changes += 1
            previous_spine_angle = spine_angle

        if left_shoulder and right_shoulder:
            shoulder_turns.append(line_angle(left_shoulder, right_shoulder))

    head_movement_px = _mean_distance(head_points)
    hip_movement_px = _mean_distance([
        mid_point(left, right)
        for left, right in zip(left_hip_points, right_hip_points)
        if left is not None and right is not None
    ])
    shoulder_turn_deg = _mean_angle_deviation(shoulder_turns)
    knee_movement_px = _mean_distance([
        mid_point(left, right)
        for left, right in zip(left_knee_points, right_knee_points)
        if left is not None and right is not None
    ])
    foot_stability_px = _mean_distance([
        mid_point(left, right)
        for left, right in zip(left_ankle_points, right_ankle_points)
        if left is not None and right is not None
    ])
    hand_path_px = None
    left_hand_path = _path_length(left_wrist_points)
    right_hand_path = _path_length(right_wrist_points)
    if left_hand_path is not None and right_hand_path is not None:
        hand_path_px = float((left_hand_path + right_hand_path) / 2.0)
    elif left_hand_path is not None:
        hand_path_px = left_hand_path
    elif right_hand_path is not None:
        hand_path_px = right_hand_path

    if len(spine_angles) >= 2:
        posture_changes += sum(1 for prev, curr in zip(spine_angles, spine_angles[1:]) if abs(curr - prev) >= 5.0)

    spine_angle_deg = float(np.mean(spine_angles)) if spine_angles else None

    return {
        "head_movement_px": head_movement_px,
        "hip_movement_px": hip_movement_px,
        "shoulder_turn_deg": shoulder_turn_deg,
        "knee_movement_px": knee_movement_px,
        "foot_stability_px": foot_stability_px,
        "hand_path_px": hand_path_px,
        "spine_angle_deg": spine_angle_deg,
        "posture_changes": float(posture_changes),
        "neck_movement_px": _mean_distance(neck_points),
    }


def compute_swing_metrics(landmarks: List[Dict], config: Dict) -> Dict[str, float]:
    """Compute basic swing metrics from landmarks.

    Returns a dict for easy JSON serialization.
    """
    metrics = _compute_from_landmarks(landmarks, config)
    coordinate_metrics = compute_coordinate_movement_metrics(landmarks)
    combined = metrics.to_dict()
    combined.update(coordinate_metrics)
    return combined
