from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from webapp.utils.storage import ensure_dir


LOGGER = logging.getLogger(__name__)
BROWSER_COMPATIBLE_CODECS = {"h264", "avc1", "mp4v", "mp42", "isom", "iso2"}

BODY_POINTS = {
    "head": "nose",
    "neck": "neck",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}

SIMPLE_POINTS = {"head", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
FULL_POINTS = set(BODY_POINTS.values()) | {"neck", "mid_hip"}

BODY_COLOR = (245, 245, 245)
TRAIL_COLORS = {
    "head": BODY_COLOR,
    "left_hand": BODY_COLOR,
    "right_hand": BODY_COLOR,
    "hips": BODY_COLOR,
    "left_hip": BODY_COLOR,
    "right_hip": BODY_COLOR,
}
PROMINENT_JOINTS = {"head", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_ankle", "right_ankle"}
KEY_LANDMARKS = {
    "head",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}

SIMPLE_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

FULL_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

TRAIL_HISTORY_KEYS = {
    "head": "nose",
    "hips": "mid_hip",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
}

KEY_POSITION_LABELS = ["A", "T", "I", "F"]

DEBUG_FRAME_REASONS = {
    "rejected_landmark": "rejected_landmark",
    "ankle_jump": "ankle_jump",
    "pose_missing": "pose_missing",
    "left_right_swap": "left_right_swap",
    "interpolated": "interpolated",
}

SKELETON_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_shoulder", "left_hip"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

MARKER_SIZES = {"head": 3, "neck": 2, "left_shoulder": 3, "right_shoulder": 3, "left_elbow": 3, "right_elbow": 3, "left_wrist": 3, "right_wrist": 3, "left_hip": 3, "right_hip": 3, "left_knee": 3, "right_knee": 3, "left_ankle": 3, "right_ankle": 3, "mid_hip": 0}


def _frame_point(value: Dict[str, Any], width: int, height: int) -> Optional[tuple[int, int, float]]:
    if not value:
        return None
    x = value.get("x")
    y = value.get("y")
    confidence = value.get("visibility", value.get("confidence"))
    if x is None or y is None:
        return None

    try:
        x_value = float(x)
        y_value = float(y)
        confidence_value = float(confidence) if confidence is not None else 0.0
    except Exception:
        return None

    if 0.0 <= x_value <= 1.5 and 0.0 <= y_value <= 1.5:
        x_value *= float(width)
        y_value *= float(height)

    return (
        int(max(0, min(width - 1, round(x_value)))),
        int(max(0, min(height - 1, round(y_value)))),
        confidence_value,
    )


def _synthesized_points(landmarks: Dict[str, Dict[str, Any]], width: int, height: int) -> Dict[str, tuple[int, int, float]]:
    points: Dict[str, tuple[int, int, float]] = {}
    for name in set(BODY_POINTS.values()):
        point = _frame_point(landmarks.get(name, {}), width, height)
        if point is not None:
            points[name] = point

    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")

    if left_shoulder and right_shoulder:
        neck_x = int(round((left_shoulder[0] + right_shoulder[0]) / 2.0))
        neck_y = int(round((left_shoulder[1] + right_shoulder[1]) / 2.0))
        neck_conf = float((left_shoulder[2] + right_shoulder[2]) / 2.0)
        points["neck"] = (neck_x, neck_y, neck_conf)

    if left_hip and right_hip:
        hip_x = int(round((left_hip[0] + right_hip[0]) / 2.0))
        hip_y = int(round((left_hip[1] + right_hip[1]) / 2.0))
        hip_conf = float((left_hip[2] + right_hip[2]) / 2.0)
        points["mid_hip"] = (hip_x, hip_y, hip_conf)

    return points


def draw_skeleton(
    frame: np.ndarray,
    landmarks: Dict[str, Dict[str, Any]],
    *,
    confidence_threshold: float = 0.25,
    style: str = "simple",
) -> Dict[str, tuple[int, int, float]]:
    height, width = frame.shape[:2]
    points = _synthesized_points(landmarks, width, height)
    connections = [
        ("head", "neck"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]
    visible_points = {
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    alpha = 0.72

    overlay = frame.copy()

    for start_name, end_name in connections:
        start_point = points.get(start_name)
        end_point = points.get(end_name)
        if start_point is None or end_point is None:
            continue
        if start_point[2] < confidence_threshold and start_name != "neck":
            continue
        if end_point[2] < confidence_threshold and end_name != "neck":
            continue
        start_alpha, _ = _confidence_style(float(start_point[2]))
        end_alpha, _ = _confidence_style(float(end_point[2]))
        if min(start_alpha, end_alpha) <= 0.0:
            continue
        line_alpha = min(start_alpha, end_alpha)
        outer_color = tuple(int(255 * 0.22 * line_alpha) for _ in range(3))
        inner_color = tuple(int(BODY_COLOR[channel] * line_alpha) for channel in range(3))
        thickness = 2 if style == "simple" else 3
        cv2.line(overlay, start_point[:2], end_point[:2], outer_color, thickness + 2, cv2.LINE_AA)
        cv2.line(overlay, start_point[:2], end_point[:2], inner_color, thickness, cv2.LINE_AA)

    for name, point in points.items():
        if name not in visible_points:
            continue
        point_alpha, point_tint = _confidence_style(float(point[2]))
        if point_alpha <= 0.0:
            continue
        radius = MARKER_SIZES.get(name, 3)
        if name in PROMINENT_JOINTS:
            radius += 1
        color = tuple(int(BODY_COLOR[channel] * point_alpha) for channel in range(3))
        if point_tint != (0, 0, 0) and point_alpha < 1.0:
            color = tuple(int((color[channel] + point_tint[channel]) / 2.0) for channel in range(3))
        cv2.circle(overlay, point[:2], radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, point[:2], radius, color, -1, cv2.LINE_AA)
        if name in PROMINENT_JOINTS:
            cv2.circle(overlay, point[:2], radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    frame[:] = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

    return points


def draw_motion_trails(
    frame: np.ndarray,
    trail_history: Dict[str, Deque[tuple[int, int]]],
    *,
    max_length: int = 12,
    style: str = "simple",
) -> np.ndarray:
    trail_layer = np.zeros_like(frame)
    for trail_name, history in trail_history.items():
        points = list(history)
        if len(points) < 2:
            continue
        color = TRAIL_COLORS.get(trail_name, (180, 220, 255))
        for index in range(1, len(points)):
            age_ratio = index / float(len(points))
            fade_floor = 0.18 if style == "simple" else 0.12
            faded_color = tuple(int(channel * max(fade_floor, age_ratio)) for channel in color)
            thickness = 1 if style == "simple" else max(1, int(round(1 + age_ratio * 2)))
            cv2.line(trail_layer, points[index - 1], points[index], faded_color, thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(frame, 1.0, trail_layer, 0.55 if style == "simple" else 0.75, 0)
    return blended


def _approximate_key_frames(frame_count: int) -> Dict[str, int]:
    if frame_count <= 0:
        return {"A": 0, "T": 0, "I": 0, "F": 0}
    return {
        "A": 0,
        "T": max(0, int(round(frame_count * 0.30))),
        "I": max(0, int(round(frame_count * 0.60))),
        "F": max(0, frame_count - 1),
    }


def _draw_key_position(frame: np.ndarray, label: str) -> None:
    cv2.rectangle(frame, (12, 12), (38, 40), (6, 12, 20), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (12, 12), (38, 40), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _confidence_style(confidence: float) -> tuple[float, tuple[int, int, int]]:
    if confidence >= 0.8:
        return 1.0, (255, 255, 255)
    if confidence >= 0.5:
        return 0.65, (210, 225, 255)
    return 0.0, (0, 0, 0)


def _max_limb_jump(width: int, height: int) -> float:
    return max(90.0, float(min(width, height)) * 0.18)


def _limb_distance(a: tuple[int, int, float] | None, b: tuple[int, int, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _skeleton_consistency_score(points: Dict[str, tuple[int, int, float]]) -> float:
    pairs = [
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]
    distances = [distance for pair in pairs if (distance := _limb_distance(points.get(pair[0]), points.get(pair[1]))) is not None]
    if not distances:
        return 0.0
    torso = _limb_distance(points.get("neck"), points.get("mid_hip")) or float(np.mean(distances))
    if torso <= 0:
        return 0.0
    score = 100.0
    for distance in distances:
        ratio = distance / torso
        if ratio < 0.25 or ratio > 2.6:
            score -= 10.0
        elif ratio < 0.4 or ratio > 2.0:
            score -= 5.0
    return max(0.0, min(100.0, score))


def _pose_quality_score(points: Dict[str, tuple[int, int, float]]) -> float:
    if not points:
        return 0.0
    detected = sum(1 for point in points.values() if point[2] >= 0.5)
    average_confidence = float(np.mean([point[2] for point in points.values()])) if points else 0.0
    consistency = _skeleton_consistency_score(points)
    coverage_score = min(100.0, detected * 5.0)
    confidence_score = average_confidence * 100.0
    return round((coverage_score * 0.3) + (confidence_score * 0.4) + (consistency * 0.3), 2)


def _point_distance(a: tuple[int, int, float] | None, b: tuple[int, int, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detect_unrealistic_jump(previous: tuple[int, int, float] | None, current: tuple[int, int, float] | None, *, threshold_px: float = 90.0) -> bool:
    if previous is None or current is None:
        return False
    return _point_distance(previous, current) > threshold_px


def detect_left_right_swap(left_point: tuple[int, int, float] | None, right_point: tuple[int, int, float] | None, *, margin_px: float = 4.0) -> bool:
    if left_point is None or right_point is None:
        return False
    return left_point[0] > right_point[0] + margin_px


def _distance_score(point_a: tuple[int, int, float] | None, point_b: tuple[int, int, float] | None) -> float:
    if point_a is None or point_b is None:
        return float("inf")
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def validate_landmark(
    name: str,
    point: tuple[int, int, float] | None,
    previous_point: tuple[int, int, float] | None,
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    jump_threshold_px: float,
) -> tuple[bool, str | None]:
    if point is None:
        return False, "missing"
    x, y, confidence = point
    if not (0 <= x < width and 0 <= y < height):
        return False, "out_of_frame"
    if confidence < confidence_threshold:
        return False, "low_confidence"
    if detect_unrealistic_jump(previous_point, point, threshold_px=jump_threshold_px):
        return False, "jump"
    if name in {"left_ankle", "right_ankle", "left_knee", "right_knee", "left_hip", "right_hip"} and confidence < max(confidence_threshold, 0.35):
        return False, "lower_body_low_confidence"
    return True, None


def smooth_landmarks(
    current: Dict[str, tuple[int, int, float]],
    previous: Dict[str, tuple[int, int, float]],
    *,
    alpha: float = 0.6,
) -> Dict[str, tuple[int, int, float]]:
    if not previous:
        return current
    smoothed: Dict[str, tuple[int, int, float]] = {}
    for name, point in current.items():
        prev = previous.get(name)
        if prev is None:
            smoothed[name] = point
            continue
        x = int(round(prev[0] * alpha + point[0] * (1.0 - alpha)))
        y = int(round(prev[1] * alpha + point[1] * (1.0 - alpha)))
        confidence = float(max(prev[2], point[2]))
        smoothed[name] = (x, y, confidence)
    return smoothed


def interpolate_missing_landmarks(
    previous: Dict[str, tuple[int, int, float]],
    next_frame: Dict[str, tuple[int, int, float]] | None,
    *,
    ratio: float = 0.5,
) -> Dict[str, tuple[int, int, float]]:
    if not previous and not next_frame:
        return {}
    if not next_frame:
        return dict(previous)
    if not previous:
        return dict(next_frame)

    interpolated: Dict[str, tuple[int, int, float]] = {}
    for name in set(previous.keys()) | set(next_frame.keys()):
        prev = previous.get(name)
        nxt = next_frame.get(name)
        if prev is None:
            interpolated[name] = nxt
            continue
        if nxt is None:
            interpolated[name] = prev
            continue
        x = int(round(prev[0] + (nxt[0] - prev[0]) * ratio))
        y = int(round(prev[1] + (nxt[1] - prev[1]) * ratio))
        confidence = float((prev[2] + nxt[2]) / 2.0)
        interpolated[name] = (x, y, confidence)
    return interpolated


def correct_ankle_tracking(
    frame_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    knee_points: Dict[str, tuple[int, int, float]],
    hip_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    jump_threshold_px: float,
) -> tuple[Dict[str, tuple[int, int, float]], int, int]:
    corrected = dict(frame_points)
    ankle_corrections = 0
    swap_corrections = 0

    for side in ("left", "right"):
        ankle_name = f"{side}_ankle"
        ankle = corrected.get(ankle_name)
        prev_ankle = previous_points.get(ankle_name)

        valid, reason = validate_landmark(
            ankle_name,
            ankle,
            prev_ankle,
            width=width,
            height=height,
            confidence_threshold=confidence_threshold,
            jump_threshold_px=jump_threshold_px,
        )
        if valid:
            continue

        ankle_corrections += 1
        if reason == "jump" and prev_ankle is not None:
            corrected[ankle_name] = prev_ankle
            continue

    left_ankle = corrected.get("left_ankle")
    right_ankle = corrected.get("right_ankle")
    left_knee = knee_points.get("left_knee")
    right_knee = knee_points.get("right_knee")

    if left_ankle is not None and right_ankle is not None:
        same_knee_distance = _distance_score(left_ankle, left_knee) + _distance_score(right_ankle, right_knee)
        swapped_knee_distance = _distance_score(left_ankle, right_knee) + _distance_score(right_ankle, left_knee)
        same_prev_distance = _distance_score(left_ankle, previous_points.get("left_ankle")) + _distance_score(right_ankle, previous_points.get("right_ankle"))
        swapped_prev_distance = _distance_score(left_ankle, previous_points.get("right_ankle")) + _distance_score(right_ankle, previous_points.get("left_ankle"))
        if swapped_knee_distance + 10.0 < same_knee_distance or swapped_prev_distance + 10.0 < same_prev_distance:
            swap_corrections += 1
            corrected["left_ankle"], corrected["right_ankle"] = corrected["right_ankle"], corrected["left_ankle"]
            left_ankle = corrected.get("left_ankle")
            right_ankle = corrected.get("right_ankle")

    if detect_left_right_swap(left_ankle, right_ankle):
        swap_corrections += 1
        corrected["left_ankle"], corrected["right_ankle"] = corrected["right_ankle"], corrected["left_ankle"]

    return corrected, ankle_corrections, swap_corrections


def _stabilize_key_landmarks(
    current_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    current_index: int,
    previous_index: int | None,
    max_hold_frames: int = 2,
) -> Dict[str, tuple[int, int, float]]:
    if not previous_points or previous_index is None:
        return current_points
    if current_index - previous_index > max_hold_frames:
        return current_points

    stabilized = dict(current_points)
    for landmark_name in KEY_LANDMARKS:
        if landmark_name not in stabilized and landmark_name in previous_points:
            stabilized[landmark_name] = previous_points[landmark_name]
    return stabilized


def _extract_point_map(landmarks: Dict[str, Dict[str, Any]], width: int, height: int) -> Dict[str, tuple[int, int, float]]:
    return {
        name: point
        for name in set(BODY_POINTS.values())
        if (point := _frame_point(landmarks.get(name, {}), width, height)) is not None
    }


def _build_dense_pose_series(
    video_frame_count: int,
    pose_frames: list[Dict[str, Any]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    jump_threshold_px: float,
    style: str,
    tracking_mode: str = "smoothed",
    debug_frames_dir: Path | None = None,
) -> tuple[Dict[int, Dict[str, tuple[int, int, float]]], Dict[str, int], Dict[str, int]]:
    if not pose_frames:
        return {}, {"frames_interpolated": 0, "frames_rejected": 0, "frames_missing": 0}, {"ankle_corrections": 0, "swap_corrections": 0}

    frame_points: Dict[int, Dict[str, tuple[int, int, float]]] = {}
    frame_reasons: Dict[int, str] = {}
    for pose_frame in pose_frames:
        frame_index = int(pose_frame.get("frame_index", 0) or 0)
        frame_points[frame_index] = _extract_point_map(pose_frame.get("landmarks", {}), width, height)

    dense_series: Dict[int, Dict[str, tuple[int, int, float]]] = {}
    metrics = {"frames_interpolated": 0, "frames_rejected": 0, "frames_missing": 0}
    corrections = {"ankle_corrections": 0, "swap_corrections": 0}
    previous_valid: Dict[str, tuple[int, int, float]] = {}
    previous_valid_index: int | None = None

    if tracking_mode == "raw":
        for current_index in range(max(0, video_frame_count)):
            dense_series[current_index] = frame_points.get(current_index, {})
            if current_index not in frame_points:
                metrics["frames_missing"] += 1
        return dense_series, metrics, corrections

    ordered_indices = sorted(frame_points.keys())
    for current_index in range(max(0, video_frame_count)):
        source_points = frame_points.get(current_index)
        if source_points is None:
            metrics["frames_missing"] += 1
            prev_candidates = [idx for idx in ordered_indices if idx < current_index]
            next_candidates = [idx for idx in ordered_indices if idx > current_index]
            prev_points = frame_points.get(prev_candidates[-1], {}) if prev_candidates else {}
            next_points = frame_points.get(next_candidates[0], {}) if next_candidates else {}
            gap = (next_candidates[0] - prev_candidates[-1]) if prev_candidates and next_candidates else None
            if prev_points and next_points and gap is not None and gap <= 2:
                source_points = interpolate_missing_landmarks(prev_points, next_points, ratio=0.5)
                metrics["frames_interpolated"] += 1
                frame_reasons[current_index] = DEBUG_FRAME_REASONS["interpolated"]
            else:
                source_points = {}
                frame_reasons[current_index] = DEBUG_FRAME_REASONS["pose_missing"]

        valid_points: Dict[str, tuple[int, int, float]] = {}
        frame_rejected = False

        for name, point in source_points.items():
            prev_point = previous_valid.get(name)
            valid, _reason = validate_landmark(
                name,
                point,
                prev_point,
                width=width,
                height=height,
                confidence_threshold=confidence_threshold,
                jump_threshold_px=jump_threshold_px,
            )
            if not valid:
                frame_rejected = True
                continue
            valid_points[name] = point

        knee_points = {name: valid_points.get(name) for name in ("left_knee", "right_knee") if valid_points.get(name) is not None}
        hip_points = {name: valid_points.get(name) for name in ("left_hip", "right_hip") if valid_points.get(name) is not None}

        corrected_points, ankle_corrections, swap_corrections = correct_ankle_tracking(
            valid_points,
            previous_valid,
            knee_points,
            hip_points,
            width=width,
            height=height,
            confidence_threshold=confidence_threshold,
            jump_threshold_px=jump_threshold_px,
        )
        corrections["ankle_corrections"] += ankle_corrections
        corrections["swap_corrections"] += swap_corrections

        if frame_rejected:
            metrics["frames_rejected"] += 1
            frame_reasons[current_index] = DEBUG_FRAME_REASONS["rejected_landmark"]

        if corrected_points:
            if previous_valid:
                corrected_points = smooth_landmarks(corrected_points, previous_valid, alpha=0.85 if style == "simple" else 0.95)
            corrected_points = _stabilize_key_landmarks(
                corrected_points,
                previous_valid,
                current_index=current_index,
                previous_index=previous_valid_index,
                max_hold_frames=2,
            )
            previous_valid = corrected_points
            previous_valid_index = current_index
        elif previous_valid:
            corrected_points = _stabilize_key_landmarks(
                {},
                previous_valid,
                current_index=current_index,
                previous_index=previous_valid_index,
                max_hold_frames=2,
            )

        dense_series[current_index] = corrected_points

        if debug_frames_dir and frame_reasons.get(current_index) in {DEBUG_FRAME_REASONS["rejected_landmark"], DEBUG_FRAME_REASONS["ankle_jump"], DEBUG_FRAME_REASONS["pose_missing"], DEBUG_FRAME_REASONS["left_right_swap"], DEBUG_FRAME_REASONS["interpolated"]}:
            debug_frames_dir.mkdir(parents=True, exist_ok=True)

    return dense_series, metrics, corrections


def save_overlay_video(frames: Iterable[np.ndarray], output_path: str | Path, fps: float, frame_size: tuple[int, int]) -> str:
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    codec_candidates = ["avc1", "H264", "mp4v"]
    writer = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_file), fourcc, float(fps or 30.0), frame_size)
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 30.0), frame_size)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return str(output_file)


def validate_overlay_video(overlay_path: str | Path, source_video_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(overlay_path)
    result: Dict[str, Any] = {
        "valid": False,
        "exists": path.exists() and path.is_file(),
        "size_mb": 0.0,
        "frame_count": 0,
        "fps": None,
        "width": None,
        "height": None,
        "codec": None,
        "can_open_with_opencv": False,
        "browser_compatible": False,
        "overlay_path": str(path),
    }
    if source_video_path is not None:
        result["source_video_path"] = str(source_video_path)

    if not result["exists"]:
        return result

    try:
        result["size_mb"] = round(path.stat().st_size / (1024.0 * 1024.0), 3)
    except Exception:
        result["size_mb"] = 0.0

    capture = cv2.VideoCapture(str(path))
    result["can_open_with_opencv"] = bool(capture.isOpened())
    if not capture.isOpened():
        capture.release()
        return result

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ").lower() or None
    capture.release()

    result.update(
        {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": codec,
            "browser_compatible": bool((path.suffix.lower() in {".mp4", ".mov", ".webm"}) and (codec in BROWSER_COMPATIBLE_CODECS if codec else False)),
        }
    )
    result["valid"] = bool(result["exists"] and result["can_open_with_opencv"] and frame_count > 0 and fps and width and height)
    return result


def load_pose_landmarks(csv_path: str | Path) -> list[Dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return []

    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Unable to load pose landmarks from %s: %s", path, exc)
        return []

    grouped: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"landmarks": {}})
    for _, row in frame.iterrows():
        frame_index = int(row.get("frame_number", row.get("frame_index", 0)) or 0)
        timestamp = row.get("timestamp", row.get("timestamp_sec"))
        entry = grouped[frame_index]
        entry["frame_index"] = frame_index
        entry["timestamp_sec"] = None if timestamp is None else float(timestamp)
        entry["landmarks"][str(row.get("keypoint_name", ""))] = {
            "x": row.get("x"),
            "y": row.get("y"),
            "visibility": row.get("confidence", 0.0),
        }

    return [grouped[index] for index in sorted(grouped)]


def generate_pose_overlay(
    video_path: str | Path,
    pose_frames: Optional[list[Dict[str, Any]]] = None,
    output_dir: str | Path | None = None,
    *,
    pose_landmarks_path: str | Path | None = None,
    output_name: Optional[str] = None,
    trail_length: int = 12,
    confidence_threshold: float = 0.25,
    overlay_style: str = "simple",
    tracking_mode: str = "smoothed",
) -> Dict[str, Any]:
    video_file = Path(video_path)
    LOGGER.info("OVERLAY START video=%s", video_file)
    overlay_root = ensure_dir(output_dir or (video_file.parent / "overlays"))
    overlay_path = overlay_root / (output_name or f"{video_file.stem}_overlay.mp4")

    if not pose_frames and pose_landmarks_path is not None:
        pose_frames = load_pose_landmarks(pose_landmarks_path)

    pose_frames = sorted(pose_frames or [], key=lambda item: int(item.get("frame_index", 0) or 0))
    if not video_file.exists() or not video_file.is_file():
        LOGGER.info("OVERLAY FAILED video missing=%s", video_file)
        return {
            "status": "error",
            "message": "Pose tracking could not be generated from this video.",
            "warnings": ["Video file could not be found."],
            "overlay_video_path": None,
            "overlay_video_url": None,
            "overlay_validation": validate_overlay_video(overlay_path, video_file),
            "frames_processed": 0,
            "frames_with_pose": 0,
            "detection_rate": None,
            "average_confidence": None,
        }

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        capture.release()
        LOGGER.info("OVERLAY FAILED unable to open source video=%s", video_file)
        return {
            "status": "error",
            "message": "Pose tracking could not be generated from this video.",
            "warnings": [
                "OpenCV could not open the uploaded video.",
                "Body not fully visible.",
                "Video too blurry.",
                "Camera angle unsuitable.",
            ],
            "overlay_video_path": None,
            "overlay_video_url": None,
            "overlay_validation": validate_overlay_video(overlay_path, video_file),
            "frames_processed": 0,
            "frames_with_pose": 0,
            "detection_rate": None,
            "average_confidence": None,
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_size = (width, height)

    if width <= 0 or height <= 0:
        capture.release()
        LOGGER.info("OVERLAY FAILED invalid source dimensions video=%s width=%s height=%s", video_file, width, height)
        return {
            "status": "error",
            "message": "Pose tracking could not be generated from this video.",
            "warnings": ["Video dimensions could not be read."],
            "overlay_video_path": None,
            "overlay_video_url": None,
            "overlay_validation": validate_overlay_video(overlay_path, video_file),
            "frames_processed": 0,
            "frames_with_pose": 0,
            "detection_rate": None,
            "average_confidence": None,
        }

    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rendered_frames: list[np.ndarray] = []
    frames_processed = 0
    frames_with_pose = 0
    confidence_values: list[float] = []
    pose_quality_values: list[float] = []
    head_tracked_frames = 0
    hands_tracked_frames = 0
    feet_tracked_frames = 0
    tracking_debug_frames: list[np.ndarray] = []
    pose_frame_count = sum(1 for frame in pose_frames if frame.get("landmarks"))
    raw_frame_indices = {int(frame.get("frame_index", 0) or 0) for frame in pose_frames}
    dense_series, dense_metrics, correction_metrics = _build_dense_pose_series(
        video_frame_count,
        pose_frames,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
        jump_threshold_px=max(60.0, min(width, height) * 0.12),
        style=overlay_style,
        tracking_mode=tracking_mode,
    )
    key_frames = _approximate_key_frames(video_frame_count or len(dense_series) or len(pose_frames))
    debug_frames_dir = overlay_root / "debug_frames" / video_file.stem / overlay_style
    debug_frames_dir.mkdir(parents=True, exist_ok=True)
    debug_frame_paths: list[str] = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        frames_processed += 1
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frames_processed) - 1
        frame_points = dense_series.get(frame_index, {})

        if frame_points:
            frames_with_pose += 1
            point_map = {
                name: {"x": point[0], "y": point[1], "visibility": point[2]}
                for name, point in frame_points.items()
            }
            points = draw_skeleton(frame, point_map, confidence_threshold=confidence_threshold, style=overlay_style)
            raw_landmark_count = len(point_map)
            average_frame_confidence = float(np.mean([point[2] for point in frame_points.values()])) if frame_points else 0.0
            pose_quality = _pose_quality_score(points)
            pose_quality_values.append(pose_quality)

            if points.get("head") and points["head"][2] >= 0.5:
                head_tracked_frames += 1

            if points.get("left_wrist") and points.get("right_wrist") and points["left_wrist"][2] >= 0.5 and points["right_wrist"][2] >= 0.5:
                hands_tracked_frames += 1
            if points.get("left_ankle") and points.get("right_ankle") and points["left_ankle"][2] >= 0.5 and points["right_ankle"][2] >= 0.5:
                feet_tracked_frames += 1

            if points:
                for point in points.values():
                    if point[2] >= confidence_threshold:
                        confidence_values.append(float(point[2]))
            interpolated_landmarks = max(0, len(frame_points) - raw_landmark_count) if frame_index not in raw_frame_indices and frame_points else 0
            estimated_landmarks = max(0, len(points) - raw_landmark_count) if tracking_mode != "raw" else 0
            LOGGER.info(
                "POSE TRACKING frame=%s tracking_mode=%s raw_landmarks=%s rendered_landmarks=%s interpolated_landmarks=%s estimated_landmarks=%s avg_confidence=%.3f pose_quality=%.2f",
                frame_index,
                tracking_mode,
                raw_landmark_count,
                len(points),
                interpolated_landmarks,
                estimated_landmarks,
                average_frame_confidence,
                pose_quality,
            )

            if overlay_style == "simple":
                key_label = None
                if frame_index <= key_frames["A"]:
                    key_label = "A"
                elif frame_index <= key_frames["T"]:
                    key_label = "T"
                elif frame_index <= key_frames["I"]:
                    key_label = "I"
                elif frame_index >= key_frames["F"]:
                    key_label = "F"
                if key_label:
                    _draw_key_position(frame, key_label)
            else:
                tracking_debug_frame = frame.copy()
                debug_lines = [
                    f"frame {frame_index + 1}",
                    f"raw {raw_landmark_count}",
                    f"interp {interpolated_landmarks}",
                    f"quality {pose_quality:.1f}",
                ]
                for debug_index, line in enumerate(debug_lines):
                    cv2.putText(
                        tracking_debug_frame,
                        line,
                        (16, 24 + (debug_index * 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                for debug_index, (name, point) in enumerate(sorted(points.items())):
                    if debug_index >= 10:
                        break
                    cv2.putText(
                        tracking_debug_frame,
                        f"{name}:{point[2]:.2f}",
                        (16, height - 160 + (debug_index * 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                tracking_debug_frames.append(tracking_debug_frame)

        if frames_processed == 1:
            cv2.putText(
                frame,
                "Pose Overlay",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if frame_index in dense_series and (frame_index % 24 == 0 or frame_index in {0, key_frames["A"], key_frames["T"], key_frames["I"], key_frames["F"]}):
            debug_frame = frame.copy()
            cv2.putText(
                debug_frame,
                f"frame {frame_index + 1} {overlay_style}",
                (18, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            debug_path = debug_frames_dir / f"{video_file.stem}_{overlay_style}_{frame_index:05d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)
            debug_frame_paths.append(str(debug_path))

        rendered_frames.append(frame)

    capture.release()

    saved_path = save_overlay_video(rendered_frames, overlay_path, fps, frame_size)
    overlay_validation = validate_overlay_video(saved_path, video_file)
    tracking_debug_video_path = None
    tracking_debug_video_url = None
    if tracking_debug_frames:
        tracking_debug_path = overlay_root / f"{video_file.stem}_tracking_debug.mp4"
        tracking_debug_video_path = save_overlay_video(tracking_debug_frames, tracking_debug_path, fps, frame_size)
        tracking_debug_video_url = f"/outputs/overlays/{Path(tracking_debug_video_path).name}"
    LOGGER.info(
        "OVERLAY VIDEO CREATED path=%s size_mb=%s frames=%s",
        saved_path,
        overlay_validation.get("size_mb"),
        overlay_validation.get("frame_count"),
    )
    LOGGER.info("OVERLAY VALIDATION %s", overlay_validation)
    detection_rate = round((frames_with_pose / frames_processed) * 100.0, 2) if frames_processed else None
    average_confidence = round(float(np.mean(confidence_values)), 3) if confidence_values else None
    average_pose_quality = round(float(np.mean(pose_quality_values)), 2) if pose_quality_values else 0.0
    quality_metrics = {
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "frames_interpolated": dense_metrics.get("frames_interpolated", 0),
        "frames_rejected": dense_metrics.get("frames_rejected", 0),
        "frames_missing": dense_metrics.get("frames_missing", 0),
        "head_tracked_frames": head_tracked_frames,
        "ankle_corrections": correction_metrics.get("ankle_corrections", 0),
        "swap_corrections": correction_metrics.get("swap_corrections", 0),
        "debug_frame_count": len(debug_frame_paths),
        "head_tracked_rate": round((head_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "hands_tracked_frames": hands_tracked_frames,
        "feet_tracked_frames": feet_tracked_frames,
        "hands_tracked_rate": round((hands_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "feet_tracked_rate": round((feet_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "average_pose_quality_score": average_pose_quality,
        "tracking_mode": tracking_mode,
    }

    if not pose_frame_count:
        LOGGER.info("POSE FAILED video=%s pose_frames=%s", video_file, pose_frame_count)
        return {
            "status": "error",
            "message": "Pose tracking could not be generated from this video.",
            "warnings": [
                "Body not fully visible.",
                "Video too blurry.",
                "Camera angle unsuitable.",
            ],
            "overlay_video_path": saved_path,
            "overlay_video_url": f"/outputs/overlays/{Path(saved_path).name}" if saved_path else None,
            "overlay_validation": overlay_validation,
            "tracking_debug_video_path": tracking_debug_video_path,
            "tracking_debug_video_url": tracking_debug_video_url,
            "frames_processed": frames_processed,
            "frames_with_pose": 0,
            "detection_rate": detection_rate,
            "average_confidence": average_confidence,
            "quality_metrics": quality_metrics,
        }

    LOGGER.info("POSE SUCCESS video=%s pose_frames=%s detections=%s", video_file, pose_frame_count, frames_with_pose)
    return {
        "status": "success",
        "message": "Pose overlay generated successfully.",
        "warnings": [],
        "overlay_video_path": saved_path,
        "overlay_video_url": f"/outputs/overlays/{Path(saved_path).name}" if saved_path else None,
        "overlay_validation": overlay_validation,
        "overlay_style": overlay_style,
        "tracking_mode": tracking_mode,
        "tracking_debug_video_path": tracking_debug_video_path,
        "tracking_debug_video_url": tracking_debug_video_url,
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "detection_rate": detection_rate,
        "average_confidence": average_confidence,
        "pose_frame_count": pose_frame_count,
        "video_fps": fps,
        "frame_size": {"width": width, "height": height},
        "quality_metrics": quality_metrics,
        "debug_frame_paths": debug_frame_paths[:12],
    }