from __future__ import annotations

import json
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

LANDMARK_INDEX_MAPPING = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

LOWER_BODY_LANDMARK_NAMES = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
HAND_LANDMARK_NAMES = ("left_wrist", "right_wrist")

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
    "lower_body_swap_corrected": "lower_body_swap_corrected",
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
HAND_STALE_FRAME_THRESHOLD = 2
HAND_MIN_REFRESH_CONFIDENCE = 0.15


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


def _hand_smoothing_alpha(
    name: str,
    current: tuple[int, int, float] | None,
    previous: tuple[int, int, float] | None,
    *,
    width: int,
    height: int,
    style: str,
) -> float:
    if name not in HAND_LANDMARK_NAMES or current is None or previous is None:
        return 0.85 if style == "simple" else 0.92

    movement = _point_distance(previous, current)
    slow_threshold = max(8.0, float(min(width, height)) * 0.018)
    fast_threshold = max(20.0, float(min(width, height)) * 0.04)

    if movement >= fast_threshold:
        return 0.22
    if movement >= slow_threshold:
        return 0.42
    return 0.68 if style == "simple" else 0.74


def _log_point_triplet(prefix: str, point: tuple[int, int, float] | None) -> dict[str, float | int | None]:
    payload = _landmark_payload(point)
    return {
        f"{prefix}_x": payload["x"],
        f"{prefix}_y": payload["y"],
        f"{prefix}_confidence": payload["confidence"],
    }


def _debug_row_to_points(debug_row: Dict[str, Any], prefix: str) -> Dict[str, tuple[int, int, float]]:
    points: Dict[str, tuple[int, int, float]] = {}
    for side in ("left", "right"):
        for part in ("wrist", "elbow", "shoulder"):
            x = debug_row.get(f"{prefix}_{side}_{part}_x")
            y = debug_row.get(f"{prefix}_{side}_{part}_y")
            confidence = debug_row.get(f"{prefix}_{side}_{part}_confidence")
            if x is None or y is None:
                continue
            points[f"{side}_{part}"] = (int(x), int(y), float(confidence or 0.0))
    return points


def _person_bbox_from_points(points: Dict[str, tuple[int, int, float]], width: int, height: int) -> tuple[int, int, int, int] | None:
    coords: list[tuple[int, int]] = []
    for name, point in points.items():
        if name in HAND_LANDMARK_NAMES or point is None:
            continue
        coords.append((int(point[0]), int(point[1])))
    if not coords:
        return None

    xs = [value[0] for value in coords]
    ys = [value[1] for value in coords]
    margin_x = max(16, int(round(width * 0.08)))
    margin_y = max(16, int(round(height * 0.10)))
    left = max(0, min(xs) - margin_x)
    top = max(0, min(ys) - margin_y)
    right = min(width - 1, max(xs) + margin_x)
    bottom = min(height - 1, max(ys) + margin_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _point_inside_bbox(point: tuple[int, int, float] | None, bbox: tuple[int, int, int, int] | None, *, margin_px: float = 0.0) -> bool:
    if point is None or bbox is None:
        return True
    left, top, right, bottom = bbox
    x, y = point[:2]
    return (left - margin_px) <= x <= (right + margin_px) and (top - margin_px) <= y <= (bottom + margin_px)


def _normalize_vector(dx: float, dy: float) -> tuple[float, float]:
    length = float(np.hypot(dx, dy))
    if length <= 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def _estimate_wrist_from_arm(
    wrist_name: str,
    current_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
) -> tuple[int, int, float] | None:
    side = wrist_name.split("_")[0]
    current_elbow = current_points.get(f"{side}_elbow") or previous_points.get(f"{side}_elbow")
    current_shoulder = current_points.get(f"{side}_shoulder") or previous_points.get(f"{side}_shoulder")
    previous_wrist = previous_points.get(wrist_name)
    previous_elbow = previous_points.get(f"{side}_elbow")

    if current_elbow is None or previous_wrist is None:
        return None

    forearm_length = _distance_score(previous_wrist, previous_elbow) if previous_elbow is not None else _distance_score(previous_wrist, current_elbow)
    if not np.isfinite(forearm_length) or forearm_length <= 1.0:
        forearm_length = max(18.0, float(min(width, height)) * 0.08)

    if current_shoulder is not None and previous_elbow is not None:
        upper_vec = _normalize_vector(current_elbow[0] - current_shoulder[0], current_elbow[1] - current_shoulder[1])
        previous_forearm = _normalize_vector(previous_wrist[0] - previous_elbow[0], previous_wrist[1] - previous_elbow[1])
        blend_x = (upper_vec[0] * 0.45) + (previous_forearm[0] * 0.55)
        blend_y = (upper_vec[1] * 0.45) + (previous_forearm[1] * 0.55)
        direction_x, direction_y = _normalize_vector(blend_x, blend_y)
        if direction_x == 0.0 and direction_y == 0.0:
            direction_x, direction_y = previous_forearm
    else:
        direction_x, direction_y = _normalize_vector(previous_wrist[0] - current_elbow[0], previous_wrist[1] - current_elbow[1])

    if direction_x == 0.0 and direction_y == 0.0:
        return None

    estimated_x = int(round(current_elbow[0] + (direction_x * forearm_length)))
    estimated_y = int(round(current_elbow[1] + (direction_y * forearm_length)))
    confidence = float(max(0.22, min(0.62, previous_wrist[2] * 0.8)))
    return estimated_x, estimated_y, confidence


def _wrist_arm_geometry_ok(
    wrist: tuple[int, int, float] | None,
    elbow: tuple[int, int, float] | None,
    shoulder: tuple[int, int, float] | None,
    *,
    reference: float,
) -> bool:
    if wrist is None or elbow is None or shoulder is None:
        return False
    upper_arm = _distance_score(shoulder, elbow)
    forearm = _distance_score(elbow, wrist)
    arm_total = upper_arm + forearm
    if not np.isfinite(arm_total):
        return False
    if arm_total < reference * 0.35 or arm_total > reference * 2.7:
        return False
    if forearm > reference * 1.6:
        return False
    return True


def _refresh_stale_hand_tracking(
    current_points: Dict[str, tuple[int, int, float]],
    raw_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    stale_state: Dict[str, int],
    *,
    person_bbox: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    confidence_threshold: float,
) -> tuple[Dict[str, tuple[int, int, float]], int, Dict[str, Any]]:
    refreshed = dict(current_points)
    stale_corrections = 0
    debug_flags: Dict[str, Any] = {
        "stale_hand_tracking": False,
        "stale_background_detected": False,
        "left_stale_hand_tracking": False,
        "right_stale_hand_tracking": False,
        "left_stale_background_detected": False,
        "right_stale_background_detected": False,
        "left_stale_hand_reason": None,
        "right_stale_hand_reason": None,
    }

    wrist_hold_threshold = max(3.0, float(min(width, height)) * 0.008)
    arm_motion_threshold = max(6.0, float(min(width, height)) * 0.015)
    refresh_confidence_threshold = max(HAND_MIN_REFRESH_CONFIDENCE, confidence_threshold * 0.5)

    for side in ("left", "right"):
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        shoulder_name = f"{side}_shoulder"
        current_wrist = refreshed.get(wrist_name)
        previous_wrist = previous_points.get(wrist_name)
        raw_wrist = raw_points.get(wrist_name)
        current_elbow = refreshed.get(elbow_name)
        previous_elbow = previous_points.get(elbow_name)
        current_shoulder = refreshed.get(shoulder_name)
        previous_shoulder = previous_points.get(shoulder_name)

        if current_wrist is None or previous_wrist is None or current_elbow is None or current_shoulder is None:
            stale_state[side] = 0
            continue

        wrist_hold = _point_distance(current_wrist, previous_wrist) <= wrist_hold_threshold
        elbow_motion = _point_distance(current_elbow, previous_elbow) if previous_elbow is not None else 0.0
        shoulder_motion = _point_distance(current_shoulder, previous_shoulder) if previous_shoulder is not None else 0.0
        arm_motion = max(elbow_motion, shoulder_motion) >= arm_motion_threshold
        raw_changed = raw_wrist is not None and _point_distance(raw_wrist, previous_wrist) > wrist_hold_threshold
        wrist_inside_person = _point_inside_bbox(current_wrist, person_bbox, margin_px=max(10.0, wrist_hold_threshold * 2.0))
        raw_inside_person = _point_inside_bbox(raw_wrist, person_bbox, margin_px=max(10.0, wrist_hold_threshold * 2.0))
        background_stuck = wrist_hold and arm_motion and (not wrist_inside_person or not raw_changed or not raw_inside_person)

        if background_stuck:
            stale_state[side] += 1
        else:
            stale_state[side] = 0

        if stale_state[side] < HAND_STALE_FRAME_THRESHOLD:
            continue

        replacement = None
        reason = ""
        if raw_wrist is not None and raw_wrist[2] >= refresh_confidence_threshold and raw_inside_person:
            replacement = raw_wrist
            reason = "raw_refresh"

        if replacement is None:
            refreshed.pop(wrist_name, None)
            debug_flags["stale_background_detected"] = True
            debug_flags[f"{side}_stale_background_detected"] = True
            debug_flags[f"{side}_stale_hand_reason"] = "background_stuck"
            stale_state[side] = 0
            stale_corrections += 1
            continue

        refreshed[wrist_name] = replacement
        stale_corrections += 1
        debug_flags["stale_hand_tracking"] = True
        debug_flags[f"{side}_stale_hand_tracking"] = True
        debug_flags[f"{side}_stale_hand_reason"] = reason
        stale_state[side] = 0

    return refreshed, stale_corrections, debug_flags


def _landmark_payload(point: tuple[int, int, float] | None) -> dict[str, float | int | None]:
    if point is None:
        return {"x": None, "y": None, "confidence": None}
    return {"x": int(point[0]), "y": int(point[1]), "confidence": float(point[2])}


def _lower_body_assignment_frame(points: Dict[str, tuple[int, int, float]], assignment: str) -> Dict[str, tuple[int, int, float]]:
    resolved = dict(points)
    if assignment == "B":
        resolved["left_knee"], resolved["right_knee"] = resolved.get("right_knee"), resolved.get("left_knee")
        resolved["left_ankle"], resolved["right_ankle"] = resolved.get("right_ankle"), resolved.get("left_ankle")
    return resolved


def _lower_body_reference_length(points: Dict[str, tuple[int, int, float]], width: int, height: int) -> float:
    reference = _limb_distance(points.get("neck"), points.get("mid_hip"))
    if reference is None:
        reference = _limb_distance(points.get("left_hip"), points.get("right_hip"))
    if reference is None:
        reference = float(min(width, height)) * 0.18
    return max(24.0, float(reference))


def _score_lower_body_candidate(
    points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
) -> tuple[float, bool, list[str]]:
    required = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
    missing = [name for name in required if points.get(name) is None]
    if missing:
        return float("inf"), False, [f"missing:{','.join(sorted(missing))}"]

    reference = _lower_body_reference_length(points, width, height)
    score = 0.0
    reasons: list[str] = []
    valid = True
    vertical_margin = max(4.0, reference * 0.08)
    side_margin = max(6.0, reference * 0.12)
    teleport_limit = max(60.0, float(min(width, height)) * 0.22)

    for side in ("left", "right"):
        opposite_side = "right" if side == "left" else "left"
        hip = points.get(f"{side}_hip")
        knee = points.get(f"{side}_knee")
        ankle = points.get(f"{side}_ankle")
        opposite_hip = points.get(f"{opposite_side}_hip")
        opposite_knee = points.get(f"{opposite_side}_knee")
        opposite_ankle = points.get(f"{opposite_side}_ankle")
        previous_hip = previous_points.get(f"{side}_hip")
        previous_knee = previous_points.get(f"{side}_knee")
        previous_ankle = previous_points.get(f"{side}_ankle")

        if hip is None or knee is None or ankle is None:
            return float("inf"), False, [f"missing_side:{side}"]

        hip_to_knee = _distance_score(hip, knee)
        knee_to_ankle = _distance_score(knee, ankle)
        same_hip_distance = hip_to_knee
        opposite_hip_distance = _distance_score(opposite_hip, knee)
        same_ankle_distance = knee_to_ankle
        opposite_ankle_distance = _distance_score(opposite_knee, ankle)

        if knee[1] <= hip[1] - vertical_margin:
            valid = False
            score += reference * 3.0
            reasons.append(f"{side}_knee_above_hip")
        if ankle[1] <= knee[1] - vertical_margin:
            valid = False
            score += reference * 3.0
            reasons.append(f"{side}_ankle_above_knee")

        if hip_to_knee > reference * 1.9 or knee_to_ankle > reference * 2.2:
            valid = False
            score += (hip_to_knee + knee_to_ankle) * 0.5
            reasons.append(f"{side}_limb_length_out_of_range")

        if same_hip_distance > opposite_hip_distance + side_margin:
            score += (same_hip_distance - opposite_hip_distance) * 2.0
            reasons.append(f"{side}_knee_closer_to_opposite_hip")

        if same_ankle_distance > opposite_ankle_distance + side_margin:
            score += (same_ankle_distance - opposite_ankle_distance) * 2.0
            reasons.append(f"{side}_foot_closer_to_opposite_knee")

        if previous_knee is not None and _distance_score(knee, previous_knee) > teleport_limit:
            score += _distance_score(knee, previous_knee) * 1.5
            reasons.append(f"{side}_knee_teleport")
        if previous_ankle is not None and _distance_score(ankle, previous_ankle) > teleport_limit:
            score += _distance_score(ankle, previous_ankle) * 1.5
            reasons.append(f"{side}_foot_teleport")
        if previous_hip is not None and _distance_score(hip, previous_hip) > teleport_limit:
            score += _distance_score(hip, previous_hip) * 0.5
            reasons.append(f"{side}_hip_jump")

        if hip[2] < confidence_threshold or knee[2] < confidence_threshold or ankle[2] < confidence_threshold:
            score += (confidence_threshold - min(hip[2], knee[2], ankle[2])) * reference * 2.0
            reasons.append(f"{side}_low_confidence")

    if _distance_score(points.get("left_ankle"), points.get("right_ankle")) < reference * 0.08:
        score += reference * 0.8
        reasons.append("feet_overlap")

    return score, valid, reasons


def _build_lower_body_frame_log(
    *,
    frame_index: int,
    raw_points: Dict[str, tuple[int, int, float]],
    corrected_points: Dict[str, tuple[int, int, float]],
    selected_assignment: str,
    same_score: float,
    swapped_score: float,
    swap_corrected: bool,
    rejected: bool,
    used_previous: bool,
    source_reason: str,
    correction_reason: str,
) -> Dict[str, Any]:
    return {
        "frame_number": frame_index,
        "selected_assignment": selected_assignment,
        "same_assignment_score": round(float(same_score), 3) if np.isfinite(same_score) else None,
        "swapped_assignment_score": round(float(swapped_score), 3) if np.isfinite(swapped_score) else None,
        "lower_body_swap_detected": bool(np.isfinite(same_score) and np.isfinite(swapped_score) and swapped_score + 1e-6 < same_score),
        "lower_body_swap_corrected": bool(swap_corrected),
        "lower_body_rejected": bool(rejected),
        "lower_body_used_previous": bool(used_previous),
        "source_reason": source_reason,
        "correction_reason": correction_reason,
        "raw_left_hip_x": _landmark_payload(raw_points.get("left_hip")).get("x"),
        "raw_left_hip_y": _landmark_payload(raw_points.get("left_hip")).get("y"),
        "raw_left_knee_x": _landmark_payload(raw_points.get("left_knee")).get("x"),
        "raw_left_knee_y": _landmark_payload(raw_points.get("left_knee")).get("y"),
        "raw_left_ankle_x": _landmark_payload(raw_points.get("left_ankle")).get("x"),
        "raw_left_ankle_y": _landmark_payload(raw_points.get("left_ankle")).get("y"),
        "raw_right_hip_x": _landmark_payload(raw_points.get("right_hip")).get("x"),
        "raw_right_hip_y": _landmark_payload(raw_points.get("right_hip")).get("y"),
        "raw_right_knee_x": _landmark_payload(raw_points.get("right_knee")).get("x"),
        "raw_right_knee_y": _landmark_payload(raw_points.get("right_knee")).get("y"),
        "raw_right_ankle_x": _landmark_payload(raw_points.get("right_ankle")).get("x"),
        "raw_right_ankle_y": _landmark_payload(raw_points.get("right_ankle")).get("y"),
        "corrected_left_hip_x": _landmark_payload(corrected_points.get("left_hip")).get("x"),
        "corrected_left_hip_y": _landmark_payload(corrected_points.get("left_hip")).get("y"),
        "corrected_left_knee_x": _landmark_payload(corrected_points.get("left_knee")).get("x"),
        "corrected_left_knee_y": _landmark_payload(corrected_points.get("left_knee")).get("y"),
        "corrected_left_ankle_x": _landmark_payload(corrected_points.get("left_ankle")).get("x"),
        "corrected_left_ankle_y": _landmark_payload(corrected_points.get("left_ankle")).get("y"),
        "corrected_right_hip_x": _landmark_payload(corrected_points.get("right_hip")).get("x"),
        "corrected_right_hip_y": _landmark_payload(corrected_points.get("right_hip")).get("y"),
        "corrected_right_knee_x": _landmark_payload(corrected_points.get("right_knee")).get("x"),
        "corrected_right_knee_y": _landmark_payload(corrected_points.get("right_knee")).get("y"),
        "corrected_right_ankle_x": _landmark_payload(corrected_points.get("right_ankle")).get("x"),
        "corrected_right_ankle_y": _landmark_payload(corrected_points.get("right_ankle")).get("y"),
    }


def _draw_lower_body_debug_overlay(
    frame: np.ndarray,
    points: Dict[str, tuple[int, int, float]],
    debug_record: Dict[str, Any],
) -> np.ndarray:
    debug_frame = frame.copy()
    colors = {
        "left": (90, 220, 120),
        "right": (255, 170, 70),
    }
    label_color = (255, 255, 255)
    for side in ("left", "right"):
        hip = points.get(f"{side}_hip")
        knee = points.get(f"{side}_knee")
        ankle = points.get(f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            continue
        color = colors[side]
        cv2.line(debug_frame, hip[:2], knee[:2], color, 4, cv2.LINE_AA)
        cv2.line(debug_frame, knee[:2], ankle[:2], color, 4, cv2.LINE_AA)
        for label, point in ((f"{side[0].upper()}_HIP", hip), (f"{side[0].upper()}_KNEE", knee), (f"{side[0].upper()}_FOOT", ankle)):
            cv2.circle(debug_frame, point[:2], 7, color, -1, cv2.LINE_AA)
            cv2.circle(debug_frame, point[:2], 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                debug_frame,
                label,
                (point[0] + 8, max(18, point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                label_color,
                1,
                cv2.LINE_AA,
            )

    banner = []
    if debug_record.get("lower_body_swap_corrected"):
        banner.append("swap corrected")
    if debug_record.get("lower_body_rejected"):
        banner.append("rejected")
    if debug_record.get("lower_body_used_previous"):
        banner.append("used previous")
    if banner:
        cv2.rectangle(debug_frame, (14, 14), (340, 54), (12, 18, 28), -1, cv2.LINE_AA)
        cv2.rectangle(debug_frame, (14, 14), (340, 54), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            debug_frame,
            " | ".join(banner),
            (24, 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return debug_frame


def _hand_reference_length(points: Dict[str, tuple[int, int, float]], width: int, height: int) -> float:
    reference = _limb_distance(points.get("left_shoulder"), points.get("right_shoulder"))
    if reference is None:
        reference = _limb_distance(points.get("neck"), points.get("mid_hip"))
    if reference is None:
        reference = float(min(width, height)) * 0.14
    return max(24.0, float(reference))


def _score_hand_candidate(
    points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    person_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[float, bool, list[str]]:
    required = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
    missing = [name for name in required if points.get(name) is None]
    if missing:
        return float("inf"), False, [f"missing:{','.join(sorted(missing))}"]

    reference = _hand_reference_length(points, width, height)
    score = 0.0
    reasons: list[str] = []
    valid = True
    side_margin = max(6.0, reference * 0.12)
    teleport_limit = max(55.0, float(min(width, height)) * 0.18)

    for side in ("left", "right"):
        opposite_side = "right" if side == "left" else "left"
        shoulder = points.get(f"{side}_shoulder")
        elbow = points.get(f"{side}_elbow")
        wrist = points.get(f"{side}_wrist")
        opposite_elbow = points.get(f"{opposite_side}_elbow")
        previous_shoulder = previous_points.get(f"{side}_shoulder")
        previous_elbow = previous_points.get(f"{side}_elbow")
        previous_wrist = previous_points.get(f"{side}_wrist")

        if shoulder is None or elbow is None or wrist is None:
            return float("inf"), False, [f"missing_side:{side}"]

        upper_arm = _distance_score(shoulder, elbow)
        forearm = _distance_score(elbow, wrist)
        arm_total = upper_arm + forearm

        if arm_total < reference * 0.35 or arm_total > reference * 2.6:
            valid = False
            score += reference * 3.0
            reasons.append(f"{side}_arm_length_out_of_range")

        if _distance_score(wrist, elbow) > reference * 1.9:
            valid = False
            score += reference * 2.0
            reasons.append(f"{side}_wrist_far_from_elbow")

        wrist_jump = previous_wrist is not None and _distance_score(wrist, previous_wrist) > teleport_limit
        arm_geometry_reasonable = reference * 0.45 <= arm_total <= reference * 2.1 and _distance_score(wrist, elbow) <= reference * 1.5

        if _distance_score(wrist, opposite_elbow) + side_margin < _distance_score(wrist, elbow):
            score += reference * 0.9
            reasons.append(f"{side}_wrist_closer_to_opposite_elbow")

        if not _point_inside_bbox(wrist, person_bbox, margin_px=max(12.0, reference * 0.10)):
            valid = False
            score += reference * 3.2
            reasons.append(f"{side}_wrist_outside_person_bbox")

        if not _point_inside_bbox(elbow, person_bbox, margin_px=max(20.0, reference * 0.12)):
            score += reference * 1.1
            reasons.append(f"{side}_elbow_outside_person_bbox")

        if wrist_jump:
            score += _distance_score(wrist, previous_wrist) * 1.6
            reasons.append(f"{side}_wrist_teleport")
        if previous_elbow is not None and _distance_score(elbow, previous_elbow) > teleport_limit:
            score += _distance_score(elbow, previous_elbow) * 0.8
            reasons.append(f"{side}_elbow_jump")
        if previous_shoulder is not None and _distance_score(shoulder, previous_shoulder) > teleport_limit:
            score += _distance_score(shoulder, previous_shoulder) * 0.3
            reasons.append(f"{side}_shoulder_jump")

        if wrist[2] < confidence_threshold:
            score += (confidence_threshold - wrist[2]) * reference * 1.5
            reasons.append(f"{side}_wrist_low_confidence")

    return score, valid, reasons


def _build_hand_frame_log(
    *,
    frame_index: int,
    raw_points: Dict[str, tuple[int, int, float]],
    smoothed_points: Dict[str, tuple[int, int, float]],
    final_points: Dict[str, tuple[int, int, float]],
    selected_assignment: str,
    same_score: float,
    swapped_score: float,
    swap_corrected: bool,
    rejected: bool,
    used_previous: bool,
    stale_corrected: bool,
    stale_flags: Dict[str, Any],
    source_reason: str,
    correction_reason: str,
) -> Dict[str, Any]:
    return {
        "frame_number": frame_index,
        "selected_assignment": selected_assignment,
        "same_assignment_score": round(float(same_score), 3) if np.isfinite(same_score) else None,
        "swapped_assignment_score": round(float(swapped_score), 3) if np.isfinite(swapped_score) else None,
        "hand_swap_detected": bool(np.isfinite(same_score) and np.isfinite(swapped_score) and swapped_score + 1e-6 < same_score),
        "hand_swap_corrected": bool(swap_corrected),
        "hand_rejected": bool(rejected),
        "hand_used_previous": bool(used_previous),
        "stale_hand_tracking": bool(stale_corrected or stale_flags.get("stale_hand_tracking", False)),
        "left_stale_hand_tracking": bool(stale_flags.get("left_stale_hand_tracking", False)),
        "right_stale_hand_tracking": bool(stale_flags.get("right_stale_hand_tracking", False)),
        "stale_background_detected": bool(stale_flags.get("stale_background_detected", False)),
        "left_stale_background_detected": bool(stale_flags.get("left_stale_background_detected", False)),
        "right_stale_background_detected": bool(stale_flags.get("right_stale_background_detected", False)),
        "left_stale_hand_reason": stale_flags.get("left_stale_hand_reason"),
        "right_stale_hand_reason": stale_flags.get("right_stale_hand_reason"),
        "left_wrist_accepted": bool(final_points.get("left_wrist") is not None),
        "right_wrist_accepted": bool(final_points.get("right_wrist") is not None),
        "left_wrist_rejection_reason": stale_flags.get("left_stale_hand_reason") or ("rejected" if raw_points.get("left_wrist") is None and final_points.get("left_wrist") is None else None),
        "right_wrist_rejection_reason": stale_flags.get("right_stale_hand_reason") or ("rejected" if raw_points.get("right_wrist") is None and final_points.get("right_wrist") is None else None),
        "interpolation_used": bool(used_previous),
        "source_reason": source_reason,
        "correction_reason": correction_reason,
        **_log_point_triplet("raw_left_wrist", raw_points.get("left_wrist")),
        **_log_point_triplet("raw_right_wrist", raw_points.get("right_wrist")),
        **_log_point_triplet("raw_left_elbow", raw_points.get("left_elbow")),
        **_log_point_triplet("raw_right_elbow", raw_points.get("right_elbow")),
        **_log_point_triplet("smoothed_left_wrist", smoothed_points.get("left_wrist")),
        **_log_point_triplet("smoothed_right_wrist", smoothed_points.get("right_wrist")),
        **_log_point_triplet("smoothed_left_elbow", smoothed_points.get("left_elbow")),
        **_log_point_triplet("smoothed_right_elbow", smoothed_points.get("right_elbow")),
        **_log_point_triplet("final_left_wrist", final_points.get("left_wrist")),
        **_log_point_triplet("final_right_wrist", final_points.get("right_wrist")),
        **_log_point_triplet("final_left_elbow", final_points.get("left_elbow")),
        **_log_point_triplet("final_right_elbow", final_points.get("right_elbow")),
    }


def _draw_hand_debug_overlay(
    frame: np.ndarray,
    raw_points: Dict[str, tuple[int, int, float]],
    smoothed_points: Dict[str, tuple[int, int, float]],
    final_points: Dict[str, tuple[int, int, float]],
    debug_record: Dict[str, Any],
) -> np.ndarray:
    debug_frame = frame.copy()
    stage_specs = [
        (raw_points, (0, 0, 255), 1),
        (smoothed_points, (255, 0, 0), 2),
        (final_points, (255, 255, 255), 3),
    ]

    for stage_points, color, thickness in stage_specs:
        for side in ("left", "right"):
            shoulder = stage_points.get(f"{side}_shoulder")
            elbow = stage_points.get(f"{side}_elbow")
            wrist = stage_points.get(f"{side}_wrist")
            if shoulder is None or elbow is None or wrist is None:
                continue
            cv2.line(debug_frame, shoulder[:2], elbow[:2], color, thickness, cv2.LINE_AA)
            cv2.line(debug_frame, elbow[:2], wrist[:2], color, thickness, cv2.LINE_AA)
            for label, point in ((f"{side[0].upper()}_SHOULDER", shoulder), (f"{side[0].upper()}_ELBOW", elbow), (f"{side[0].upper()}_WRIST", wrist)):
                cv2.circle(debug_frame, point[:2], 7 if label.endswith("WRIST") else 5, color, -1, cv2.LINE_AA)
                cv2.circle(debug_frame, point[:2], 9 if label.endswith("WRIST") else 7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(debug_frame, label, (point[0] + 8, max(18, point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for side in ("left", "right"):
        raw_wrist = raw_points.get(f"{side}_wrist")
        smoothed_wrist = smoothed_points.get(f"{side}_wrist")
        final_wrist = final_points.get(f"{side}_wrist")
        lines = [
            f"{side.upper()} RAW  {('missing' if raw_wrist is None else f'{raw_wrist[0]},{raw_wrist[1]} conf {raw_wrist[2]:.2f}')}",
            f"{side.upper()} SMOOTH {('missing' if smoothed_wrist is None else f'{smoothed_wrist[0]},{smoothed_wrist[1]} conf {smoothed_wrist[2]:.2f}')}",
            f"{side.upper()} FINAL {('missing' if final_wrist is None else f'{final_wrist[0]},{final_wrist[1]} conf {final_wrist[2]:.2f}')}",
        ]
        base_y = 74 if side == "left" else 134
        for offset, text in enumerate(lines):
            cv2.putText(debug_frame, text, (14, base_y + (offset * 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    banner = []
    if debug_record.get("hand_swap_corrected"):
        banner.append("swap corrected")
    if debug_record.get("hand_rejected"):
        banner.append("rejected")
    if debug_record.get("hand_used_previous"):
        banner.append("used previous")
    if debug_record.get("stale_hand_tracking"):
        banner.append("stale corrected")
    if banner:
        cv2.rectangle(debug_frame, (14, 14), (470, 58), (12, 18, 28), -1, cv2.LINE_AA)
        cv2.rectangle(debug_frame, (14, 14), (470, 58), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(debug_frame, " | ".join(banner), (24, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return debug_frame


def _resolve_hand_assignment(
    frame_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    person_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[Dict[str, tuple[int, int, float]], Dict[str, Any]]:
    resolved = dict(frame_points)
    hand_swap_corrections = 0
    hand_rejections = 0
    hand_interpolations = 0
    hand_recoveries = 0

    same_candidate = dict(resolved)
    swapped_candidate = dict(resolved)
    swapped_candidate["left_wrist"], swapped_candidate["right_wrist"] = swapped_candidate.get("right_wrist"), swapped_candidate.get("left_wrist")

    same_score, same_valid, same_reasons = _score_hand_candidate(
        same_candidate,
        previous_points,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
        person_bbox=person_bbox,
    )
    swapped_score, swapped_valid, swapped_reasons = _score_hand_candidate(
        swapped_candidate,
        previous_points,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
        person_bbox=person_bbox,
    )

    selected_assignment = "A"
    selected_candidate = same_candidate
    selected_reasons = same_reasons
    swap_corrected = False
    used_previous = False
    rejected = False
    correction_reason = "same_assignment_selected"
    hand_reference = _hand_reference_length(selected_candidate, width, height)
    left_used_previous = False
    right_used_previous = False
    left_rejected = False
    right_rejected = False

    if swapped_valid and (not same_valid or swapped_score + 1e-6 < same_score):
        selected_assignment = "B"
        selected_candidate = swapped_candidate
        selected_reasons = swapped_reasons
        swap_corrected = True
        hand_swap_corrections += 1
        correction_reason = "swapped_assignment_better"

    for side in ("left", "right"):
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        shoulder_name = f"{side}_shoulder"
        wrist_point = selected_candidate.get(wrist_name)
        elbow_point = selected_candidate.get(elbow_name)
        shoulder_point = selected_candidate.get(shoulder_name)
        previous_wrist = previous_points.get(wrist_name)
        wrist_inside_person = _point_inside_bbox(wrist_point, person_bbox, margin_px=max(10.0, width * 0.02))
        estimated_wrist = _estimate_wrist_from_arm(
            wrist_name,
            selected_candidate,
            previous_points,
            width=width,
            height=height,
        )

        if wrist_point is None or wrist_point[2] < confidence_threshold or elbow_point is None or shoulder_point is None or not wrist_inside_person:
            if estimated_wrist is not None and _wrist_arm_geometry_ok(estimated_wrist, elbow_point, shoulder_point, reference=hand_reference) and _point_inside_bbox(estimated_wrist, person_bbox, margin_px=max(10.0, width * 0.02)):
                selected_candidate[wrist_name] = estimated_wrist
                used_previous = True
                hand_interpolations += 1
                hand_recoveries += 1
                selected_reasons.append(f"{side}_wrist_arm_recovered")
                if side == "left":
                    left_used_previous = True
                else:
                    right_used_previous = True
            elif previous_wrist is not None and previous_wrist[2] >= max(0.2, confidence_threshold * 0.75):
                if _point_inside_bbox(previous_wrist, person_bbox, margin_px=max(10.0, width * 0.02)) and _wrist_arm_geometry_ok(previous_wrist, elbow_point, shoulder_point, reference=hand_reference):
                    selected_candidate[wrist_name] = previous_wrist
                    used_previous = True
                    hand_interpolations += 1
                    selected_reasons.append(f"{side}_wrist_previous_frame")
                    if side == "left":
                        left_used_previous = True
                    else:
                        right_used_previous = True
                else:
                    selected_candidate.pop(wrist_name, None)
                    rejected = True
                    hand_rejections += 1
                    selected_reasons.append(f"{side}_wrist_outside_person_bbox")
            else:
                selected_candidate.pop(wrist_name, None)
                rejected = True
                hand_rejections += 1
                if side == "left":
                    left_rejected = True
                else:
                    right_rejected = True
        else:
            jump_threshold = max(50.0, min(width, height) * 0.16)
            wrist_jump = previous_wrist is not None and detect_unrealistic_jump(previous_wrist, wrist_point, threshold_px=jump_threshold)
            arm_total = _distance_score(shoulder_point, elbow_point) + _distance_score(elbow_point, wrist_point)
            arm_geometry_reasonable = hand_reference * 0.45 <= arm_total <= hand_reference * 2.1 and _distance_score(wrist_point, elbow_point) <= hand_reference * 1.5 and wrist_inside_person

            if wrist_jump and not arm_geometry_reasonable:
                if estimated_wrist is not None and _wrist_arm_geometry_ok(estimated_wrist, elbow_point, shoulder_point, reference=hand_reference) and _point_inside_bbox(estimated_wrist, person_bbox, margin_px=max(10.0, width * 0.02)):
                    selected_candidate[wrist_name] = estimated_wrist
                    used_previous = True
                    hand_interpolations += 1
                    hand_recoveries += 1
                    selected_reasons.append(f"{side}_wrist_arm_recovered")
                    if side == "left":
                        left_used_previous = True
                    else:
                        right_used_previous = True
                elif previous_wrist is not None and previous_wrist[2] >= max(0.2, confidence_threshold * 0.75) and _point_inside_bbox(previous_wrist, person_bbox, margin_px=max(10.0, width * 0.02)):
                    selected_candidate[wrist_name] = previous_wrist
                    used_previous = True
                    hand_interpolations += 1
                    selected_reasons.append(f"{side}_wrist_previous_frame")
                    if side == "left":
                        left_used_previous = True
                    else:
                        right_used_previous = True
                else:
                    selected_candidate.pop(wrist_name, None)
                    rejected = True
                    hand_rejections += 1
                    selected_reasons.append(f"{side}_wrist_background_stuck")
                    if side == "left":
                        left_rejected = True
                    else:
                        right_rejected = True
            elif wrist_jump:
                selected_reasons.append(f"{side}_wrist_jump_kept")
                if wrist_point[2] < confidence_threshold:
                    selected_candidate.pop(wrist_name, None)
                    rejected = True
                    hand_rejections += 1
                    if side == "left":
                        left_rejected = True
                    else:
                        right_rejected = True
            elif previous_wrist is not None and wrist_point[2] < confidence_threshold * 0.85 and _point_inside_bbox(previous_wrist, person_bbox, margin_px=max(10.0, width * 0.02)):
                selected_candidate[wrist_name] = previous_wrist
                used_previous = True
                hand_interpolations += 1
                selected_reasons.append(f"{side}_wrist_previous_frame")
                if side == "left":
                    left_used_previous = True
                else:
                    right_used_previous = True

    if not selected_reasons:
        selected_reasons = ["hand_ok"]

    resolved.update({name: point for name, point in selected_candidate.items() if name in HAND_LANDMARK_NAMES and point is not None})
    debug_record = {
        "selected_assignment": selected_assignment,
        "same_assignment_score": same_score,
        "swapped_assignment_score": swapped_score,
        "same_assignment_valid": same_valid,
        "swapped_assignment_valid": swapped_valid,
        "hand_swap_corrected": swap_corrected,
        "hand_rejected": rejected,
        "hand_used_previous": used_previous,
        "left_wrist_used_previous": left_used_previous,
        "right_wrist_used_previous": right_used_previous,
        "left_wrist_rejected": left_rejected,
        "right_wrist_rejected": right_rejected,
        "hand_reason": ";".join(dict.fromkeys(selected_reasons)),
        "hand_correction_reason": correction_reason,
        "hand_swap_detected": bool(swapped_valid and (not same_valid or swapped_score + 1e-6 < same_score)),
    }
    return resolved, {
        "hand_swap_corrections": hand_swap_corrections,
        "hand_rejections": hand_rejections,
        "hand_interpolations": hand_interpolations,
        "hand_recoveries": hand_recoveries,
        "debug_record": debug_record,
    }


def _resolve_lower_body_assignment(
    frame_points: Dict[str, tuple[int, int, float]],
    previous_points: Dict[str, tuple[int, int, float]],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
) -> tuple[Dict[str, tuple[int, int, float]], Dict[str, Any]]:
    resolved = dict(frame_points)
    knee_swap_corrections = 0
    foot_swap_corrections = 0
    lower_body_rejections = 0
    lower_body_interpolations = 0
    same_candidate = _lower_body_assignment_frame(resolved, "A")
    swapped_candidate = _lower_body_assignment_frame(resolved, "B")

    same_score, same_valid, same_reasons = _score_lower_body_candidate(
        same_candidate,
        previous_points,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
    )
    swapped_score, swapped_valid, swapped_reasons = _score_lower_body_candidate(
        swapped_candidate,
        previous_points,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
    )

    selected_assignment = "A"
    selected_candidate = same_candidate
    selected_reasons = same_reasons
    swap_corrected = False
    used_previous = False
    rejected = False
    correction_reason = "same_assignment_selected"

    if swapped_valid and (not same_valid or swapped_score + 1e-6 < same_score):
        selected_assignment = "B"
        selected_candidate = swapped_candidate
        selected_reasons = swapped_reasons
        swap_corrected = True
        knee_swap_corrections += 1
        foot_swap_corrections += 1
        correction_reason = "swapped_assignment_better"
    elif not same_valid and swapped_valid:
        selected_assignment = "B"
        selected_candidate = swapped_candidate
        selected_reasons = swapped_reasons
        swap_corrected = True
        knee_swap_corrections += 1
        foot_swap_corrections += 1
        correction_reason = "same_assignment_invalid_swapped_valid"

    if not selected_reasons:
        selected_reasons = ["lower_body_ok"]

    critical_failure = (
        not same_valid
        and not swapped_valid
        or selected_candidate.get("left_knee") is None
        or selected_candidate.get("right_knee") is None
        or selected_candidate.get("left_ankle") is None
        or selected_candidate.get("right_ankle") is None
    )
    if critical_failure:
        lower_body_rejections += 1
        rejected = True
    resolved.update({name: point for name, point in selected_candidate.items() if name in LOWER_BODY_LANDMARK_NAMES and point is not None})

    for side in ("left", "right"):
        knee_name = f"{side}_knee"
        ankle_name = f"{side}_ankle"
        knee_point = resolved.get(knee_name)
        ankle_point = resolved.get(ankle_name)
        previous_knee = previous_points.get(knee_name)
        previous_ankle = previous_points.get(ankle_name)

        if knee_point is None or knee_point[2] < confidence_threshold or (previous_knee is not None and detect_unrealistic_jump(previous_knee, knee_point, threshold_px=max(60.0, min(width, height) * 0.18))):
            if previous_knee is not None and previous_knee[2] >= max(0.2, confidence_threshold * 0.75):
                resolved[knee_name] = previous_knee
                lower_body_interpolations += 1
                used_previous = True
                selected_reasons.append(f"{side}_knee_previous_frame")
            else:
                resolved.pop(knee_name, None)
                lower_body_rejections += 1
                rejected = True

        if ankle_point is None or ankle_point[2] < confidence_threshold or (previous_ankle is not None and detect_unrealistic_jump(previous_ankle, ankle_point, threshold_px=max(60.0, min(width, height) * 0.20))):
            if previous_ankle is not None and previous_ankle[2] >= max(0.2, confidence_threshold * 0.75):
                resolved[ankle_name] = previous_ankle
                lower_body_interpolations += 1
                used_previous = True
                selected_reasons.append(f"{side}_foot_previous_frame")
            else:
                resolved.pop(ankle_name, None)
                lower_body_rejections += 1
                rejected = True

        if knee_name in resolved and ankle_name in resolved:
            if resolved[knee_name][1] >= resolved[ankle_name][1] - max(10.0, min(width, height) * 0.03):
                if previous_knee is not None and previous_knee[2] >= max(0.2, confidence_threshold * 0.75):
                    resolved[knee_name] = previous_knee
                    lower_body_interpolations += 1
                    used_previous = True
                    selected_reasons.append(f"{side}_knee_finish_hold")
                if previous_ankle is not None and previous_ankle[2] >= max(0.2, confidence_threshold * 0.75):
                    resolved[ankle_name] = previous_ankle
                    lower_body_interpolations += 1
                    used_previous = True
                    selected_reasons.append(f"{side}_foot_finish_hold")

    debug_record = {
        "selected_assignment": selected_assignment,
        "same_assignment_score": same_score,
        "swapped_assignment_score": swapped_score,
        "same_assignment_valid": same_valid,
        "swapped_assignment_valid": swapped_valid,
        "lower_body_swap_corrected": swap_corrected,
        "lower_body_rejected": rejected,
        "lower_body_used_previous": used_previous,
        "lower_body_reason": ";".join(dict.fromkeys(selected_reasons)),
        "lower_body_correction_reason": correction_reason,
        "lower_body_swap_detected": bool(swapped_valid and (not same_valid or swapped_score + 1e-6 < same_score)),
    }

    return resolved, {
        "knee_swap_corrections": knee_swap_corrections,
        "foot_swap_corrections": foot_swap_corrections,
        "lower_body_rejections": lower_body_rejections,
        "lower_body_interpolations": lower_body_interpolations,
        "debug_record": debug_record,
    }


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
    alpha_map: Dict[str, float] | None = None,
) -> Dict[str, tuple[int, int, float]]:
    if not previous:
        return current
    smoothed: Dict[str, tuple[int, int, float]] = {}
    for name, point in current.items():
        prev = previous.get(name)
        if prev is None:
            smoothed[name] = point
            continue
        point_alpha = alpha_map.get(name, alpha) if alpha_map else alpha
        x = int(round(prev[0] * point_alpha + point[0] * (1.0 - point_alpha)))
        y = int(round(prev[1] * point_alpha + point[1] * (1.0 - point_alpha)))
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
        if landmark_name in HAND_LANDMARK_NAMES:
            continue
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
) -> tuple[Dict[int, Dict[str, tuple[int, int, float]]], Dict[str, int], Dict[str, int], list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not pose_frames:
        return {}, {"frames_interpolated": 0, "frames_rejected": 0, "frames_missing": 0}, {"ankle_corrections": 0, "swap_corrections": 0, "knee_swap_corrections": 0, "foot_swap_corrections": 0, "lower_body_rejections": 0, "lower_body_interpolations": 0, "hand_swap_corrections": 0, "hand_rejections": 0, "hand_interpolations": 0, "hand_recoveries": 0, "stale_hand_corrections": 0, "frames_with_missing_wrist_detection": 0}, [], []

    frame_points: Dict[int, Dict[str, tuple[int, int, float]]] = {}
    frame_reasons: Dict[int, str] = {}
    for pose_frame in pose_frames:
        frame_index = int(pose_frame.get("frame_index", 0) or 0)
        frame_points[frame_index] = _extract_point_map(pose_frame.get("landmarks", {}), width, height)

    dense_series: Dict[int, Dict[str, tuple[int, int, float]]] = {}
    metrics = {"frames_interpolated": 0, "frames_rejected": 0, "frames_missing": 0}
    corrections = {"ankle_corrections": 0, "swap_corrections": 0, "knee_swap_corrections": 0, "foot_swap_corrections": 0, "lower_body_rejections": 0, "lower_body_interpolations": 0, "hand_swap_corrections": 0, "hand_rejections": 0, "hand_interpolations": 0, "hand_recoveries": 0, "stale_hand_corrections": 0, "frames_with_missing_wrist_detection": 0}
    debug_rows: list[Dict[str, Any]] = []
    hand_debug_rows: list[Dict[str, Any]] = []
    previous_valid: Dict[str, tuple[int, int, float]] = {}
    previous_valid_index: int | None = None
    hand_gap_active = {"left": False, "right": False}
    hand_stale_state = {"left": 0, "right": 0}
    left_hand_tracked_frames = 0
    right_hand_tracked_frames = 0
    left_wrist_confidences: list[float] = []
    right_wrist_confidences: list[float] = []
    left_wrist_missing_frames = 0
    right_wrist_missing_frames = 0

    if tracking_mode == "raw":
        for current_index in range(max(0, video_frame_count)):
            dense_series[current_index] = frame_points.get(current_index, {})
            if current_index not in frame_points:
                metrics["frames_missing"] += 1
        return dense_series, metrics, corrections, debug_rows, hand_debug_rows

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

        lower_body_resolved, lower_body_summary = _resolve_lower_body_assignment(
            valid_points,
            previous_valid,
            width=width,
            height=height,
            confidence_threshold=confidence_threshold,
        )
        corrections["knee_swap_corrections"] += int(lower_body_summary.get("knee_swap_corrections", 0))
        corrections["foot_swap_corrections"] += int(lower_body_summary.get("foot_swap_corrections", 0))
        corrections["lower_body_rejections"] += int(lower_body_summary.get("lower_body_rejections", 0))
        corrections["lower_body_interpolations"] += int(lower_body_summary.get("lower_body_interpolations", 0))
        lower_body_debug = dict(lower_body_summary.get("debug_record", {}))
        person_bbox = _person_bbox_from_points(lower_body_resolved, width, height)

        debug_rows.append(
            _build_lower_body_frame_log(
                frame_index=current_index,
                raw_points=source_points,
                corrected_points=lower_body_resolved,
                selected_assignment=str(lower_body_debug.get("selected_assignment", "A")),
                same_score=float(lower_body_debug.get("same_assignment_score", float("inf"))),
                swapped_score=float(lower_body_debug.get("swapped_assignment_score", float("inf"))),
                swap_corrected=bool(lower_body_debug.get("lower_body_swap_corrected", False)),
                rejected=bool(lower_body_debug.get("lower_body_rejected", False)),
                used_previous=bool(lower_body_debug.get("lower_body_used_previous", False)),
                source_reason=frame_reasons.get(current_index, "source"),
                correction_reason=str(lower_body_debug.get("lower_body_correction_reason", "")),
            )
        )

        hand_resolved, hand_summary = _resolve_hand_assignment(
            lower_body_resolved,
            previous_valid,
            width=width,
            height=height,
            confidence_threshold=confidence_threshold,
            person_bbox=person_bbox,
        )
        corrections["hand_swap_corrections"] += int(hand_summary.get("hand_swap_corrections", 0))
        corrections["hand_rejections"] += int(hand_summary.get("hand_rejections", 0))
        corrections["hand_recoveries"] += int(hand_summary.get("hand_recoveries", 0))
        hand_debug = dict(hand_summary.get("debug_record", {}))

        missing_wrist_frame = False
        if source_points.get("left_wrist") is None:
            left_wrist_missing_frames += 1
            missing_wrist_frame = True
        if source_points.get("right_wrist") is None:
            right_wrist_missing_frames += 1
            missing_wrist_frame = True
        if missing_wrist_frame:
            corrections["frames_with_missing_wrist_detection"] += 1

        for side in ("left", "right"):
            wrist_name = f"{side}_wrist"
            raw_wrist = source_points.get(wrist_name)
            if raw_wrist is not None and raw_wrist[2] >= confidence_threshold and hand_resolved.get(wrist_name) is not None:
                hand_gap_active[side] = False
            if hand_debug.get(f"{side}_wrist_used_previous") and not hand_gap_active[side]:
                corrections["hand_interpolations"] += 1
                hand_gap_active[side] = True
            if hand_debug.get(f"{side}_wrist_rejected") and raw_wrist is None:
                hand_gap_active[side] = False

        knee_points = {name: hand_resolved.get(name) for name in ("left_knee", "right_knee") if hand_resolved.get(name) is not None}
        hip_points = {name: hand_resolved.get(name) for name in ("left_hip", "right_hip") if hand_resolved.get(name) is not None}

        corrected_points, ankle_corrections, swap_corrections = correct_ankle_tracking(
            hand_resolved,
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

        alpha_map = {
            name: _hand_smoothing_alpha(name, corrected_points.get(name), previous_valid.get(name), width=width, height=height, style=style)
            for name in HAND_LANDMARK_NAMES
            if corrected_points.get(name) is not None
        }

        if frame_rejected:
            metrics["frames_rejected"] += 1
            frame_reasons[current_index] = DEBUG_FRAME_REASONS["rejected_landmark"]

        if corrected_points:
            if previous_valid:
                smoothed_points = smooth_landmarks(corrected_points, previous_valid, alpha=0.85 if style == "simple" else 0.95, alpha_map=alpha_map)
            else:
                smoothed_points = dict(corrected_points)
            stale_refreshed_points, stale_corrections, stale_flags = _refresh_stale_hand_tracking(
                smoothed_points,
                source_points,
                previous_valid,
                hand_stale_state,
                person_bbox=person_bbox,
                width=width,
                height=height,
                confidence_threshold=confidence_threshold,
            )
            corrections["stale_hand_corrections"] += stale_corrections
            corrected_points = _stabilize_key_landmarks(
                stale_refreshed_points,
                previous_valid,
                current_index=current_index,
                previous_index=previous_valid_index,
                max_hold_frames=2,
            )
            previous_valid = corrected_points
            previous_valid_index = current_index
        elif previous_valid:
            smoothed_points = {}
            stale_flags = {"stale_hand_tracking": False, "left_stale_hand_tracking": False, "right_stale_hand_tracking": False, "left_stale_hand_reason": None, "right_stale_hand_reason": None}
            corrected_points = _stabilize_key_landmarks(
                {},
                previous_valid,
                current_index=current_index,
                previous_index=previous_valid_index,
                max_hold_frames=2,
            )
        else:
            smoothed_points = dict(corrected_points)
            stale_flags = {"stale_hand_tracking": False, "left_stale_hand_tracking": False, "right_stale_hand_tracking": False, "left_stale_hand_reason": None, "right_stale_hand_reason": None}

        if corrected_points:
            left_final_wrist = corrected_points.get("left_wrist")
            right_final_wrist = corrected_points.get("right_wrist")
            if left_final_wrist is not None and left_final_wrist[2] >= 0.5:
                left_hand_tracked_frames += 1
                left_wrist_confidences.append(float(left_final_wrist[2]))
            if right_final_wrist is not None and right_final_wrist[2] >= 0.5:
                right_hand_tracked_frames += 1
                right_wrist_confidences.append(float(right_final_wrist[2]))

        hand_debug_rows.append(
            _build_hand_frame_log(
                frame_index=current_index,
                raw_points=source_points,
                smoothed_points=smoothed_points,
                final_points=corrected_points,
                selected_assignment=str(hand_debug.get("selected_assignment", "A")),
                same_score=float(hand_debug.get("same_assignment_score", float("inf"))),
                swapped_score=float(hand_debug.get("swapped_assignment_score", float("inf"))),
                swap_corrected=bool(hand_debug.get("hand_swap_corrected", False)),
                rejected=bool(hand_debug.get("hand_rejected", False)),
                used_previous=bool(hand_debug.get("hand_used_previous", False)),
                stale_corrected=bool(stale_flags.get("stale_hand_tracking", False)),
                stale_flags=stale_flags,
                source_reason=frame_reasons.get(current_index, "source"),
                correction_reason=str(hand_debug.get("hand_correction_reason", "")),
            )
        )

        dense_series[current_index] = corrected_points

        if debug_frames_dir and frame_reasons.get(current_index) in {DEBUG_FRAME_REASONS["rejected_landmark"], DEBUG_FRAME_REASONS["ankle_jump"], DEBUG_FRAME_REASONS["pose_missing"], DEBUG_FRAME_REASONS["left_right_swap"], DEBUG_FRAME_REASONS["interpolated"]}:
            debug_frames_dir.mkdir(parents=True, exist_ok=True)

    return dense_series, metrics, corrections, debug_rows, hand_debug_rows


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
    lower_body_debug_frames: list[np.ndarray] = []
    hand_debug_frames: list[np.ndarray] = []
    pose_frame_count = sum(1 for frame in pose_frames if frame.get("landmarks"))
    raw_frame_indices = {int(frame.get("frame_index", 0) or 0) for frame in pose_frames}
    LOGGER.info("POSE LANDMARK MAPPING %s", json.dumps(LANDMARK_INDEX_MAPPING, sort_keys=True))
    dense_series, dense_metrics, correction_metrics, lower_body_debug_rows, hand_debug_rows = _build_dense_pose_series(
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

            lower_body_debug_frame = _draw_lower_body_debug_overlay(
                frame,
                frame_points,
                lower_body_debug_rows[frame_index] if frame_index < len(lower_body_debug_rows) else {},
            )
            lower_body_debug_frames.append(lower_body_debug_frame)

            hand_debug_row = hand_debug_rows[frame_index] if frame_index < len(hand_debug_rows) else {}
            hand_debug_frame = _draw_hand_debug_overlay(
                frame,
                _debug_row_to_points(hand_debug_row, "raw"),
                _debug_row_to_points(hand_debug_row, "smoothed"),
                _debug_row_to_points(hand_debug_row, "final"),
                hand_debug_row,
            )
            hand_debug_frames.append(hand_debug_frame)

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

    raw_pose_csv_path = overlay_root / f"{video_file.stem}_raw_pose_landmarks.csv"
    corrected_pose_csv_path = overlay_root / f"{video_file.stem}_corrected_pose_landmarks.csv"
    lower_body_debug_csv_path = overlay_root / f"{video_file.stem}_lower_body_tracking_debug.csv"
    hand_debug_csv_path = overlay_root / f"{video_file.stem}_hand_tracking_debug.csv"
    experiments_root = ensure_dir(overlay_root.parent / "experiments")
    hand_tracking_experiments_csv_path = experiments_root / "hand_tracking_debug.csv"

    raw_rows: list[Dict[str, Any]] = []
    for pose_frame in pose_frames:
        frame_index = int(pose_frame.get("frame_index", 0) or 0)
        landmarks = pose_frame.get("landmarks", {}) or {}
        for name, landmark in landmarks.items():
            raw_rows.append(
                {
                    "frame_number": frame_index,
                    "keypoint_name": name,
                    "x": landmark.get("x"),
                    "y": landmark.get("y"),
                    "confidence": landmark.get("visibility", landmark.get("confidence")),
                }
            )

    corrected_rows: list[Dict[str, Any]] = []
    for frame_index, frame_points in sorted(dense_series.items()):
        for name, point in frame_points.items():
            corrected_rows.append(
                {
                    "frame_number": frame_index,
                    "keypoint_name": name,
                    "x": point[0],
                    "y": point[1],
                    "confidence": point[2],
                }
            )

    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(raw_pose_csv_path, index=False)
    if corrected_rows:
        pd.DataFrame(corrected_rows).to_csv(corrected_pose_csv_path, index=False)
    if lower_body_debug_rows:
        pd.DataFrame(lower_body_debug_rows).to_csv(lower_body_debug_csv_path, index=False)
    if hand_debug_rows:
        pd.DataFrame(hand_debug_rows).to_csv(hand_debug_csv_path, index=False)
        tracking_debug_rows = [
            {
                "frame_number": row.get("frame_number"),
                "left_wrist_raw_x": row.get("raw_left_wrist_x"),
                "left_wrist_raw_y": row.get("raw_left_wrist_y"),
                "right_wrist_raw_x": row.get("raw_right_wrist_x"),
                "right_wrist_raw_y": row.get("raw_right_wrist_y"),
                "left_wrist_confidence": row.get("raw_left_wrist_confidence"),
                "right_wrist_confidence": row.get("raw_right_wrist_confidence"),
                "left_wrist_accepted": row.get("left_wrist_accepted"),
                "right_wrist_accepted": row.get("right_wrist_accepted"),
                "left_wrist_rejection_reason": row.get("left_wrist_rejection_reason"),
                "right_wrist_rejection_reason": row.get("right_wrist_rejection_reason"),
                "stale_background_detected": row.get("stale_background_detected"),
                "interpolation_used": row.get("interpolation_used"),
            }
            for row in hand_debug_rows
        ]
        pd.DataFrame(tracking_debug_rows).to_csv(hand_tracking_experiments_csv_path, index=False)

    saved_path = save_overlay_video(rendered_frames, overlay_path, fps, frame_size)
    overlay_validation = validate_overlay_video(saved_path, video_file)
    tracking_debug_video_path = None
    tracking_debug_video_url = None
    lower_body_debug_video_path = None
    lower_body_debug_video_url = None
    hand_debug_video_path = None
    hand_debug_video_url = None
    hand_background_debug_video_path = None
    hand_background_debug_video_url = None
    wrist_tracking_debug_video_path = None
    wrist_tracking_debug_video_url = None
    if tracking_debug_frames:
        tracking_debug_path = overlay_root / f"{video_file.stem}_tracking_debug.mp4"
        tracking_debug_video_path = save_overlay_video(tracking_debug_frames, tracking_debug_path, fps, frame_size)
        tracking_debug_video_url = f"/outputs/overlays/{Path(tracking_debug_video_path).name}"
    if lower_body_debug_frames:
        lower_body_debug_path = overlay_root / "lower_body_debug.mp4"
        lower_body_debug_video_path = save_overlay_video(lower_body_debug_frames, lower_body_debug_path, fps, frame_size)
        lower_body_debug_video_url = f"/outputs/overlays/{Path(lower_body_debug_video_path).name}"
    if hand_debug_frames:
        hand_debug_path = overlay_root / "hand_tracking_debug.mp4"
        hand_debug_video_path = save_overlay_video(hand_debug_frames, hand_debug_path, fps, frame_size)
        hand_debug_video_url = f"/outputs/overlays/{Path(hand_debug_video_path).name}"
        wrist_debug_path = overlay_root / "wrist_tracking_debug.mp4"
        wrist_tracking_debug_video_path = save_overlay_video(hand_debug_frames, wrist_debug_path, fps, frame_size)
        wrist_tracking_debug_video_url = f"/outputs/overlays/{Path(wrist_tracking_debug_video_path).name}"
        hand_background_debug_path = overlay_root / "hand_background_debug.mp4"
        hand_background_debug_video_path = save_overlay_video(hand_debug_frames, hand_background_debug_path, fps, frame_size)
        hand_background_debug_video_url = f"/outputs/overlays/{Path(hand_background_debug_video_path).name}"
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
    left_hand_points = [point for frame in dense_series.values() if (point := frame.get("left_wrist")) is not None]
    right_hand_points = [point for frame in dense_series.values() if (point := frame.get("right_wrist")) is not None]
    left_hand_confidence = round(float(np.mean([point[2] for point in left_hand_points])), 3) if left_hand_points else None
    right_hand_confidence = round(float(np.mean([point[2] for point in right_hand_points])), 3) if right_hand_points else None
    left_hand_tracking_rate = round((sum(1 for point in left_hand_points if point[2] >= 0.5) / frames_processed) * 100.0, 2) if frames_processed else None
    right_hand_tracking_rate = round((sum(1 for point in right_hand_points if point[2] >= 0.5) / frames_processed) * 100.0, 2) if frames_processed else None
    frames_with_missing_wrist_detection = sum(1 for row in hand_debug_rows if row.get("raw_left_wrist_x") is None or row.get("raw_right_wrist_x") is None)
    left_wrist_missing_frames = sum(1 for row in hand_debug_rows if row.get("raw_left_wrist_x") is None)
    right_wrist_missing_frames = sum(1 for row in hand_debug_rows if row.get("raw_right_wrist_x") is None)
    quality_metrics = {
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "frames_interpolated": dense_metrics.get("frames_interpolated", 0),
        "frames_rejected": dense_metrics.get("frames_rejected", 0),
        "frames_missing": dense_metrics.get("frames_missing", 0),
        "frames_with_missing_wrist_detection": frames_with_missing_wrist_detection,
        "head_tracked_frames": head_tracked_frames,
        "ankle_corrections": correction_metrics.get("ankle_corrections", 0),
        "swap_corrections": correction_metrics.get("swap_corrections", 0),
        "knee_swap_corrections": correction_metrics.get("knee_swap_corrections", 0),
        "foot_swap_corrections": correction_metrics.get("foot_swap_corrections", 0),
        "lower_body_rejections": correction_metrics.get("lower_body_rejections", 0),
        "lower_body_interpolations": correction_metrics.get("lower_body_interpolations", 0),
        "hand_swap_corrections": correction_metrics.get("hand_swap_corrections", 0),
        "hand_rejections": correction_metrics.get("hand_rejections", 0),
        "hand_interpolations": correction_metrics.get("hand_interpolations", 0),
        "hand_recoveries": correction_metrics.get("hand_recoveries", 0),
        "wrist_recovery_count": correction_metrics.get("hand_recoveries", 0) + correction_metrics.get("stale_hand_corrections", 0),
        "stale_hand_corrections": correction_metrics.get("stale_hand_corrections", 0),
        "stale_background_detected": bool(sum(1 for row in hand_debug_rows if row.get("stale_background_detected"))),
        "debug_frame_count": len(debug_frame_paths),
        "head_tracked_rate": round((head_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "hands_tracked_frames": hands_tracked_frames,
        "feet_tracked_frames": feet_tracked_frames,
        "hand_tracking_rate": round((hands_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "hands_tracked_rate": round((hands_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "feet_tracked_rate": round((feet_tracked_frames / frames_processed) * 100.0, 2) if frames_processed else None,
        "left_hand_tracking_rate": left_hand_tracking_rate,
        "right_hand_tracking_rate": right_hand_tracking_rate,
        "left_hand_confidence": left_hand_confidence,
        "right_hand_confidence": right_hand_confidence,
        "left_wrist_missing_frames": left_wrist_missing_frames,
        "right_wrist_missing_frames": right_wrist_missing_frames,
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
            "hand_tracking_debug_video_path": hand_debug_video_path,
            "hand_tracking_debug_video_url": hand_debug_video_url,
            "hand_tracking_debug_csv": str(hand_debug_csv_path) if hand_debug_csv_path.exists() else None,
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
        "raw_pose_landmarks_csv": str(raw_pose_csv_path) if raw_pose_csv_path.exists() else None,
        "corrected_pose_landmarks_csv": str(corrected_pose_csv_path) if corrected_pose_csv_path.exists() else None,
        "lower_body_tracking_debug_csv": str(lower_body_debug_csv_path) if lower_body_debug_csv_path.exists() else None,
        "hand_tracking_debug_csv": str(hand_debug_csv_path) if hand_debug_csv_path.exists() else None,
        "tracking_debug_video_path": tracking_debug_video_path,
        "tracking_debug_video_url": tracking_debug_video_url,
        "lower_body_debug_video_path": lower_body_debug_video_path,
        "lower_body_debug_video_url": lower_body_debug_video_url,
        "hand_tracking_debug_video_path": hand_debug_video_path,
        "hand_tracking_debug_video_url": hand_debug_video_url,
        "hand_tracking_debug_csv": str(hand_debug_csv_path) if hand_debug_csv_path.exists() else None,
        "hand_background_debug_csv": str(hand_tracking_experiments_csv_path) if hand_tracking_experiments_csv_path.exists() else None,
        "hand_background_debug_video_path": hand_background_debug_video_path,
        "hand_background_debug_video_url": hand_background_debug_video_url,
        "wrist_tracking_debug_video_path": wrist_tracking_debug_video_path,
        "wrist_tracking_debug_video_url": wrist_tracking_debug_video_url,
        "hand_background_debug_csv": str(hand_tracking_experiments_csv_path) if hand_tracking_experiments_csv_path.exists() else None,
        "wrist_tracking_debug_csv": str(hand_tracking_experiments_csv_path) if hand_tracking_experiments_csv_path.exists() else None,
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "detection_rate": detection_rate,
        "average_confidence": average_confidence,
        "pose_frame_count": pose_frame_count,
        "video_fps": fps,
        "frame_size": {"width": width, "height": height},
        "quality_metrics": quality_metrics,
        "debug_frame_paths": debug_frame_paths[:12],
        "landmark_mapping": LANDMARK_INDEX_MAPPING,
    }