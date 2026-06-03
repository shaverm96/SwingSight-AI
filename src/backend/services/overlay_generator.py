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

TRAIL_KEYS = {
    "head": "nose",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "hips": "mid_hip",
}

POINT_COLORS = {
    "head": (76, 232, 255),
    "neck": (0, 220, 170),
    "left_shoulder": (86, 179, 255),
    "right_shoulder": (86, 179, 255),
    "left_elbow": (255, 170, 74),
    "right_elbow": (255, 170, 74),
    "left_wrist": (255, 112, 112),
    "right_wrist": (255, 112, 112),
    "left_hip": (166, 127, 255),
    "right_hip": (166, 127, 255),
    "left_knee": (255, 224, 102),
    "right_knee": (255, 224, 102),
    "left_ankle": (255, 255, 255),
    "right_ankle": (255, 255, 255),
}

TRAIL_COLORS = {
    "head": (76, 232, 255),
    "left_hand": (255, 112, 112),
    "right_hand": (255, 112, 112),
    "hips": (166, 127, 255),
}


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
) -> Dict[str, tuple[int, int, float]]:
    height, width = frame.shape[:2]
    points = _synthesized_points(landmarks, width, height)

    for start_name, end_name in SKELETON_CONNECTIONS:
        start_point = points.get(start_name)
        end_point = points.get(end_name)
        if start_point is None or end_point is None:
            continue
        if start_point[2] < confidence_threshold and start_name != "neck":
            continue
        if end_point[2] < confidence_threshold and end_name != "neck":
            continue
        cv2.line(frame, start_point[:2], end_point[:2], (105, 245, 160), 2, cv2.LINE_AA)

    for name, point in points.items():
        if name not in POINT_COLORS:
            continue
        if point[2] < confidence_threshold and name not in {"neck", "mid_hip"}:
            continue
        radius = 6 if name in {"head", "neck", "mid_hip"} else 4
        cv2.circle(frame, point[:2], radius, POINT_COLORS[name], -1, cv2.LINE_AA)
        cv2.circle(frame, point[:2], radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    return points


def draw_motion_trails(
    frame: np.ndarray,
    trail_history: Dict[str, Deque[tuple[int, int]]],
    *,
    max_length: int = 12,
) -> np.ndarray:
    trail_layer = np.zeros_like(frame)
    for trail_name, history in trail_history.items():
        points = list(history)
        if len(points) < 2:
            continue
        color = TRAIL_COLORS.get(trail_name, (180, 220, 255))
        for index in range(1, len(points)):
            age_ratio = index / float(len(points))
            faded_color = tuple(int(channel * max(0.15, age_ratio)) for channel in color)
            thickness = max(1, int(round(1 + age_ratio * 3)))
            cv2.line(trail_layer, points[index - 1], points[index], faded_color, thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(frame, 1.0, trail_layer, 0.85, 0)
    return blended


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

    pose_index = 0
    active_pose: Optional[Dict[str, Any]] = None
    rendered_frames: list[np.ndarray] = []
    trail_history: Dict[str, Deque[tuple[int, int]]] = {
        key: deque(maxlen=trail_length) for key in TRAIL_KEYS
    }
    frames_processed = 0
    frames_with_pose = 0
    confidence_values: list[float] = []
    pose_frame_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frames_processed += 1
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frames_processed) - 1

        while pose_index < len(pose_frames):
            pose_frame = pose_frames[pose_index]
            pose_frame_index = int(pose_frame.get("frame_index", 0) or 0)
            if pose_frame_index <= frame_index:
                active_pose = pose_frame
                pose_index += 1
            else:
                break

        if active_pose and active_pose.get("landmarks"):
            pose_frame_count += 1
            points = draw_skeleton(frame, active_pose["landmarks"], confidence_threshold=confidence_threshold)

            if points:
                frames_with_pose += 1
                for name, landmark_name in TRAIL_KEYS.items():
                    if landmark_name == "mid_hip":
                        left_hip = points.get("left_hip")
                        right_hip = points.get("right_hip")
                        if left_hip and right_hip:
                            center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
                            trail_history[name].append(center)
                            confidence_values.append(float((left_hip[2] + right_hip[2]) / 2.0))
                        continue

                    point = points.get(landmark_name)
                    if point and point[2] >= confidence_threshold:
                        trail_history[name].append(point[:2])
                        confidence_values.append(float(point[2]))

            frame = draw_motion_trails(frame, trail_history, max_length=trail_length)

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

        rendered_frames.append(frame)

    capture.release()

    saved_path = save_overlay_video(rendered_frames, overlay_path, fps, frame_size)
    overlay_validation = validate_overlay_video(saved_path, video_file)
    LOGGER.info(
        "OVERLAY VIDEO CREATED path=%s size_mb=%s frames=%s",
        saved_path,
        overlay_validation.get("size_mb"),
        overlay_validation.get("frame_count"),
    )
    LOGGER.info("OVERLAY VALIDATION %s", overlay_validation)
    detection_rate = round((frames_with_pose / frames_processed) * 100.0, 2) if frames_processed else None
    average_confidence = round(float(np.mean(confidence_values)), 3) if confidence_values else None

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
            "frames_processed": frames_processed,
            "frames_with_pose": 0,
            "detection_rate": detection_rate,
            "average_confidence": average_confidence,
        }

    LOGGER.info("POSE SUCCESS video=%s pose_frames=%s detections=%s", video_file, pose_frame_count, frames_with_pose)
    return {
        "status": "success",
        "message": "Pose overlay generated successfully.",
        "warnings": [],
        "overlay_video_path": saved_path,
        "overlay_video_url": f"/outputs/overlays/{Path(saved_path).name}" if saved_path else None,
        "overlay_validation": overlay_validation,
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "detection_rate": detection_rate,
        "average_confidence": average_confidence,
        "pose_frame_count": pose_frame_count,
        "video_fps": fps,
        "frame_size": {"width": width, "height": height},
    }