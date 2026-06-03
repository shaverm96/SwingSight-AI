from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


LOGGER = logging.getLogger(__name__)


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
]


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_sec: float
    landmarks: Dict[str, Landmark]


class PoseEstimator(Protocol):
    def infer(self, video_path: str) -> List[PoseFrame]:
        """Run pose estimation and return per-frame landmarks."""


class DummyPoseEstimator:
    def infer(self, video_path: str) -> List[PoseFrame]:
        _ = video_path
        return []


class YoloPoseEstimator:
    """YOLOv8-pose stub.

    Replace infer() with real YOLOv8-pose inference.
    """

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if YOLO is None:
            return None

        candidate_paths = []
        if self.model_path:
            candidate_paths.append(self.model_path)
        candidate_paths.append("yolov8n-pose.pt")

        for candidate in candidate_paths:
            try:
                self._model = YOLO(candidate)
                return self._model
            except Exception:
                continue
        return None

    def infer(self, video_path: str) -> List[PoseFrame]:
        model = self._load_model()
        if model is None:
            LOGGER.info("POSE backend unavailable video=%s backend=yolov8_pose loaded=false", video_path)
            return []

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            LOGGER.info("POSE backend unable to open video=%s backend=yolov8_pose", video_path)
            return []

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: List[PoseFrame] = []
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_sec = frame_index / fps if fps > 0 else float(frame_index)
            landmarks: Dict[str, Landmark] = {}
            raw_keypoints = 0
            average_confidence = 0.0

            try:
                results = model.predict(frame, verbose=False)
                result = results[0] if results else None
                keypoints = getattr(result, "keypoints", None)
                if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                    xy = keypoints.xy.cpu().numpy()
                    conf = keypoints.conf.cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
                    if len(xy) > 0:
                        person_xy = xy[0]
                        person_conf = conf[0] if conf is not None and len(conf) > 0 else [1.0] * len(person_xy)
                        landmarks = map_keypoints(person_xy, person_conf)
                        raw_keypoints = len(landmarks)
                        average_confidence = float(sum(lm.visibility for lm in landmarks.values()) / max(1, len(landmarks)))
            except Exception as exc:
                LOGGER.warning("POSE frame inference failed video=%s frame=%s error=%s", video_path, frame_index, exc)
                landmarks = {}

            LOGGER.info(
                "POSE FRAME video=%s backend=yolov8_pose frame=%s detected_landmarks=%s avg_confidence=%.3f",
                video_path,
                frame_index,
                raw_keypoints,
                average_confidence,
            )

            frames.append(PoseFrame(frame_index=frame_index, timestamp_sec=timestamp_sec, landmarks=landmarks))

            frame_index += 1

        capture.release()
        return frames


def build_pose_estimator(config: Dict) -> PoseEstimator:
    settings = config.get("pose_estimation", {})
    backend = settings.get("backend", "dummy")
    model_path = settings.get("model_path", "")
    if backend == "yolov8_pose" or YOLO is not None:
        return YoloPoseEstimator(
            model_path=str(model_path),
            confidence_threshold=float(settings.get("confidence_threshold", 0.4)),
        )

    return DummyPoseEstimator()


def _frames_to_dicts(frames: List[PoseFrame]) -> List[Dict]:
    serialized = []
    for frame in frames:
        serialized.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "landmarks": {
                    name: {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for name, lm in frame.landmarks.items()
                },
            }
        )
    return serialized


def map_keypoints(xy: Iterable[Iterable[float]], conf: Iterable[float]) -> Dict[str, Landmark]:
    """Map YOLOv8-pose keypoints to named landmarks.

    Expects xy as an iterable of (x, y) pairs and conf as confidences.
    """
    landmarks: Dict[str, Landmark] = {}
    for name, coords, score in zip(COCO_KEYPOINTS, xy, conf):
        x, y = coords
        landmarks[name] = Landmark(name=name, x=float(x), y=float(y), z=0.0, visibility=float(score))
    return landmarks


def build_landmark_rows(
    video_id: str,
    club_type: str,
    frame_index: int,
    timestamp_sec: float,
    landmarks: Dict[str, Landmark],
) -> List[Dict[str, float | int | str]]:
    """Create rows for a landmark dataframe schema."""
    rows: List[Dict[str, float | int | str]] = []
    for name, lm in landmarks.items():
        rows.append(
            {
                "video_id": video_id,
                "frame_number": frame_index,
                "timestamp": timestamp_sec,
                "club_type": club_type,
                "keypoint_name": name,
                "x": lm.x,
                "y": lm.y,
                "confidence": lm.visibility,
            }
        )
    return rows


def run_pose_estimation(video_path: str, config: Dict) -> List[Dict]:
    """Extract pose landmarks from the video and return serializable dicts."""
    estimator = build_pose_estimator(config)
    frames = estimator.infer(video_path)
    return _frames_to_dicts(frames)
