from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol


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

    def infer(self, video_path: str) -> List[PoseFrame]:
        _ = video_path
        # TODO: run YOLOv8-pose and return PoseFrame objects.
        return []


def build_pose_estimator(config: Dict) -> PoseEstimator:
    settings = config.get("pose_estimation", {})
    backend = settings.get("backend", "dummy")
    if backend == "yolov8_pose":
        return YoloPoseEstimator(
            model_path=settings.get("model_path", ""),
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
