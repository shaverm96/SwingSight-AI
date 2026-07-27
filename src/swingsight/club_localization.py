"""Locate a plausible club-head/grip region in a full frame before classification.

The five-way club-type CNN (see ``club_cnn.py``) is trained on catalog-style
reference photos: an isolated club head, filling the frame, plain background.
Feeding it a full "address position" photo -- golfer, room, background
clutter, and a small club -- puts the input far outside that training
distribution, which is why raw full-frame photos are misclassified far more
often than the cropped catalog photos used for validation.

This module uses the YOLOv8-pose model already bundled with the project
(``yolov8n-pose.pt``) to find the golfer's wrists/grip position, then crops a
region around them sized relative to body scale (shoulder width). This is a
practical stand-in for a dedicated, separately trained club detector -- it
will not be perfect (occluded grips, extreme camera angles, multiple people
in frame), but it removes most of the irrelevant background before
classification and is built entirely from a model already in this repo, so
it needs no new training data to ship.

If no confident person/wrist detection is available, the original image is
returned unmodified (``cropped=False``) so callers can still classify it,
while treating the result with extra caution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

# Standard COCO-17 keypoint order used by Ultralytics YOLOv8-pose, matching
# swingsight.pose_estimation.COCO_KEYPOINTS.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

# Keypoints used to bound the crop. A club can be held with two hands close
# together (address position) or one hand raised to show it to the camera
# (the other arm relaxed at the side) -- using only the wrist midpoint biases
# the crop toward whichever arm happens to be idle. Spanning the whole
# shoulder-to-wrist region for both arms is more robust to either pose.
UPPER_BODY_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)

DEFAULT_POSE_MODEL_PATH = "yolov8n-pose.pt"
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3
DEFAULT_PADDING_FRACTION = 0.45  # padding added to the upper-body box, as a fraction of its own width/height
DEFAULT_MIN_PADDING_SCALE = 0.6  # padding floor, as a multiple of shoulder width, for small/compact boxes


@dataclass(frozen=True)
class ClubCropResult:
    """The outcome of trying to localize the club/grip region in a frame."""

    image: Image.Image
    cropped: bool
    box: Optional[tuple[int, int, int, int]]
    reasoning: str


@lru_cache(maxsize=4)
def _load_pose_model(model_path: str):
    if YOLO is None:
        return None
    try:
        return YOLO(model_path)
    except Exception:  # pragma: no cover - missing/corrupt weights file
        return None


def locate_club_crop(
    image: Image.Image,
    *,
    model_path: str = DEFAULT_POSE_MODEL_PATH,
    min_keypoint_confidence: float = DEFAULT_MIN_KEYPOINT_CONFIDENCE,
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
    min_padding_scale: float = DEFAULT_MIN_PADDING_SCALE,
) -> ClubCropResult:
    """Crop ``image`` around the golfer's arms/hands using YOLOv8-pose keypoints.

    Falls back to returning the original image, uncropped, if ultralytics is
    unavailable or no confident person/arm keypoints are found.
    """
    rgb = image.convert("RGB")

    model = _load_pose_model(model_path)
    if model is None:
        return ClubCropResult(rgb, False, None, "Pose model unavailable; classifying the full frame.")

    frame = np.array(rgb)
    try:
        results = model.predict(frame, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return ClubCropResult(rgb, False, None, "No person detected; classifying the full frame.")

        result = results[0]
        boxes = result.boxes
        person_index = int(boxes.conf.argmax().item()) if boxes is not None and len(boxes) > 0 else 0

        xy = result.keypoints.xy[person_index].cpu().numpy()
        conf = result.keypoints.conf[person_index].cpu().numpy() if result.keypoints.conf is not None else None

        def confident(index: int) -> bool:
            return conf is None or conf[index] >= min_keypoint_confidence

        if confident(LEFT_SHOULDER) and confident(RIGHT_SHOULDER):
            shoulder_width = float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]))
        else:
            shoulder_width = 0.0
        if shoulder_width <= 1e-3:
            # Shoulders not visible (tight crop, turned away, etc.) -- fall back to
            # a fraction of the frame size for the padding floor below.
            shoulder_width = 0.18 * min(rgb.width, rgb.height)

        # Span the whole shoulder-to-wrist region for both arms rather than just
        # the wrist midpoint. A club can be held with two hands close together
        # (address position) or one hand raised to show it to the camera while
        # the other arm hangs relaxed -- spanning both arms means whichever one
        # is actually holding the club stays inside the box either way.
        points = [
            xy[index]
            for index in UPPER_BODY_KEYPOINTS
            if confident(index) and not np.allclose(xy[index], 0)
        ]
        if not points:
            return ClubCropResult(rgb, False, None, "No confident arm keypoints; classifying the full frame.")

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        box_left, box_right = min(xs), max(xs)
        box_top, box_bottom = min(ys), max(ys)
        box_width = max(box_right - box_left, 1.0)
        box_height = max(box_bottom - box_top, 1.0)

        pad_x = max(padding_fraction * box_width, min_padding_scale * shoulder_width)
        pad_y = max(padding_fraction * box_height, min_padding_scale * shoulder_width)

        left = int(max(0, box_left - pad_x))
        top = int(max(0, box_top - pad_y))
        right = int(min(rgb.width, box_right + pad_x))
        bottom = int(min(rgb.height, box_bottom + pad_y))

        if right - left < 16 or bottom - top < 16:
            return ClubCropResult(rgb, False, None, "Degenerate crop box; classifying the full frame.")

        crop = rgb.crop((left, top, right, bottom))
        return ClubCropResult(
            crop,
            True,
            (left, top, right, bottom),
            "Cropped around the detected arm/hand region.",
        )
    except Exception as error:
        # This crop is a nice-to-have accuracy boost, not a required step.
        # Any failure here (unexpected ultralytics/torch output shape, a
        # keypoint schema mismatch across versions, etc.) must never block
        # classification -- fall back to the uncropped frame instead of
        # letting the exception propagate up through detect_club().
        return ClubCropResult(rgb, False, None, f"Pose-based crop failed ({error}); classifying the full frame.")
