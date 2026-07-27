"""Locate a plausible club-head/grip region in a full frame before classification.

The five-way club-type CNN (see ``club_cnn.py``) is trained on catalog-style
reference photos: an isolated club head, filling the frame, plain background.
Feeding it a full "show it to the camera" photo -- background clutter and a
small club -- puts the input far outside that training distribution. Held-out
catalog test accuracy for the checkpoint this module feeds is ~99%, so a live
misclassification is a sign of that domain gap, not the classifier itself:
tightening the crop around the club closes most of the gap.

This module detects a HAND directly with MediaPipe's HandLandmarker task
(``hand_landmarker.task``, bundled with the project) and crops tightly around
it. Unlike an earlier version of this module built on YOLOv8-pose body
keypoints, this does not require a visible torso/shoulders or any particular
stance -- a close-up shot of just a hand holding a club up to the camera is
enough, which matches how this feature is actually used.

If no confident hand is found, the original image is returned unmodified
(``cropped=False``) so callers can still classify it, while treating the
result with extra caution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
except Exception:  # pragma: no cover - optional dependency
    mp = None
    BaseOptions = None
    HandLandmarker = None
    HandLandmarkerOptions = None

DEFAULT_HAND_MODEL_PATH = "hand_landmarker.task"
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.3
DEFAULT_MIN_PRESENCE_CONFIDENCE = 0.3
# Crop side length, as a multiple of the detected hand's bounding-box size.
# Calibrated from a known-good manual crop where the hand's landmark bbox was
# ~50px inside a 288px crop that fully framed the club head (288/50 ~= 5.8).
# Treat this as a starting point -- tune against real captures.
DEFAULT_CROP_SCALE = 5.5
# If two hands are detected, merge them into one crop when their centers are
# closer together than this multiple of the larger hand's size (two-handed
# grip); otherwise the larger/closer-looking hand is used alone.
DEFAULT_MERGE_RATIO = 1.5


@dataclass(frozen=True)
class ClubCropResult:
    """The outcome of trying to localize the club/grip region in a frame."""

    image: Image.Image
    cropped: bool
    box: Optional[tuple[int, int, int, int]]
    reasoning: str


@lru_cache(maxsize=2)
def _load_hand_detector(model_path: str, min_detection_confidence: float, min_presence_confidence: float):
    if mp is None or HandLandmarker is None:
        return None
    try:
        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
        )
        return HandLandmarker.create_from_options(options)
    except Exception:  # pragma: no cover - missing/corrupt model file or runtime issue
        return None


def _hand_bbox(landmarks, width: int, height: int) -> tuple[float, float, float, float]:
    xs = [landmark.x * width for landmark in landmarks]
    ys = [landmark.y * height for landmark in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def locate_club_crop(
    image: Image.Image,
    *,
    model_path: str = DEFAULT_HAND_MODEL_PATH,
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    min_presence_confidence: float = DEFAULT_MIN_PRESENCE_CONFIDENCE,
    crop_scale: float = DEFAULT_CROP_SCALE,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
) -> ClubCropResult:
    """Crop ``image`` tightly around a detected hand -- no body or stance required.

    Falls back to returning the original image, uncropped, if mediapipe is
    unavailable or no hand is found. Any unexpected failure also falls back
    rather than raising -- this crop is an accuracy boost, not a required
    step, and must never block classification.
    """
    rgb = image.convert("RGB")

    detector = _load_hand_detector(model_path, min_detection_confidence, min_presence_confidence)
    if detector is None:
        return ClubCropResult(rgb, False, None, "Hand detector unavailable; classifying the full frame.")

    try:
        frame = np.ascontiguousarray(np.array(rgb))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(mp_image)

        if not result.hand_landmarks:
            return ClubCropResult(rgb, False, None, "No hand detected; classifying the full frame.")

        boxes = [_hand_bbox(landmarks, rgb.width, rgb.height) for landmarks in result.hand_landmarks]
        sizes = [max(box[2] - box[0], box[3] - box[1]) for box in boxes]

        if len(boxes) > 1:
            # There is no elbow/arm context here to tell which hand is
            # "engaged" the way the earlier pose-based version could, so:
            # merge close-together hands (two-handed grip), otherwise fall
            # back to the larger/closer-looking hand.
            (left0, top0, right0, bottom0), (left1, top1, right1, bottom1) = boxes[0], boxes[1]
            center0 = ((left0 + right0) / 2.0, (top0 + bottom0) / 2.0)
            center1 = ((left1 + right1) / 2.0, (top1 + bottom1) / 2.0)
            center_distance = float(np.hypot(center0[0] - center1[0], center0[1] - center1[1]))
            larger_size = max(sizes)

            if center_distance < merge_ratio * larger_size:
                min_x = min(left0, left1)
                min_y = min(top0, top1)
                max_x = max(right0, right1)
                max_y = max(bottom0, bottom1)
                hand_size = max(max_x - min_x, max_y - min_y)
                center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                reasoning = "Cropped around both hands (close together)."
            else:
                index = int(np.argmax(sizes))
                left, top, right, bottom = boxes[index]
                hand_size = sizes[index]
                center = ((left + right) / 2.0, (top + bottom) / 2.0)
                reasoning = "Cropped around the larger/closer of two detected hands."
        else:
            left, top, right, bottom = boxes[0]
            hand_size = sizes[0]
            center = ((left + right) / 2.0, (top + bottom) / 2.0)
            reasoning = "Cropped around the detected hand."

        side = max(48.0, crop_scale * hand_size)
        left_bound = int(max(0, center[0] - side / 2))
        top_bound = int(max(0, center[1] - side / 2))
        right_bound = int(min(rgb.width, center[0] + side / 2))
        bottom_bound = int(min(rgb.height, center[1] + side / 2))

        if right_bound - left_bound < 16 or bottom_bound - top_bound < 16:
            return ClubCropResult(rgb, False, None, "Degenerate crop box; classifying the full frame.")

        crop = rgb.crop((left_bound, top_bound, right_bound, bottom_bound))
        return ClubCropResult(crop, True, (left_bound, top_bound, right_bound, bottom_bound), reasoning)
    except Exception as error:
        # This crop is a nice-to-have accuracy boost, not a required step.
        # Any failure here (unexpected mediapipe output shape, a runtime
        # backend issue, etc.) must never block classification -- fall back
        # to the uncropped frame instead of letting the exception propagate
        # up through detect_club().
        return ClubCropResult(rgb, False, None, f"Hand-based crop failed ({error}); classifying the full frame.")
