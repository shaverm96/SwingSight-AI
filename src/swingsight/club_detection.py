from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

from PIL import Image

CLUB_CATEGORIES = ["driver_wood", "hybrid", "iron_wedge"]
CLUB_LABELS: Dict[str, str] = {
    "driver_wood": "Driver / Wood",
    "hybrid": "Hybrid",
    "iron_wedge": "Iron / Wedge",
}


@dataclass(frozen=True)
class ClubDetectionResult:
    category: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class ClubDetector(Protocol):
    def predict(self, image: Image.Image) -> ClubDetectionResult:
        """Return a club detection result for a single image."""


class DummyClubDetector:
    """Placeholder detector for the MVP scaffolding."""

    def predict(self, image: Image.Image) -> ClubDetectionResult:
        _ = image
        return ClubDetectionResult(category="iron_wedge", confidence=0.55)


class YoloClubDetector:
    """YOLOv8 detector stub.

    Replace the body of predict() with YOLOv8 inference.
    """

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def predict(self, image: Image.Image) -> ClubDetectionResult:
        _ = image
        # TODO: load model and run inference to return a real prediction.
        return ClubDetectionResult(category="iron_wedge", confidence=0.55)


def build_club_detector(config: Dict) -> ClubDetector:
    """Build the configured club detector."""
    settings = config.get("club_detection", {})
    model_type = settings.get("model_type", "dummy")
    if model_type == "yolov8":
        return YoloClubDetector(
            model_path=settings.get("model_path", ""),
            confidence_threshold=float(settings.get("confidence_threshold", 0.4)),
        )

    return DummyClubDetector()


def detect_club_category(image_file, config: Dict) -> Tuple[str, float]:
    """Return a broad club category and confidence."""
    image = Image.open(image_file)
    detector = build_club_detector(config)
    result = detector.predict(image)
    return result.category, result.confidence
