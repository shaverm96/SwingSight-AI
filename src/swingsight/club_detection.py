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


class CnnBroadCategoryDetector:
    """Compatibility wrapper around the first stage of club recognition."""

    def __init__(self, config: Dict) -> None:
        self.config = config

    def predict(self, image: Image.Image) -> ClubDetectionResult:
        from swingsight.club_recognition import classify_broad_category

        result = classify_broad_category(image, self.config)
        if result.category == "wood":
            return ClubDetectionResult(category="driver_wood", confidence=result.confidence)
        if result.category == "iron":
            return ClubDetectionResult(category="iron_wedge", confidence=result.confidence)
        return ClubDetectionResult(category="unknown", confidence=0.0)


def build_club_detector(config: Dict) -> ClubDetector:
    """Build the broad CNN compatibility wrapper.

    The actual YOLO club-head crop is performed inside ``club_recognition``;
    this legacy surface exposes only the first iron/wood decision.
    """

    return CnnBroadCategoryDetector(config)


def detect_club_category(image_file, config: Dict) -> Tuple[str, float]:
    """Return a compatibility broad category from the stage-one CNN.

    New callers should use ``recognize_club_from_frame`` for all three stages.
    This wrapper deliberately avoids the retired dummy detector prediction.
    """

    image = Image.open(image_file)
    result = build_club_detector(config).predict(image)
    return result.category, result.confidence
