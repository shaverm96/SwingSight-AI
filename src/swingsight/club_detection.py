from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image

CLUB_CATEGORIES = ["driver_wood", "hybrid", "iron_wedge"]
CLUB_LABELS: Dict[str, str] = {
    "driver_wood": "Driver / Wood",
    "hybrid": "Hybrid",
    "iron_wedge": "Iron / Wedge",
}


def detect_club_category(image_file, config: Dict) -> Tuple[str, float]:
    """Return a broad club category and confidence.

    This is a placeholder for a future model. Integrate YOLOv8 or a
    lightweight classifier here and return the predicted category.
    """
    _ = Image.open(image_file)

    # Placeholder: always return iron/wedge at mid confidence.
    return "iron_wedge", 0.55
