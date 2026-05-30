from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

from PIL import Image


@dataclass(frozen=True)
class ClubHeadDetection:
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    source: str


@dataclass(frozen=True)
class BroadCategoryResult:
    category: str
    confidence: float
    reasoning: Optional[str]
    source: str


@dataclass(frozen=True)
class WoodSizingResult:
    club_type: str
    confidence: float
    reasoning: Optional[str]


@dataclass(frozen=True)
class OcrResult:
    text: Optional[str]
    confidence: float
    source: str


def recognize_club_from_frame(image_path: str, config: Dict) -> Dict:
    image = Image.open(image_path).convert("RGB")

    detection = detect_club_head(image, config)
    crop = crop_region(image, detection.bbox)
    broad = classify_broad_category(crop, config)

    if broad.category == "wood":
        sizing = estimate_wood_size(detection, crop, config)
        predicted = sizing.club_type
        confidence = aggregate_confidence([detection.confidence, broad.confidence, sizing.confidence])
        reasoning = join_reasoning([broad.reasoning, sizing.reasoning])
        ocr_result = None
    else:
        ocr_result = read_club_marking(crop, config)
        predicted = interpret_marking(ocr_result.text)
        confidence = aggregate_confidence([detection.confidence, broad.confidence, ocr_result.confidence])
        reasoning = join_reasoning([
            broad.reasoning,
            "marking detected" if ocr_result.text else "marking not detected",
        ])

    confirm_threshold = float(config.get("club_recognition", {}).get("confirm_threshold", 0.6))
    status = "confirmed" if predicted and confidence >= confirm_threshold else "uncertain"

    return {
        "status": status,
        "detected_category": "Wood" if broad.category == "wood" else "Iron/Wedge",
        "predicted_club": predicted or "Unknown",
        "confidence": round(float(confidence), 3),
        "reasoning": reasoning,
        "bbox": detection.bbox,
        "ocr": None if ocr_result is None else {
            "text": ocr_result.text,
            "confidence": round(float(ocr_result.confidence), 3),
            "source": ocr_result.source,
        },
        "sources": {
            "detector": detection.source,
            "broad_classifier": broad.source,
        },
    }


def detect_club_head(image: Image.Image, config: Dict) -> ClubHeadDetection:
    settings = config.get("club_recognition", {})
    model_path = settings.get("yolo_model_path", "models/club_detector.pt")
    confidence_threshold = float(settings.get("detector_confidence_threshold", 0.4))
    class_name = settings.get("yolo_class_name")

    # TODO: Load YOLOv8 weights and return the best bounding box for the club head.
    # This stub attempts to use ultralytics if installed, otherwise returns the full frame.
    try:
        model = _get_yolo_model(model_path)
        if model is None:
            raise RuntimeError("YOLO model not available")
        results = model.predict(image, conf=confidence_threshold, verbose=False)
        if not results:
            return ClubHeadDetection(bbox=None, confidence=0.0, source="yolov8")
        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
            return ClubHeadDetection(bbox=None, confidence=0.0, source="yolov8")

        best = select_best_box(result, class_name)
        if best is None:
            return ClubHeadDetection(bbox=None, confidence=0.0, source="yolov8")

        x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
        return ClubHeadDetection(bbox=(x1, y1, x2, y2), confidence=float(best.conf), source="yolov8")
    except Exception:
        width, height = image.size
        return ClubHeadDetection(bbox=(0, 0, width, height), confidence=0.25, source="fallback_full_frame")


def crop_region(image: Image.Image, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    if not bbox:
        return image
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image.width))
    y1 = max(0, min(y1, image.height))
    x2 = max(x1 + 1, min(x2, image.width))
    y2 = max(y1 + 1, min(y2, image.height))
    return image.crop((x1, y1, x2, y2))


def classify_broad_category(image: Image.Image, config: Dict) -> BroadCategoryResult:
    # TODO: Replace heuristic with a pretrained CNN classifier.
    # Placeholder heuristic: wider aspect ratio suggests wood-style head.
    settings = config.get("club_recognition", {})
    aspect_ratio = image.width / float(max(1, image.height))
    wood_threshold = float(settings.get("wood_aspect_threshold", 1.15))
    if aspect_ratio >= wood_threshold:
        return BroadCategoryResult(
            category="wood",
            confidence=0.55,
            reasoning="wide head aspect detected",
            source="cnn_stub",
        )
    return BroadCategoryResult(
        category="iron_wedge",
        confidence=0.55,
        reasoning="narrow head region detected",
        source="cnn_stub",
    )


def estimate_wood_size(detection: ClubHeadDetection, image: Image.Image, config: Dict) -> WoodSizingResult:
    settings = config.get("club_recognition", {})
    driver_threshold = float(settings.get("driver_area_threshold", 0.08))
    fairway_threshold = float(settings.get("fairway_area_threshold", 0.05))

    area_ratio = 0.0
    if detection.bbox:
        x1, y1, x2, y2 = detection.bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        area_ratio = area / float(image.width * image.height)

    if area_ratio >= driver_threshold:
        return WoodSizingResult("Driver", 0.6, "large rounded head detected")
    if area_ratio >= fairway_threshold:
        return WoodSizingResult("Fairway Wood", 0.55, "medium head detected")
    return WoodSizingResult("Hybrid", 0.5, "compact head detected")


def read_club_marking(image: Image.Image, config: Dict) -> OcrResult:
    settings = config.get("club_recognition", {})
    ocr_backend = settings.get("ocr_backend", "auto")

    cropped = preprocess_marking_region(image)

    if ocr_backend in {"easyocr", "auto"}:
        text, confidence = _try_easyocr(cropped)
        if text:
            return OcrResult(text=text, confidence=confidence, source="easyocr")

    if ocr_backend in {"tesseract", "auto"}:
        text, confidence = _try_tesseract(cropped)
        if text:
            return OcrResult(text=text, confidence=confidence, source="tesseract")

    # TODO: Replace with a custom marking classifier if OCR is unreliable.
    return OcrResult(text=None, confidence=0.0, source="ocr_unavailable")


def preprocess_marking_region(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_height = int(height * 0.5)
    crop_width = int(width * 0.6)
    left = max(0, (width - crop_width) // 2)
    upper = max(0, height - crop_height)
    right = min(width, left + crop_width)
    lower = min(height, upper + crop_height)
    return image.crop((left, upper, right, lower))


def interpret_marking(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = "".join(ch for ch in text.upper().strip() if ch.isalnum())
    if not cleaned:
        return None

    if cleaned.isdigit():
        number = int(cleaned)
        if 1 <= number <= 9:
            return f"{number} Iron"
        if number > 40:
            return f"{number}° Wedge"

    if cleaned in {"S", "SW", "A", "AW", "P", "PW"}:
        return f"{cleaned} Wedge"

    return None


def aggregate_confidence(values: list[float]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


def join_reasoning(parts: list[Optional[str]]) -> Optional[str]:
    cleaned = [p for p in parts if p]
    if not cleaned:
        return None
    return "; ".join(cleaned)


def predicted_to_category(predicted: Optional[str], broad_category: str) -> str:
    if broad_category == "iron_wedge":
        return "iron_wedge"
    if predicted == "Hybrid":
        return "hybrid"
    return "driver_wood"


def select_best_box(result, class_name: Optional[str]):
    boxes = list(result.boxes)
    if not boxes:
        return None

    if class_name and hasattr(result, "names"):
        normalized = str(class_name).strip().lower()
        name_map = result.names
        filtered = []
        for box in boxes:
            class_id = int(box.cls)
            label = name_map.get(class_id) if isinstance(name_map, dict) else name_map[class_id]
            if str(label).lower() == normalized:
                filtered.append(box)
        if filtered:
            return max(filtered, key=lambda b: float(b.conf))
        return None

    return max(boxes, key=lambda b: float(b.conf))


@lru_cache(maxsize=1)
def _get_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    # TODO: Replace with a real club-head detector weights file.
    if not model_path:
        return None
    return YOLO(model_path)


def _try_easyocr(image: Image.Image) -> tuple[Optional[str], float]:
    try:
        import easyocr
    except Exception:
        return None, 0.0

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image)
    if not results:
        return None, 0.0
    best = max(results, key=lambda r: r[2])
    return str(best[1]).strip(), float(best[2])


def _try_tesseract(image: Image.Image) -> tuple[Optional[str], float]:
    try:
        import pytesseract
    except Exception:
        return None, 0.0

    text = pytesseract.image_to_string(image, config="--psm 6")
    cleaned = text.strip()
    return cleaned if cleaned else None, 0.35 if cleaned else 0.0
