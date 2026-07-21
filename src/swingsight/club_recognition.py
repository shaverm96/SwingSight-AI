from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Optional, Tuple

from PIL import Image

from swingsight.club_cnn import CnnPrediction, classify_image
from swingsight.image_preprocessing import apply_adaptive_contrast_rgb
from swingsight.runtime import select_yolo_device


@dataclass(frozen=True)
class ClubHeadDetection:
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    source: str


@dataclass(frozen=True)
class BroadCategoryResult:
    category: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    reasoning: Optional[str]
    source: str


@dataclass(frozen=True)
class ClubDetailResult:
    club_type: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    reasoning: Optional[str]
    source: str


def recognize_club_from_frame(image_path: str, config: Dict) -> Dict:
    image = Image.open(image_path).convert("RGB")
    preprocessing = config.get("preprocessing", {}) or {}
    enhanced_image = apply_adaptive_contrast_rgb(
        image,
        enabled=bool(preprocessing.get("adaptive_contrast", True)),
        clip_limit=float(preprocessing.get("adaptive_contrast_clip_limit", 2.0)),
        tile_grid_size=tuple(preprocessing.get("adaptive_contrast_tile_grid_size", [8, 8])) if preprocessing.get("adaptive_contrast_tile_grid_size") else (8, 8),
    )

    detection = detect_club_head(enhanced_image, config)
    crop = crop_region(enhanced_image, detection.bbox)

    # The five-way checkpoint is the preferred live-capture model when configured.
    # Keep the existing staged path available for installations that do not use it.
    five_way = classify_five_way_club_type(crop, config)
    if five_way.source != "not_configured":
        return _five_way_recognition_result(detection, five_way, config)

    broad = classify_broad_category(crop, config)

    detail: ClubDetailResult
    if broad.category == "wood":
        detail = classify_wood_type(crop, config)
        predicted = detail.club_type
        detected_category = "Wood"
        ocr_payload = None
        sources = {
            "detector": detection.source,
            "broad_classifier": broad.source,
            "wood_type_classifier": detail.source,
        }
    elif broad.category == "iron":
        marking_crop = preprocess_marking_region(crop)
        detail = classify_iron_number(marking_crop, config)
        predicted = detail.club_type
        detected_category = "Iron"
        ocr_payload = {
            "text": _iron_text_from_prediction(predicted),
            "confidence": round(float(detail.confidence), 3),
            "source": detail.source,
        }
        sources = {
            "detector": detection.source,
            "broad_classifier": broad.source,
            "iron_number_classifier": detail.source,
        }
    else:
        detail = ClubDetailResult(
            club_type=None,
            confidence=0.0,
            probabilities={},
            reasoning="A valid iron-or-wood prediction is required before the next CNN stage can run.",
            source="not_run",
        )
        predicted = None
        detected_category = "Unknown"
        ocr_payload = None
        sources = {"detector": detection.source, "broad_classifier": broad.source}

    confirm_threshold = float(config.get("club_recognition", {}).get("confirm_threshold", 0.6))
    confidence = aggregate_hierarchical_confidence(
        detection.confidence,
        broad.confidence,
        detail.confidence,
    )
    status = recognition_status(predicted, confidence, confirm_threshold, broad, detail)

    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "confidence": round(float(confidence), 3),
        "reasoning": join_reasoning([
            "club-head detector unavailable; classified the submitted frame" if detection.source == "fallback_full_frame" else None,
            broad.reasoning,
            detail.reasoning,
        ]),
        "bbox": detection.bbox,
        "ocr": ocr_payload,
        "sources": sources,
        "stage_confidences": {
            "detector": round(float(detection.confidence), 3),
            "broad_category": round(float(broad.confidence), 3),
            "detail": round(float(detail.confidence), 3),
        },
        "probabilities": {
            "broad_category": broad.probabilities,
            "detail": detail.probabilities,
        },
    }


def detect_club_head(image: Image.Image, config: Dict) -> ClubHeadDetection:
    settings = config.get("club_recognition", {})
    model_path = settings.get("yolo_model_path", "models/club_detector.pt")
    confidence_threshold = float(settings.get("detector_confidence_threshold", 0.4))
    class_name = settings.get("yolo_class_name")

    try:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"YOLO model not found at {path}")
        model = _get_yolo_model(str(path))
        if model is None:
            raise RuntimeError("YOLO model not available")
        results = model.predict(image, conf=confidence_threshold, verbose=False, device=select_yolo_device())
        if not results:
            return _full_frame_detection(image)
        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
            return _full_frame_detection(image)

        best = select_best_box(result, class_name)
        if best is None:
            return _full_frame_detection(image)

        x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
        return ClubHeadDetection(bbox=(x1, y1, x2, y2), confidence=float(best.conf), source="yolov8")
    except Exception:
        return _full_frame_detection(image)


def _full_frame_detection(image: Image.Image) -> ClubHeadDetection:
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


def classify_five_way_club_type(image: Image.Image, config: Dict) -> ClubDetailResult:
    """Classify a club as Driver, Wood, Hybrid, Iron, or Wedge."""

    settings = config.get("club_recognition", {})
    checkpoint_path = settings.get("five_way_cnn_model_path")
    if not checkpoint_path:
        return ClubDetailResult(
            club_type=None,
            confidence=0.0,
            probabilities={},
            reasoning=None,
            source="not_configured",
        )

    prediction = _run_stage_cnn(
        image,
        checkpoint_path=checkpoint_path,
        task="club_type_5way",
        minimum_confidence=float(settings.get("five_way_cnn_min_confidence", 0.0)),
    )
    club_type = normalize_five_way_club_type(prediction.label)
    reasoning = prediction.reasoning
    if prediction.label is not None and club_type is None:
        reasoning = (
            f"Five-way club CNN returned unsupported label {prediction.label!r}; "
            "expected driver, wood, hybrid, iron, or wedge."
        )

    return ClubDetailResult(
        club_type=club_type,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        reasoning=reasoning,
        source=prediction.source,
    )


def _five_way_recognition_result(
    detection: ClubHeadDetection,
    prediction: ClubDetailResult,
    config: Dict,
) -> Dict:
    confirm_threshold = float(config.get("club_recognition", {}).get("confirm_threshold", 0.6))
    unavailable_sources = {"cnn_missing", "cnn_unavailable", "cnn_invalid", "cnn_inference_error"}
    if prediction.source in unavailable_sources:
        status = "unavailable"
    elif prediction.club_type and prediction.confidence >= confirm_threshold:
        status = "confirmed"
    else:
        status = "uncertain"

    predicted = prediction.club_type
    detected_category = "Wood" if predicted in {"Driver", "Wood", "Hybrid"} else "Iron" if predicted else "Unknown"
    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "confidence": round(float(prediction.confidence), 3),
        "reasoning": join_reasoning([
            "club-head detector unavailable; classified the submitted frame" if detection.source == "fallback_full_frame" else None,
            prediction.reasoning,
        ]),
        "bbox": detection.bbox,
        "ocr": None,
        "sources": {
            "detector": detection.source,
            "club_type_5way_classifier": prediction.source,
        },
        "stage_confidences": {
            "detector": round(float(detection.confidence), 3),
            "club_type_5way": round(float(prediction.confidence), 3),
        },
        "probabilities": {
            "club_type_5way": prediction.probabilities,
        },
    }


def classify_broad_category(image: Image.Image, config: Dict) -> BroadCategoryResult:
    settings = config.get("club_recognition", {})
    prediction = _run_stage_cnn(
        image,
        checkpoint_path=settings.get("broad_cnn_model_path"),
        task="broad_category",
        minimum_confidence=float(settings.get("broad_cnn_min_confidence", 0.0)),
    )
    category = _normalize_broad_label(prediction.label)
    if prediction.label is not None and category is None:
        return BroadCategoryResult(
            category=None,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            reasoning=f"Broad-category CNN returned unsupported label {prediction.label!r}.",
            source=prediction.source,
        )
    return BroadCategoryResult(
        category=category,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        reasoning=prediction.reasoning,
        source=prediction.source,
    )


def classify_iron_number(image: Image.Image, config: Dict) -> ClubDetailResult:
    """Read the iron number from the bottom marking with the second CNN stage."""

    settings = config.get("club_recognition", {})
    prediction = _run_stage_cnn(
        image,
        checkpoint_path=settings.get("iron_number_cnn_model_path"),
        task="iron_number",
        minimum_confidence=float(settings.get("iron_number_cnn_min_confidence", 0.0)),
    )
    number = normalize_iron_number(prediction.label)
    if prediction.label is not None and number is None:
        reasoning = f"Iron-number CNN returned unsupported label {prediction.label!r}; expected 1 through 9."
    else:
        reasoning = prediction.reasoning
    return ClubDetailResult(
        club_type=f"{number} Iron" if number is not None else None,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        reasoning=reasoning,
        source=prediction.source,
    )


def classify_wood_type(image: Image.Image, config: Dict) -> ClubDetailResult:
    """Choose Driver, Wood, or Hybrid after the broad CNN selects wood."""

    settings = config.get("club_recognition", {})
    prediction = _run_stage_cnn(
        image,
        checkpoint_path=settings.get("wood_type_cnn_model_path"),
        task="wood_type",
        minimum_confidence=float(settings.get("wood_type_cnn_min_confidence", 0.0)),
    )
    club_type = normalize_wood_type(prediction.label)
    if prediction.label is not None and club_type is None:
        reasoning = f"Wood-type CNN returned unsupported label {prediction.label!r}."
    else:
        reasoning = prediction.reasoning
    return ClubDetailResult(
        club_type=club_type,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        reasoning=reasoning,
        source=prediction.source,
    )


def preprocess_marking_region(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_height = int(height * 0.5)
    crop_width = int(width * 0.6)
    left = max(0, (width - crop_width) // 2)
    upper = max(0, height - crop_height)
    right = min(width, left + crop_width)
    lower = min(height, upper + crop_height)
    return image.crop((left, upper, right, lower))


def aggregate_hierarchical_confidence(detector: float, broad: float, detail: float) -> float:
    """Weight the two CNN decisions above crop-localization certainty."""

    detector_score = max(0.0, min(1.0, float(detector)))
    broad_score = max(0.0, min(1.0, float(broad)))
    detail_score = max(0.0, min(1.0, float(detail)))
    return (0.15 * detector_score) + (0.425 * broad_score) + (0.425 * detail_score)


def recognition_status(
    predicted: Optional[str],
    confidence: float,
    confirm_threshold: float,
    broad: BroadCategoryResult,
    detail: ClubDetailResult,
) -> str:
    unavailable_sources = {"cnn_missing", "cnn_unavailable", "cnn_invalid", "cnn_inference_error"}
    if broad.source in unavailable_sources or detail.source in unavailable_sources:
        return "unavailable"
    if predicted and confidence >= confirm_threshold:
        return "confirmed"
    return "uncertain"


def _run_stage_cnn(
    image: Image.Image,
    *,
    checkpoint_path: object,
    task: str,
    minimum_confidence: float,
) -> CnnPrediction:
    return classify_image(
        image,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        expected_task=task,
        minimum_confidence=max(0.0, min(1.0, minimum_confidence)),
    )


def _normalize_broad_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    normalized = _normalize_label(label)
    if normalized in {"iron", "iron_wedge", "ironwedge", "wedge"}:
        return "iron"
    if normalized in {"wood", "woods", "wood_style", "woodstyle", "driver_wood", "driverwood"}:
        return "wood"
    return None


def normalize_five_way_club_type(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    normalized = _normalize_label(label)
    labels = {
        "driver": "Driver",
        "wood": "Wood",
        "fairway_wood": "Wood",
        "hybrid": "Hybrid",
        "iron": "Iron",
        "wedge": "Wedge",
    }
    return labels.get(normalized)


def normalize_iron_number(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    match = re.fullmatch(r"(?:iron_?)?([1-9])(?:_?iron)?", _normalize_label(label))
    if not match:
        return None
    return int(match.group(1))


def normalize_wood_type(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    normalized = _normalize_label(label)
    if normalized in {"driver", "d"}:
        return "Driver"
    if normalized in {"wood", "fairway_wood", "fairwaywood", "fairway", "fw"}:
        return "Wood"
    if normalized in {"hybrid", "rescue"}:
        return "Hybrid"
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _iron_text_from_prediction(predicted: Optional[str]) -> Optional[str]:
    if not predicted:
        return None
    match = re.match(r"^([1-9]) Iron$", predicted)
    return match.group(1) if match else None


def join_reasoning(parts: list[Optional[str]]) -> Optional[str]:
    cleaned = [p for p in parts if p]
    if not cleaned:
        return None
    return "; ".join(cleaned)


def predicted_to_category(predicted: Optional[str], broad_category: str) -> str:
    if broad_category == "iron_wedge":
        return "iron_wedge"
    if broad_category != "wood":
        return "unknown"
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
