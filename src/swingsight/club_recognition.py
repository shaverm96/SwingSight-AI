from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from PIL import Image

from swingsight.club_cnn import CnnPrediction, classify_image
from swingsight.image_preprocessing import apply_adaptive_contrast_rgb
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

    # The five-way checkpoint is the preferred live-capture model when configured.
    # Keep the existing staged path available for installations that do not use it.
    five_way = classify_five_way_club_type(enhanced_image, config)
    if five_way.source != "not_configured":
        return _five_way_recognition_result(enhanced_image, five_way, config)

    broad = classify_broad_category(enhanced_image, config)

    detail: ClubDetailResult
    if broad.category == "wood":
        detail = classify_wood_type(enhanced_image, config)
        predicted = detail.club_type
        detected_category = "Wood"
        ocr_payload = None
        sources = {"broad_classifier": broad.source, "wood_type_classifier": detail.source}
    elif broad.category == "iron":
        marking_crop = preprocess_marking_region(enhanced_image)
        detail = classify_iron_number(marking_crop, config)
        predicted = detail.club_type
        detected_category = "Iron"
        ocr_payload = {
            "text": _iron_text_from_prediction(predicted),
            "confidence": round(float(detail.confidence), 3),
            "source": detail.source,
        }
        sources = {"broad_classifier": broad.source, "iron_number_classifier": detail.source}
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
        sources = {"broad_classifier": broad.source}

    confirm_threshold = float(config.get("club_recognition", {}).get("confirm_threshold", 0.6))
    confidence = aggregate_hierarchical_confidence(broad.confidence, detail.confidence)
    status = recognition_status(predicted, confidence, confirm_threshold, broad, detail)

    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "confidence": round(float(confidence), 3),
        "reasoning": join_reasoning([broad.reasoning, detail.reasoning]),
        "bbox": None,
        "ocr": ocr_payload,
        "sources": sources,
        "stage_confidences": {
            "broad_category": round(float(broad.confidence), 3),
            "detail": round(float(detail.confidence), 3),
        },
        "probabilities": {
            "broad_category": broad.probabilities,
            "detail": detail.probabilities,
        },
    }

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
    image: Image.Image,
    prediction: ClubDetailResult,
    config: Dict,
) -> Dict:
    """Finish the five-way decision and, for irons/wedges, read the club marking."""

    settings = config.get("club_recognition", {})
    confirm_threshold = float(settings.get("confirm_threshold", 0.6))
    unavailable_sources = {"cnn_missing", "cnn_unavailable", "cnn_invalid", "cnn_inference_error"}
    family = prediction.club_type
    marking: Optional[ClubDetailResult] = None
    predicted = family
    confidence = prediction.confidence
    ocr_payload = None

    if family in {"Iron", "Wedge"} and prediction.source not in unavailable_sources:
        marking_crop = preprocess_marking_region(image)
        marking = classify_club_marking(marking_crop, family, config)
        if marking.club_type:
            predicted = marking.club_type
            confidence = (0.55 * prediction.confidence) + (0.45 * marking.confidence)
            ocr_payload = {
                "text": _club_marking_text(marking.club_type),
                "confidence": round(float(marking.confidence), 3),
                "source": marking.source,
            }

    if prediction.source in unavailable_sources:
        status = "unavailable"
    elif family in {"Iron", "Wedge"} and (marking is None or not marking.club_type):
        # Do not invent a number or loft. Let the UI ask for a tighter sole/face view.
        status = "needs_marking"
    elif predicted and confidence >= confirm_threshold:
        status = "confirmed"
    else:
        status = "uncertain"

    detected_category = "Wood" if family in {"Driver", "Wood", "Hybrid"} else family or "Unknown"
    sources = {"club_type_5way_classifier": prediction.source}
    stage_confidences = {
        "club_type_5way": round(float(prediction.confidence), 3),
    }
    probabilities = {"club_type_5way": prediction.probabilities}
    reasoning = [prediction.reasoning]
    if marking is not None:
        sources["club_marking_classifier"] = marking.source
        stage_confidences["club_marking"] = round(float(marking.confidence), 3)
        probabilities["club_marking"] = marking.probabilities
        reasoning.append(marking.reasoning)

    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "confidence": round(float(confidence), 3),
        "reasoning": join_reasoning(reasoning),
        "bbox": None,
        "ocr": ocr_payload,
        "sources": sources,
        "stage_confidences": stage_confidences,
        "probabilities": probabilities,
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


def classify_club_marking(image: Image.Image, family: str, config: Dict) -> ClubDetailResult:
    """Read an iron number, wedge loft, or letter from the club's sole/face."""

    settings = config.get("club_recognition", {})
    checkpoint_path = settings.get("club_marking_cnn_model_path")
    if checkpoint_path:
        prediction = _run_stage_cnn(
            image,
            checkpoint_path=checkpoint_path,
            task="club_marking",
            minimum_confidence=float(settings.get("club_marking_cnn_min_confidence", 0.0)),
        )
        club_type = normalize_club_marking(prediction.label, family)
        reasoning = prediction.reasoning
        if prediction.label is not None and club_type is None:
            reasoning = (
                f"Club-marking CNN returned {prediction.label!r}, which is not valid "
                f"for a {family.lower()}."
            )
        return ClubDetailResult(
            club_type=club_type,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            reasoning=reasoning,
            source=prediction.source,
        )

    # Preserve compatibility with an existing numbers-only iron checkpoint.
    if family == "Iron" and settings.get("iron_number_cnn_model_path"):
        return classify_iron_number(image, config)

    return ClubDetailResult(
        club_type=None,
        confidence=0.0,
        probabilities={},
        reasoning=(
            "The club family is visible, but the marking model is not configured. "
            "Install models/trained/club_marking_cnn.pt to identify the exact club."
        ),
        source="not_configured",
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


def aggregate_hierarchical_confidence(broad: float, detail: float) -> float:
    """Combine the two fallback CNN decisions."""

    broad_score = max(0.0, min(1.0, float(broad)))
    detail_score = max(0.0, min(1.0, float(detail)))
    return (0.5 * broad_score) + (0.5 * detail_score)


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


def normalize_club_marking(label: Optional[str], family: str) -> Optional[str]:
    """Convert a marking-CNN label into the exact player-facing club name."""

    if label is None:
        return None
    normalized = _normalize_label(label)
    iron_match = re.fullmatch(r"(?:iron_?)?([1-9])(?:_?iron)?", normalized)
    if family == "Iron" and iron_match:
        return f"{iron_match.group(1)} Iron"

    wedge_letters = {
        "p": "Pitching Wedge",
        "pw": "Pitching Wedge",
        "pitching_wedge": "Pitching Wedge",
        "a": "Approach Wedge",
        "aw": "Approach Wedge",
        "approach_wedge": "Approach Wedge",
        "g": "Gap Wedge",
        "gw": "Gap Wedge",
        "gap_wedge": "Gap Wedge",
        "s": "Sand Wedge",
        "sw": "Sand Wedge",
        "sand_wedge": "Sand Wedge",
        "l": "Lob Wedge",
        "lw": "Lob Wedge",
        "lob_wedge": "Lob Wedge",
    }
    if normalized in wedge_letters:
        return wedge_letters[normalized]

    degree_match = re.fullmatch(r"(?:wedge_?)?(4[6-9]|5[0-9]|6[0-4])(?:_?deg(?:ree)?)?", normalized)
    if family == "Wedge" and degree_match:
        return f"{degree_match.group(1)}° Wedge"
    return None


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


def _club_marking_text(predicted: Optional[str]) -> Optional[str]:
    if not predicted:
        return None
    number_match = re.match(r"^([1-9]) Iron$", predicted)
    if number_match:
        return number_match.group(1)
    degree_match = re.match(r"^(\d{2})° Wedge$", predicted)
    if degree_match:
        return degree_match.group(1)
    initials = {
        "Pitching Wedge": "P",
        "Approach Wedge": "A",
        "Gap Wedge": "G",
        "Sand Wedge": "S",
        "Lob Wedge": "L",
    }
    return initials.get(predicted)


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
