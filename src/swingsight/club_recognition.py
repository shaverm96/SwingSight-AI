from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from PIL import Image

from swingsight.club_cnn import CnnPrediction, classify_image
from swingsight.club_localization import ClubCropResult, locate_club_crop
from swingsight.club_marking_ocr import OcrCandidate, OcrReadResult, read_marking_candidates
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


@dataclass(frozen=True)
class ClubMarkingResult:
    """Exact Iron or Wedge result produced by pretrained full-image OCR."""

    designation: Optional[str]
    exact_club: Optional[str]
    raw_text: Optional[str]
    confidence: float
    bbox: Optional[list[list[float]]]
    reasoning: Optional[str]
    source: str
    elapsed_ms: float
    attempts: int


def recognize_club_from_frame(image_path: str, config: Dict) -> Dict:
    image = Image.open(image_path).convert("RGB")

    # The five-way CNN is trained on catalog-style photos of an isolated club
    # head, not full "person holding a club" frames. Crop toward a detected
    # hand first (using the MediaPipe HandLandmarker model already bundled
    # with the project) so the classifier sees something closer to its
    # training distribution. Unlike an earlier pose-based version, this does
    # not require a visible body or golf stance. This is a heuristic ROI,
    # not a trained club detector -- see swingsight.club_localization for
    # details and failure modes.
    localization_settings = config.get("club_localization", {}) or {}
    crop_result = None
    if bool(localization_settings.get("enabled", True)):
        try:
            crop_result = locate_club_crop(
                image,
                model_path=localization_settings.get("hand_model_path", "hand_landmarker.task"),
                min_detection_confidence=float(localization_settings.get("min_detection_confidence", 0.3)),
                min_presence_confidence=float(localization_settings.get("min_presence_confidence", 0.3)),
                crop_scale=float(localization_settings.get("crop_scale", 5.5)),
                max_crop_fraction=float(localization_settings.get("max_crop_fraction", 0.75)),
                merge_ratio=float(localization_settings.get("merge_ratio", 1.5)),
            )
        except Exception as error:
            # Belt-and-suspenders: locate_club_crop already catches its own
            # failures, but a crop is never allowed to block classification,
            # so a second guard here costs nothing.
            crop_result = ClubCropResult(image, False, None, f"Club localization raised an error ({error}); classifying the full frame.")
    working_image = crop_result.image if crop_result is not None else image

    preprocessing = config.get("preprocessing", {}) or {}
    enhanced_image = apply_adaptive_contrast_rgb(
        working_image,
        enabled=bool(preprocessing.get("adaptive_contrast", True)),
        clip_limit=float(preprocessing.get("adaptive_contrast_clip_limit", 2.0)),
        tile_grid_size=tuple(preprocessing.get("adaptive_contrast_tile_grid_size", [8, 8])) if preprocessing.get("adaptive_contrast_tile_grid_size") else (8, 8),
    )

    # The five-way checkpoint is the preferred live-capture model when configured.
    # Keep the existing staged path available for installations that do not use it.
    five_way = classify_five_way_club_type(enhanced_image, config)
    if five_way.source != "not_configured":
        result = _five_way_recognition_result(enhanced_image, five_way, config)
        result["club_localization"] = {
            "cropped": bool(crop_result.cropped) if crop_result is not None else False,
            "box": list(crop_result.box) if crop_result is not None and crop_result.box else None,
            "reasoning": crop_result.reasoning if crop_result is not None else "Club localization disabled by config.",
        }
        return result

    broad = classify_broad_category(enhanced_image, config)

    detail: ClubDetailResult
    marking: Optional[ClubMarkingResult] = None
    if broad.category == "wood":
        detail = classify_wood_type(enhanced_image, config)
        predicted = detail.club_type
        detected_category = detail.club_type or "Wood"
        ocr_payload = None
        sources = {"broad_classifier": broad.source, "wood_type_classifier": detail.source}
    elif broad.category == "iron":
        marking = classify_club_marking(enhanced_image, "Iron", config)
        detail = ClubDetailResult(
            club_type=marking.exact_club,
            confidence=marking.confidence,
            probabilities={},
            reasoning=marking.reasoning,
            source=marking.source,
        )
        predicted = marking.exact_club or "Iron"
        detected_category = "Iron"
        ocr_payload = _ocr_payload(marking)
        sources = {"broad_classifier": broad.source, "club_marking_ocr": detail.source}
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
    confidence = aggregate_hierarchical_confidence(broad.confidence, detail.confidence) if marking and marking.exact_club else broad.confidence
    status = recognition_status(predicted, confidence, confirm_threshold, broad, detail)
    if marking and not marking.exact_club:
        status = "marking_unavailable" if marking.source in _OCR_UNAVAILABLE_SOURCES else "needs_marking"

    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "club_type": detected_category if detected_category != "Unknown" else None,
        "club_number": marking.designation if marking else None,
        "exact_club": marking.exact_club if marking else None,
        "confidence": round(float(confidence), 3),
        "reasoning": join_reasoning([broad.reasoning, detail.reasoning]),
        "bbox": marking.bbox if marking else None,
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
    marking: Optional[ClubMarkingResult] = None
    predicted = family
    confidence = prediction.confidence
    ocr_payload = None

    if family in {"Iron", "Wedge"} and prediction.source not in unavailable_sources:
        marking = classify_club_marking(image, family, config)
        ocr_payload = _ocr_payload(marking)
        if marking.exact_club:
            predicted = marking.exact_club
            confidence = (0.55 * prediction.confidence) + (0.45 * marking.confidence)

    if prediction.source in unavailable_sources:
        status = "unavailable"
    elif family in {"Iron", "Wedge"} and (marking is None or not marking.exact_club):
        status = "marking_unavailable" if marking and marking.source in _OCR_UNAVAILABLE_SOURCES else "needs_marking"
    elif predicted and confidence >= confirm_threshold:
        status = "confirmed"
    else:
        status = "uncertain"

    detected_category = family or "Unknown"
    sources = {"club_type_5way_classifier": prediction.source}
    stage_confidences = {
        "club_type_5way": round(float(prediction.confidence), 3),
    }
    probabilities = {"club_type_5way": prediction.probabilities}
    reasoning = [prediction.reasoning]
    if marking is not None:
        sources["club_marking_ocr"] = marking.source
        stage_confidences["club_marking_ocr"] = round(float(marking.confidence), 3)
        probabilities["club_marking_ocr"] = {}
        reasoning.append(marking.reasoning)

    return {
        "status": status,
        "detected_category": detected_category,
        "predicted_club": predicted or "Unknown",
        "club_type": family,
        "club_number": marking.designation if marking else None,
        "exact_club": marking.exact_club if marking else None,
        "confidence": round(float(confidence), 3),
        "reasoning": join_reasoning(reasoning),
        "bbox": marking.bbox if marking else None,
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


_OCR_UNAVAILABLE_SOURCES = {"ocr_unavailable", "ocr_unsupported_backend", "ocr_inference_error"}


def classify_club_marking(image: Image.Image, family: str, config: Dict) -> ClubMarkingResult:
    """Read an exact club marking from the complete image with pretrained OCR.

    The accepted values are intentionally constrained by ``family``. A word
    such as ``SW`` cannot turn an Iron into a Wedge, and an arbitrary number
    cannot become a guessed Iron.
    """

    settings = config.get("club_recognition", {}) or {}
    ocr_settings = settings.get("marking_ocr", {}) or {}
    minimum_confidence = _bounded_confidence(ocr_settings.get("min_confidence", 0.70))

    first_pass = read_marking_candidates(image, config, rotations=(0,))
    result = _select_marking_candidate(first_pass, family, minimum_confidence)
    if result.exact_club or first_pass.source in _OCR_UNAVAILABLE_SOURCES:
        return result

    if not bool(ocr_settings.get("try_sideways_rotations_on_failure", True)):
        return result

    sideways_pass = read_marking_candidates(image, config, rotations=(90, 270))
    combined = OcrReadResult(
        candidates=first_pass.candidates + sideways_pass.candidates,
        source=sideways_pass.source if sideways_pass.source != "rapidocr" else first_pass.source,
        reasoning=join_reasoning([first_pass.reasoning, sideways_pass.reasoning]),
        elapsed_ms=first_pass.elapsed_ms + sideways_pass.elapsed_ms,
        attempts=first_pass.attempts + sideways_pass.attempts,
    )
    return _select_marking_candidate(combined, family, minimum_confidence)


def _select_marking_candidate(
    ocr_result: OcrReadResult,
    family: str,
    minimum_confidence: float,
) -> ClubMarkingResult:
    valid: list[tuple[OcrCandidate, str, str]] = []
    below_threshold: list[tuple[OcrCandidate, str, str]] = []

    for candidate in ocr_result.candidates:
        normalized = normalize_marking_designation(candidate.text, family)
        if normalized is None:
            continue
        designation, exact_club = normalized
        item = (candidate, designation, exact_club)
        if candidate.confidence >= minimum_confidence:
            valid.append(item)
        else:
            below_threshold.append(item)

    if valid:
        candidate, designation, exact_club = max(valid, key=lambda item: item[0].confidence)
        return ClubMarkingResult(
            designation=designation,
            exact_club=exact_club,
            raw_text=candidate.text,
            confidence=candidate.confidence,
            bbox=candidate.bbox,
            reasoning=None,
            source=ocr_result.source,
            elapsed_ms=ocr_result.elapsed_ms,
            attempts=ocr_result.attempts,
        )

    if below_threshold:
        candidate, designation, exact_club = max(below_threshold, key=lambda item: item[0].confidence)
        reasoning = (
            f"OCR read {candidate.text!r} as {designation!r}, but its confidence "
            f"({candidate.confidence:.2f}) is below the {minimum_confidence:.2f} threshold."
        )
        return ClubMarkingResult(
            designation=None,
            exact_club=None,
            raw_text=candidate.text,
            confidence=candidate.confidence,
            bbox=candidate.bbox,
            reasoning=reasoning,
            source=ocr_result.source,
            elapsed_ms=ocr_result.elapsed_ms,
            attempts=ocr_result.attempts,
        )

    return ClubMarkingResult(
        designation=None,
        exact_club=None,
        raw_text=ocr_result.candidates[0].text if ocr_result.candidates else None,
        confidence=max((candidate.confidence for candidate in ocr_result.candidates), default=0.0),
        bbox=ocr_result.candidates[0].bbox if ocr_result.candidates else None,
        reasoning=ocr_result.reasoning or f"No valid {family.lower()} marking was recognized.",
        source=ocr_result.source,
        elapsed_ms=ocr_result.elapsed_ms,
        attempts=ocr_result.attempts,
    )


def _ocr_payload(marking: ClubMarkingResult) -> Dict:
    return {
        "text": marking.designation,
        "raw_text": marking.raw_text,
        "confidence": round(float(marking.confidence), 3),
        "source": marking.source,
        "bbox": marking.bbox,
        "elapsed_ms": marking.elapsed_ms,
        "attempts": marking.attempts,
    }


def _bounded_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.70


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


def _normalize_club_marking_legacy(label: Optional[str], family: str) -> Optional[str]:
    """Legacy custom-CNN label normalization retained for checkpoint compatibility."""

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


def normalize_marking_designation(label: Optional[str], family: str) -> Optional[tuple[str, str]]:
    """Normalize OCR text only when it is valid for the detected club family.

    OCR character substitutions are context-sensitive: ``S`` remains a valid
    sand-wedge mark, but can represent ``5`` when a two-digit loft is read.
    This prevents unrelated branding text from becoming a guessed club.
    """

    if label is None:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", str(label).upper())
    if not compact:
        return None

    if family == "Iron":
        compact = compact.replace("IRON", "")
        corrected = compact.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6"}))
        if re.fullmatch(r"[2-9]", corrected):
            return corrected, f"{corrected} Iron"
        return None

    if family != "Wedge":
        return None

    aliases = {
        "P": ("PW", "Pitching Wedge"), "PW": ("PW", "Pitching Wedge"),
        "PITCHING": ("PW", "Pitching Wedge"), "PITCHINGWEDGE": ("PW", "Pitching Wedge"),
        "A": ("AW", "Approach Wedge"), "AW": ("AW", "Approach Wedge"),
        "APPROACH": ("AW", "Approach Wedge"), "APPROACHWEDGE": ("AW", "Approach Wedge"),
        "G": ("GW", "Gap Wedge"), "GW": ("GW", "Gap Wedge"),
        "GAP": ("GW", "Gap Wedge"), "GAPWEDGE": ("GW", "Gap Wedge"),
        "S": ("SW", "Sand Wedge"), "SW": ("SW", "Sand Wedge"),
        "SAND": ("SW", "Sand Wedge"), "SANDWEDGE": ("SW", "Sand Wedge"),
        "L": ("LW", "Lob Wedge"), "LW": ("LW", "Lob Wedge"),
        "LOB": ("LW", "Lob Wedge"), "LOBWEDGE": ("LW", "Lob Wedge"),
        # Common single-character OCR swaps in a two-letter wedge mark.
        "5W": ("SW", "Sand Wedge"), "6W": ("GW", "Gap Wedge"),
    }
    if compact in aliases:
        return aliases[compact]

    loft_text = compact.removesuffix("DEGREES").removesuffix("DEG").removeprefix("WEDGE")
    corrected_loft = loft_text.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6"}))
    if re.fullmatch(r"(?:4[6-9]|5[0-9]|6[0-4])", corrected_loft):
        return corrected_loft, f"{corrected_loft}° Wedge"
    return None


def normalize_club_marking(label: Optional[str], family: str) -> Optional[str]:
    """Return the exact player-facing name for a valid OCR marking."""

    normalized = normalize_marking_designation(label, family)
    return normalized[1] if normalized else None


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
    if broad_category == "hybrid":
        return "hybrid"
    if broad_category != "wood":
        return "unknown"
    if predicted == "Hybrid":
        return "hybrid"
    return "driver_wood"
