"""Lightweight, full-image OCR for exact iron and wedge markings.

The club-type CNN owns the broad Driver/Wood/Hybrid/Iron/Wedge decision.
This module deliberately has no golf-club training dependency: it uses the
pretrained PP-OCR models shipped by RapidOCR and ONNX Runtime only after the
first stage identifies an Iron or Wedge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from time import perf_counter
from typing import Any, Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OcrCandidate:
    """One text item returned by a pretrained OCR engine."""

    text: str
    confidence: float
    bbox: Optional[list[list[float]]] = None
    rotation_degrees: int = 0


@dataclass(frozen=True)
class OcrReadResult:
    """A non-throwing OCR result suitable for the recognition pipeline."""

    candidates: tuple[OcrCandidate, ...]
    source: str
    reasoning: Optional[str]
    elapsed_ms: float
    attempts: int


def is_ocr_backend_available(settings: Optional[Dict] = None) -> bool:
    """Return whether the configured optional OCR dependency is installed."""

    backend = str((settings or {}).get("backend", "rapidocr")).lower()
    return backend == "rapidocr" and importlib.util.find_spec("rapidocr") is not None


@lru_cache(maxsize=1)
def _get_rapidocr_engine() -> tuple[Optional[Any], Optional[str]]:
    """Create RapidOCR once; model initialization is intentionally lazy."""

    try:
        from rapidocr import RapidOCR
    except Exception as exc:  # pragma: no cover - depends on optional package state
        return None, f"RapidOCR is unavailable: {type(exc).__name__}. Install rapidocr and onnxruntime."

    try:
        return RapidOCR(), None
    except Exception as exc:  # pragma: no cover - model/runtime initialization is environment-specific
        return None, f"RapidOCR could not initialize: {type(exc).__name__}."


def read_marking_candidates(
    image: Image.Image,
    config: Dict,
    *,
    rotations: Sequence[int] = (0,),
) -> OcrReadResult:
    """Read all text in a full club image with a bounded set of rotations.

    The first call normally uses only the original orientation. Callers should
    request the two sideways rotations only after no valid club marking is
    found, keeping ordinary inference to a single OCR pass.
    """

    settings = (config.get("club_recognition", {}) or {}).get("marking_ocr", {}) or {}
    backend = str(settings.get("backend", "rapidocr")).lower()
    if backend != "rapidocr":
        return OcrReadResult(
            candidates=(),
            source="ocr_unsupported_backend",
            reasoning=f"Unsupported club-marking OCR backend {backend!r}.",
            elapsed_ms=0.0,
            attempts=0,
        )

    engine, error = _get_rapidocr_engine()
    if engine is None:
        return OcrReadResult(
            candidates=(),
            source="ocr_unavailable",
            reasoning=error,
            elapsed_ms=0.0,
            attempts=0,
        )

    started = perf_counter()
    candidates: list[OcrCandidate] = []
    failures: list[str] = []
    original = _prepare_image(image, settings)
    original_height, original_width = original.shape[:2]

    for rotation in rotations:
        try:
            normalized_rotation = int(rotation) % 360
        except (TypeError, ValueError):
            continue
        if normalized_rotation not in {0, 90, 180, 270}:
            continue

        prepared = _rotate_image(original, normalized_rotation)
        try:
            output = engine(prepared)
        except Exception as exc:  # pragma: no cover - defensive boundary around optional model runtime
            failures.append(type(exc).__name__)
            continue

        for candidate in _extract_candidates(output):
            candidates.append(
                OcrCandidate(
                    text=candidate.text,
                    confidence=candidate.confidence,
                    bbox=_restore_bbox(candidate.bbox, normalized_rotation, original_width, original_height),
                    rotation_degrees=normalized_rotation,
                )
            )

    elapsed_ms = (perf_counter() - started) * 1000.0
    if failures and not candidates:
        reasoning = f"RapidOCR inference failed ({', '.join(sorted(set(failures)))})."
        source = "ocr_inference_error"
    elif not candidates:
        reasoning = "No readable text was found in the submitted club image."
        source = "rapidocr"
    else:
        reasoning = None
        source = "rapidocr"

    return OcrReadResult(
        candidates=tuple(candidates),
        source=source,
        reasoning=reasoning,
        elapsed_ms=round(elapsed_ms, 1),
        attempts=len(tuple(rotations)),
    )


def _prepare_image(image: Image.Image, settings: Dict) -> np.ndarray:
    """Apply a small, deterministic preprocessing step without cropping."""

    rgb = np.asarray(image.convert("RGB"))
    max_side = max(1, int(settings.get("max_image_side", 1600)))
    height, width = rgb.shape[:2]
    current_max = max(height, width)
    if current_max > max_side:
        scale = max_side / float(current_max)
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

    if not bool(settings.get("enhance_contrast", True)):
        return rgb

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(settings.get("contrast_clip_limit", 2.0)), tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if bool(settings.get("sharpen", False)):
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.rot90(image, 1).copy()
    if rotation == 180:
        return np.rot90(image, 2).copy()
    if rotation == 270:
        return np.rot90(image, 3).copy()
    return image


def _extract_candidates(output: Any) -> Iterable[OcrCandidate]:
    """Support RapidOCR's current data class and the legacy tuple response."""

    if output is None:
        return ()

    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    boxes = getattr(output, "boxes", None)
    if texts is not None:
        return _zip_candidates(texts, scores, boxes)

    if isinstance(output, tuple) and output:
        output = output[0]
    if isinstance(output, list):
        legacy: list[OcrCandidate] = []
        for item in output:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, text_score = item[0], item[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                continue
            legacy.append(OcrCandidate(str(text_score[0]), _safe_confidence(text_score[1]), _coerce_box(box)))
        return legacy
    return ()


def _zip_candidates(texts: Iterable[Any], scores: Optional[Iterable[Any]], boxes: Optional[Iterable[Any]]) -> list[OcrCandidate]:
    text_values = list(texts or ())
    score_values = list(scores or ())
    box_values = list(boxes or ())
    return [
        OcrCandidate(
            text=str(text),
            confidence=_safe_confidence(score_values[index] if index < len(score_values) else 0.0),
            bbox=_coerce_box(box_values[index] if index < len(box_values) else None),
        )
        for index, text in enumerate(text_values)
        if str(text).strip()
    ]


def _safe_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _coerce_box(box: Any) -> Optional[list[list[float]]]:
    if box is None:
        return None
    try:
        array = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != 2:
        return None
    return [[round(float(x), 2), round(float(y), 2)] for x, y in array.tolist()]


def _restore_bbox(
    bbox: Optional[list[list[float]]], rotation: int, original_width: int, original_height: int
) -> Optional[list[list[float]]]:
    if bbox is None or rotation == 0:
        return bbox

    restored: list[list[float]] = []
    for x, y in bbox:
        if rotation == 90:
            restored.append([round(original_width - 1 - y, 2), round(x, 2)])
        elif rotation == 180:
            restored.append([round(original_width - 1 - x, 2), round(original_height - 1 - y, 2)])
        else:  # 270 degrees counter-clockwise
            restored.append([round(y, 2), round(original_height - 1 - x, 2)])
    return restored
