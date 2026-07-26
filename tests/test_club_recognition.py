from __future__ import annotations

import pytest
from PIL import Image

import swingsight.club_recognition as club_recognition
from swingsight.club_cnn import SUPPORTED_TASKS
from swingsight.club_marking_ocr import OcrCandidate, OcrReadResult


def _image_path(tmp_path):
    path = tmp_path / "club.jpg"
    Image.new("RGB", (160, 120), color=(30, 80, 120)).save(path)
    return path


def _ocr_result(*candidates, source="rapidocr", reasoning=None, attempts=1):
    return OcrReadResult(
        candidates=tuple(candidates),
        source=source,
        reasoning=reasoning,
        elapsed_ms=12.5,
        attempts=attempts,
    )


def _five_way_result(family: str, confidence: float = 0.93):
    return club_recognition.ClubDetailResult(
        family,
        confidence,
        {family.lower(): confidence},
        f"five-way CNN selected {family.lower()}",
        "cnn",
    )


def test_wood_branch_uses_the_wood_type_cnn_without_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "classify_broad_category",
        lambda image, config: club_recognition.BroadCategoryResult(
            "wood", 0.9, {"iron": 0.1, "wood": 0.9}, "broad CNN selected wood", "cnn"
        ),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_wood_type",
        lambda image, config: club_recognition.ClubDetailResult(
            "Hybrid", 0.95, {"driver": 0.02, "wood": 0.03, "hybrid": 0.95}, "wood CNN selected hybrid", "cnn"
        ),
    )
    monkeypatch.setattr(club_recognition, "read_marking_candidates", lambda *args, **kwargs: pytest.fail("OCR must not run"))

    result = club_recognition.recognize_club_from_frame(str(_image_path(tmp_path)), {"club_recognition": {"confirm_threshold": 0.6}})

    assert result["status"] == "confirmed"
    assert result["detected_category"] == "Hybrid"
    assert result["predicted_club"] == "Hybrid"
    assert result["exact_club"] is None
    assert result["sources"]["wood_type_classifier"] == "cnn"


def test_iron_branch_uses_full_image_pretrained_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "classify_broad_category",
        lambda image, config: club_recognition.BroadCategoryResult(
            "iron", 0.94, {"iron": 0.94, "wood": 0.06}, "broad CNN selected iron", "cnn"
        ),
    )
    monkeypatch.setattr(
        club_recognition,
        "read_marking_candidates",
        lambda image, config, rotations=(0,): _ocr_result(OcrCandidate("7", 0.91, [[1.0, 2.0], [3.0, 4.0]])),
    )

    result = club_recognition.recognize_club_from_frame(str(_image_path(tmp_path)), {"club_recognition": {"confirm_threshold": 0.6}})

    assert result["status"] == "confirmed"
    assert result["club_type"] == "Iron"
    assert result["club_number"] == "7"
    assert result["exact_club"] == "7 Iron"
    assert result["predicted_club"] == "7 Iron"
    assert result["ocr"]["raw_text"] == "7"
    assert result["sources"]["club_marking_ocr"] == "rapidocr"


@pytest.mark.parametrize("number", ["2", "3", "4", "5", "6", "7", "8", "9"])
def test_every_supported_iron_number_is_normalized(number):
    assert club_recognition.normalize_marking_designation(number, "Iron") == (number, f"{number} Iron")


@pytest.mark.parametrize(
    ("raw", "designation", "exact_club"),
    [
        ("P", "PW", "Pitching Wedge"), ("PW", "PW", "Pitching Wedge"),
        ("A", "AW", "Approach Wedge"), ("AW", "AW", "Approach Wedge"),
        ("G", "GW", "Gap Wedge"), ("GW", "GW", "Gap Wedge"),
        ("S", "SW", "Sand Wedge"), ("SW", "SW", "Sand Wedge"),
        ("L", "LW", "Lob Wedge"), ("LW", "LW", "Lob Wedge"),
        ("5W", "SW", "Sand Wedge"), ("5G", "56", "56° Wedge"),
    ],
)
def test_wedge_markings_and_common_ocr_confusions_are_normalized(raw, designation, exact_club):
    assert club_recognition.normalize_marking_designation(raw, "Wedge") == (designation, exact_club)


def test_low_confidence_and_invalid_markings_keep_the_detected_family(monkeypatch, tmp_path):
    def reader(image, config, rotations=(0,)):
        if rotations == (0,):
            return _ocr_result(OcrCandidate("S", 0.42))
        return _ocr_result(OcrCandidate("BRAND", 0.99), attempts=2)

    monkeypatch.setattr(club_recognition, "classify_five_way_club_type", lambda image, config: _five_way_result("Wedge"))
    monkeypatch.setattr(club_recognition, "read_marking_candidates", reader)

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["status"] == "needs_marking"
    assert result["club_type"] == "Wedge"
    assert result["club_number"] is None
    assert result["exact_club"] is None
    assert result["predicted_club"] == "Wedge"
    assert result["ocr"]["raw_text"] == "S"
    assert result["ocr"]["confidence"] == 0.42


def test_sideways_ocr_is_only_tried_after_the_original_image_fails(monkeypatch):
    calls = []

    def reader(image, config, rotations=(0,)):
        calls.append(rotations)
        if rotations == (0,):
            return _ocr_result(OcrCandidate("BRAND", 0.99))
        return _ocr_result(OcrCandidate("6", 0.92), attempts=2)

    monkeypatch.setattr(club_recognition, "read_marking_candidates", reader)
    result = club_recognition.classify_club_marking(Image.new("RGB", (80, 60)), "Iron", {"club_recognition": {}})

    assert result.designation == "6"
    assert result.exact_club == "6 Iron"
    assert calls == [(0,), (90, 270)]


def test_tiled_ocr_fallback_recovers_a_small_wedge_loft(monkeypatch):
    calls = []

    def reader(image, config, rotations=(0,)):
        calls.append((image.size, rotations))
        if image.size == (240, 160):
            return _ocr_result(OcrCandidate("BRAND", 0.99))
        return _ocr_result(OcrCandidate("60", 0.94))

    monkeypatch.setattr(club_recognition, "read_marking_candidates", reader)

    result = club_recognition.classify_club_marking(
        Image.new("RGB", (240, 160)),
        "Wedge",
        {
            "club_recognition": {
                "marking_ocr": {
                    "fallback_tile_grid": [2, 2],
                    "fallback_tile_overlap": 0.0,
                    "fallback_tile_min_side": 1200,
                }
            }
        },
    )

    assert result.designation == "60"
    assert result.exact_club == "60° Wedge"
    assert result.source == "rapidocr_tiled"
    assert calls[:2] == [((240, 160), (0,)), ((240, 160), (90, 270))]
    assert any(size == (120, 80) and rotations == (0,) for size, rotations in calls)


def test_no_marking_or_ocr_runtime_does_not_crash(monkeypatch):
    monkeypatch.setattr(
        club_recognition,
        "read_marking_candidates",
        lambda image, config, rotations=(0,): _ocr_result(source="ocr_unavailable", reasoning="OCR dependency is missing."),
    )

    result = club_recognition.classify_club_marking(Image.new("RGB", (80, 60)), "Iron", {"club_recognition": {}})

    assert result.exact_club is None
    assert result.source == "ocr_unavailable"
    assert result.reasoning == "OCR dependency is missing."


def test_five_way_club_type_checkpoint_task_is_supported():
    assert "club_type_5way" in SUPPORTED_TASKS


def test_five_way_driver_keeps_current_behavior_and_skips_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr(club_recognition, "classify_five_way_club_type", lambda image, config: _five_way_result("Driver", 0.91))
    monkeypatch.setattr(club_recognition, "read_marking_candidates", lambda *args, **kwargs: pytest.fail("OCR must not run"))

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["status"] == "confirmed"
    assert result["club_type"] == "Driver"
    assert result["predicted_club"] == "Driver"
    assert result["ocr"] is None


def test_uncertain_five_way_prediction_yields_to_a_visible_club_loft(monkeypatch, tmp_path):
    """A full live frame can confuse the broad model while OCR still sees the club."""

    monkeypatch.setattr(club_recognition, "classify_five_way_club_type", lambda image, config: _five_way_result("Driver", 0.46))
    monkeypatch.setattr(
        club_recognition,
        "read_marking_candidates",
        lambda image, config, rotations=(0,): _ocr_result(OcrCandidate("60", 0.996)),
    )

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["status"] == "confirmed"
    assert result["club_type"] == "Wedge"
    assert result["club_number"] == "60"
    assert result["exact_club"].endswith("Wedge")
    assert result["confidence"] == 0.996
    assert result["sources"]["club_marking_ocr"] == "rapidocr"


def test_five_way_iron_returns_complete_exact_club_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(club_recognition, "classify_five_way_club_type", lambda image, config: _five_way_result("Iron"))
    monkeypatch.setattr(
        club_recognition,
        "read_marking_candidates",
        lambda image, config, rotations=(0,): _ocr_result(OcrCandidate("8", 0.95)),
    )

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["club_type"] == "Iron"
    assert result["club_number"] == "8"
    assert result["exact_club"] == "8 Iron"


def test_five_way_checkpoint_is_automatically_resolved_for_the_runtime(tmp_path):
    from backend.services.model_manager import configure_five_way_club_checkpoint

    trained_models = tmp_path / "models" / "trained"
    trained_models.mkdir(parents=True)
    checkpoint = trained_models / "club_type_5way.pt"
    checkpoint.write_bytes(b"checkpoint")

    config = {"club_recognition": {}}
    resolved = configure_five_way_club_checkpoint(config, tmp_path, trained_models)

    assert resolved == checkpoint.resolve()
    assert config["club_recognition"]["five_way_cnn_model_path"] == str(checkpoint.resolve())
