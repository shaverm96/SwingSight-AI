from __future__ import annotations

from PIL import Image

import swingsight.club_recognition as club_recognition


def _image_path(tmp_path):
    path = tmp_path / "club.jpg"
    Image.new("RGB", (160, 120), color=(30, 80, 120)).save(path)
    return path


def test_wood_branch_uses_the_wood_type_cnn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "detect_club_head",
        lambda image, config: club_recognition.ClubHeadDetection((0, 0, image.width, image.height), 0.8, "yolov8"),
    )
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

    result = club_recognition.recognize_club_from_frame(str(_image_path(tmp_path)), {"club_recognition": {"confirm_threshold": 0.6}})

    assert result["status"] == "confirmed"
    assert result["detected_category"] == "Wood"
    assert result["predicted_club"] == "Hybrid"
    assert result["sources"]["wood_type_classifier"] == "cnn"
    assert "iron_number_classifier" not in result["sources"]


def test_iron_branch_uses_the_marking_cnn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "detect_club_head",
        lambda image, config: club_recognition.ClubHeadDetection((0, 0, image.width, image.height), 0.9, "yolov8"),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_broad_category",
        lambda image, config: club_recognition.BroadCategoryResult(
            "iron", 0.94, {"iron": 0.94, "wood": 0.06}, "broad CNN selected iron", "cnn"
        ),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_iron_number",
        lambda image, config: club_recognition.ClubDetailResult(
            "7 Iron", 0.91, {"6": 0.03, "7": 0.91, "8": 0.06}, "number CNN selected 7", "cnn"
        ),
    )

    result = club_recognition.recognize_club_from_frame(str(_image_path(tmp_path)), {"club_recognition": {"confirm_threshold": 0.6}})

    assert result["status"] == "confirmed"
    assert result["detected_category"] == "Iron"
    assert result["predicted_club"] == "7 Iron"
    assert result["ocr"] == {"text": "7", "confidence": 0.91, "source": "cnn"}
    assert result["sources"]["iron_number_classifier"] == "cnn"
    assert "wood_type_classifier" not in result["sources"]


def test_missing_stage_checkpoint_is_unavailable():
    result = club_recognition.classify_broad_category(Image.new("RGB", (100, 100)), {"club_recognition": {}})

    assert result.category is None
    assert result.source in {"cnn_missing", "cnn_unavailable"}


def test_label_normalization_accepts_checkpoint_label_variants():
    assert club_recognition.normalize_iron_number("7 Iron") == 7
    assert club_recognition.normalize_iron_number("iron_8") == 8
    assert club_recognition.normalize_iron_number("10") is None
    assert club_recognition.normalize_wood_type("fairway_wood") == "Wood"
    assert club_recognition.normalize_wood_type("hybrid") == "Hybrid"
