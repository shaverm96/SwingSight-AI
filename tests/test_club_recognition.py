from __future__ import annotations

from PIL import Image

import swingsight.club_recognition as club_recognition
from swingsight.club_cnn import SUPPORTED_TASKS


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


def test_five_way_club_type_checkpoint_task_is_supported():
    assert "club_type_5way" in SUPPORTED_TASKS


def test_five_way_model_is_preferred_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "detect_club_head",
        lambda image, config: club_recognition.ClubHeadDetection((0, 0, image.width, image.height), 0.8, "yolov8"),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_five_way_club_type",
        lambda image, config: club_recognition.ClubDetailResult(
            "Driver", 0.91, {"driver": 0.91, "wood": 0.03}, "five-way CNN selected driver", "cnn"
        ),
    )

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["status"] == "confirmed"
    assert result["detected_category"] == "Wood"
    assert result["predicted_club"] == "Driver"
    assert result["sources"]["club_type_5way_classifier"] == "cnn"
    assert "broad_classifier" not in result["sources"]


def test_five_way_label_normalization_uses_ui_labels():
    assert club_recognition.normalize_five_way_club_type("driver") == "Driver"
    assert club_recognition.normalize_five_way_club_type("fairway_wood") == "Wood"
    assert club_recognition.normalize_five_way_club_type("hybrid") == "Hybrid"
    assert club_recognition.normalize_five_way_club_type("iron") == "Iron"
    assert club_recognition.normalize_five_way_club_type("wedge") == "Wedge"


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


def test_five_way_iron_runs_the_separate_marking_cnn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        club_recognition,
        "detect_club_head",
        lambda image, config: club_recognition.ClubHeadDetection((0, 0, image.width, image.height), 0.8, "yolov8"),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_five_way_club_type",
        lambda image, config: club_recognition.ClubDetailResult(
            "Iron", 0.93, {"iron": 0.93}, "five-way CNN selected iron", "cnn"
        ),
    )
    monkeypatch.setattr(
        club_recognition,
        "classify_club_marking",
        lambda image, family, config: club_recognition.ClubDetailResult(
            "7 Iron", 0.95, {"7": 0.95}, "marking CNN selected 7", "cnn"
        ),
    )

    result = club_recognition.recognize_club_from_frame(
        str(_image_path(tmp_path)),
        {"club_recognition": {"five_way_cnn_model_path": "models/trained/club_type_5way.pt", "confirm_threshold": 0.6}},
    )

    assert result["status"] == "confirmed"
    assert result["predicted_club"] == "7 Iron"
    assert result["ocr"]["text"] == "7"
    assert result["sources"]["club_marking_classifier"] == "cnn"


def test_club_marking_labels_support_pitching_wedges_and_lofts():
    assert club_recognition.normalize_club_marking("P", "Wedge") == "Pitching Wedge"
    assert club_recognition.normalize_club_marking("56", "Wedge") == "56° Wedge"
    assert club_recognition.normalize_club_marking("7", "Iron") == "7 Iron"
    assert club_recognition.normalize_club_marking("7", "Wedge") is None
