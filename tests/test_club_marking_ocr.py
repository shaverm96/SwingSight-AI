from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

import swingsight.club_marking_ocr as marking_ocr


def test_rapidocr_output_is_converted_to_confident_text_candidates(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            assert image.ndim == 3
            return SimpleNamespace(
                txts=("SW", "TITLEIST"),
                scores=(0.94, 0.88),
                boxes=(((1, 2), (3, 2), (3, 4), (1, 4)), ((5, 6), (7, 6), (7, 8), (5, 8))),
            )

    engine = FakeEngine()
    monkeypatch.setattr(marking_ocr, "_get_rapidocr_engine", lambda: (engine, None))

    result = marking_ocr.read_marking_candidates(Image.new("RGB", (80, 60)), {"club_recognition": {}}, rotations=(0,))

    assert result.source == "rapidocr"
    assert result.attempts == 1
    assert engine.calls == 1
    assert [(item.text, item.confidence) for item in result.candidates] == [("SW", 0.94), ("TITLEIST", 0.88)]
    assert result.candidates[0].bbox == [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]


def test_ocr_runtime_error_becomes_a_safe_failure_result(monkeypatch):
    class BrokenEngine:
        def __call__(self, image):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(marking_ocr, "_get_rapidocr_engine", lambda: (BrokenEngine(), None))

    result = marking_ocr.read_marking_candidates(Image.new("RGB", (80, 60)), {"club_recognition": {}}, rotations=(0,))

    assert result.candidates == ()
    assert result.source == "ocr_inference_error"
    assert "RuntimeError" in (result.reasoning or "")


def test_large_images_are_downscaled_without_manual_marking_crop():
    prepared = marking_ocr._prepare_image(
        Image.new("RGB", (4000, 2000)),
        {"max_image_side": 1000, "enhance_contrast": True},
    )

    assert prepared.shape[:2] == (500, 1000)
