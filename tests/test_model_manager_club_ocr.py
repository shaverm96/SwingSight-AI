from __future__ import annotations

from backend.services.model_manager import ModelManager


def test_uploaded_club_image_replaces_video_frame_detection_with_exact_club():
    manager = ModelManager.__new__(ModelManager)
    manager.analyze_pose = lambda video_path: {
        "club": "Not detected",
        "club_detection": {"club": "Not detected"},
        "club_note": "Club recognition needs a clearer view of the club face or sole.",
        "warnings": [],
    }
    manager.detect_club = lambda image_path: {
        "club": "7 Iron",
        "detected_club": "7 Iron",
        "club_type": "Iron",
        "club_number": "7",
        "exact_club": "7 Iron",
        "status": "confirmed",
        "raw": {"club_type": "Iron", "club_number": "7", "exact_club": "7 Iron"},
    }
    manager._club_detection_note = lambda detection: None
    manager._fallback_club_note = lambda: "Club recognition needs a clearer view of the club face or sole."

    analysis = manager.analyze_swing("swing.mp4", club_image_path="club.jpg")

    assert analysis["club"] == "7 Iron"
    assert analysis["club_detection"]["club_type"] == "Iron"
    assert analysis["club_detection"]["club_number"] == "7"
    assert analysis["club_detection"]["exact_club"] == "7 Iron"
