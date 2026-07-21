from __future__ import annotations

from webapp.services.gemini_coaching_service import GeminiCoachingService, build_gemini_evidence, validate_coaching


def test_gemini_evidence_uses_only_computed_measurements():
    evidence = build_gemini_evidence(
        {
            "club": "Driver",
            "advanced_metrics": {
                "hip_rotation": 78.0,
                "club_speed_mph": 102.0,
                "impact_frame": 42,
                "ball_speed_mph": 149.0,
                "video_path": "/private/swing.mov",
            },
            "tracking": {"quality_metrics": {"frames_with_pose": 120}},
            "club_detection": {"status": "confirmed", "confidence": 0.93},
            "debug": {"file_paths": {"video_path": "/private/swing.mov"}},
        },
        {"strengths": ["Balanced finish"], "improvements": ["Keep posture"], "next_focus": "Tempo"},
    )

    assert evidence["club"]["label"] == "Driver"
    assert evidence["body_movement"]["hip_rotation"] == 78.0
    assert evidence["speed"]["club_speed_mph"] == 102.0
    assert evidence["impact_and_ball"]["ball_speed_mph"] == 149.0
    assert "/private/swing.mov" not in str(evidence)


def test_gemini_response_validation_limits_coaching_lists():
    coaching = validate_coaching(
        {
            "summary": "A measured summary.",
            "overall_score": 82.7,
            "score_rationale": "Measured body motion was consistent.",
            "next_focus": "Finish balanced.",
            "strengths": ["One", "Two", "Three", "Four"],
            "improvements": ["One"],
            "tips": ["One"],
            "drills": [{"title": "Pause drill", "description": "Pause at the top."}],
            "data_gaps": ["Impact was not measured."],
        }
    )

    assert coaching is not None
    assert coaching["strengths"] == ["One", "Two", "Three"]
    assert coaching["overall_score"] == 83
    assert coaching["score_rationale"] == "Measured body motion was consistent."
    assert coaching["drills"][0]["title"] == "Pause drill"


def test_gemini_skips_network_without_a_runtime_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = GeminiCoachingService({"enabled": True, "api_key_env": "GEMINI_API_KEY"})

    result = service.coach({"advanced_metrics": {}, "tracking": {}}, {"strengths": [], "improvements": [], "next_focus": "Tempo"})

    assert result["status"] == "not_configured"
    assert result["coaching"] is None
