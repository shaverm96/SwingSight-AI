from __future__ import annotations

from flask import Flask

from webapp.services.report_service import (
    _measurement_rows,
    build_report_filename,
    format_degrees,
    format_pixels,
    format_ratio,
    is_valid_metric,
)

from webapp.routes import dashboard


def test_pdf_report_can_be_downloaded_after_generation(tmp_path, monkeypatch):
    """The generated PDF must be served from the same directory it is written to."""
    reports_dir = tmp_path / "reports"
    monkeypatch.chdir(tmp_path)
    app = Flask(__name__)
    # A relative path is the normal configuration and previously made the
    # POST succeed while the returned download URL responded with a 404.
    app.config["SWINGSIGHT_CONFIG"] = {"paths": {"reports_dir": "reports"}}
    app.register_blueprint(dashboard.dashboard_bp)

    monkeypatch.setattr(dashboard, "analysis_service", None)
    try:
        with app.app_context():
            service = dashboard._service()
            assert service.reports_dir == reports_dir
            service._results["pdf-smoke"] = {
                "analysis_id": "pdf-smoke",
                "summary": {
                    "club": "8-Iron",
                    "swing_score": 78,
                    "next_focus": "Keep a smooth tempo through impact.",
                },
                "strengths": ["Balanced finish."],
                "improvements": ["Turn through the shot."],
                "advanced_metrics": {"spine_angle_deg": 11.7},
            }

        client = app.test_client()
        created = client.post("/api/reports/pdf-smoke")

        assert created.status_code == 200
        payload = created.get_json()
        assert payload == {
            "download_url": "/reports/SwingSight_8-Iron_Swing_Report.pdf",
            "format": "pdf",
        }
        assert reports_dir.joinpath("SwingSight_8-Iron_Swing_Report.pdf").is_file()

        downloaded = client.get(payload["download_url"])
        assert downloaded.status_code == 200
        assert downloaded.mimetype == "application/pdf"
        assert downloaded.data.startswith(b"%PDF-")
        assert "attachment" in downloaded.headers["Content-Disposition"]
        assert b"/Count 2" in downloaded.data
    finally:
        dashboard.analysis_service = None


def test_report_filename_uses_the_club_or_a_safe_fallback():
    assert build_report_filename("8 Iron") == "SwingSight_8-Iron_Swing_Report.pdf"
    assert build_report_filename("60 Wedge") == "SwingSight_60-Wedge_Swing_Report.pdf"
    assert build_report_filename("Not detected") == "SwingSight_Swing_Report.pdf"


def test_report_metric_formatting_hides_invalid_values_and_rounds_measurements():
    assert not is_valid_metric(None)
    assert not is_valid_metric(float("nan"))
    assert not is_valid_metric(float("inf"))
    assert format_degrees(-85.37469103038005) == "85.4°"
    assert format_pixels(2386.8164038074756) == "2,387 px"
    assert format_ratio(0.7007416822437862) == "0.70"

    rows = _measurement_rows(
        {
            "spine_angle_deg": -85.37469103038005,
            "head_movement_px": 2386.8164038074756,
            "lateral_shift_ratio": 0.7007416822437862,
            "tempo_estimate": 0.0,
        }
    )
    assert rows == [
        ("Spine Angle", "85.4°"),
        ("Head Movement", "2,387 px"),
        ("Lateral Shift", "0.70"),
    ]
