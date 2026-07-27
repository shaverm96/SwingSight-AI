from __future__ import annotations

from flask import Flask

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
        created = client.post("/api/reports/pdf-smoke", json={"format": "pdf"})

        assert created.status_code == 200
        payload = created.get_json()
        assert payload["download_url"] == "/reports/swing_report_pdf-smoke.pdf"
        assert reports_dir.joinpath("swing_report_pdf-smoke.pdf").is_file()

        downloaded = client.get(payload["download_url"])
        assert downloaded.status_code == 200
        assert downloaded.mimetype == "application/pdf"
        assert downloaded.data.startswith(b"%PDF-")
        assert "attachment" in downloaded.headers["Content-Disposition"]
    finally:
        dashboard.analysis_service = None
