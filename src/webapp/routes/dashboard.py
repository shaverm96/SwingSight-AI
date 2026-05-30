from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, render_template, request, send_from_directory

from webapp.services.analysis_service import AnalysisContext, AnalysisService
from webapp.utils.storage import save_filestorage


dashboard_bp = Blueprint("dashboard", __name__)
analysis_service: AnalysisService | None = None
uploaded_files: dict[str, str] = {}


def _service() -> AnalysisService:
    global analysis_service
    if analysis_service is None:
        config = current_app.config["SWINGSIGHT_CONFIG"]
        analysis_service = AnalysisService(config)
    return analysis_service


@dashboard_bp.get("/")
def home() -> str:
    return render_template("dashboard.html")


@dashboard_bp.post("/api/upload-video")
def upload_video() -> Response:
    file_obj = request.files.get("video")
    if file_obj is None:
        return jsonify({"error": "video file is required"}), 400

    service = _service()
    file_id, file_path = save_filestorage(file_obj, service.uploads_dir, "video")
    uploaded_files[file_id] = file_path

    return jsonify(
        {
            "upload_id": file_id,
            "file_name": Path(file_path).name,
            "preview_url": f"/uploads/{Path(file_path).name}",
        }
    )


@dashboard_bp.post("/api/upload-club-image")
def upload_club_image() -> Response:
    file_obj = request.files.get("image")
    if file_obj is None:
        return jsonify({"error": "club image file is required"}), 400

    service = _service()
    file_id, file_path = save_filestorage(file_obj, service.uploads_dir, "club")
    uploaded_files[file_id] = file_path

    return jsonify(
        {
            "upload_id": file_id,
            "file_name": Path(file_path).name,
            "preview_url": f"/uploads/{Path(file_path).name}",
        }
    )


@dashboard_bp.post("/api/analyze")
def analyze_swing() -> Response:
    payload = request.get_json(silent=True) or {}
    video_upload_id = payload.get("video_upload_id")
    club_image_upload_id = payload.get("club_image_upload_id")
    club_category = payload.get("club_category")

    if not video_upload_id:
        return jsonify({"error": "video_upload_id is required"}), 400

    video_path = uploaded_files.get(video_upload_id)
    if not video_path:
        return jsonify({"error": "video upload not found"}), 404

    club_image_path = uploaded_files.get(club_image_upload_id) if club_image_upload_id else None

    result = _service().run_analysis(
        AnalysisContext(
            video_path=video_path,
            club_image_path=club_image_path,
            manual_club_category=club_category,
        )
    )
    return jsonify(result)


@dashboard_bp.get("/api/results/<analysis_id>")
def get_results(analysis_id: str) -> Response:
    result = _service().get_result(analysis_id)
    if result is None:
        return jsonify({"error": "analysis_id not found"}), 404
    return jsonify(result)


@dashboard_bp.post("/api/reports/<analysis_id>")
def create_report(analysis_id: str) -> Response:
    payload = request.get_json(silent=True) or {}
    report_format = payload.get("format", "pdf")
    report_path = _service().generate_report(analysis_id, report_format)

    if report_path is None:
        return jsonify({"error": "unable to generate report"}), 400

    report_file = Path(report_path)
    return jsonify(
        {
            "report_path": str(report_file),
            "download_url": f"/reports/{report_file.name}",
            "format": report_format,
        }
    )


@dashboard_bp.get("/uploads/<path:filename>")
def uploaded_file(filename: str) -> Response:
    service = _service()
    return send_from_directory(service.uploads_dir, filename)


@dashboard_bp.get("/outputs/<path:filename>")
def output_file(filename: str) -> Response:
    service = _service()
    return send_from_directory(service.outputs_dir, filename)


@dashboard_bp.get("/reports/<path:filename>")
def report_file(filename: str) -> Response:
    service = _service()
    return send_from_directory(service.reports_dir, filename, as_attachment=True)
