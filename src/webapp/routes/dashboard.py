from __future__ import annotations

from pathlib import Path
import mimetypes

from flask import Blueprint, Response, current_app, jsonify, render_template, request, send_file, send_from_directory

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



@dashboard_bp.post("/api/club-detect")
def club_detect() -> Response:
    """Receive a single camera frame and run club detection/classification."""
    frame = request.files.get("frame")
    if frame is None:
        return jsonify({"error": "frame is required"}), 400

    service = _service()
    file_id, file_path = save_filestorage(frame, service.uploads_dir, "frame")
    runtime = current_app.extensions.get("swing_runtime", {})
    model_manager = runtime.get("model_manager")
    if model_manager is None:
        return jsonify({"error": "model manager is unavailable"}), 503
    result = model_manager.detect_club(file_path)
    return jsonify({"upload_id": file_id, "file_name": Path(file_path).name, "result": result})


@dashboard_bp.post("/api/body-check")
def body_check() -> Response:
    """Receive a single camera frame and check whether the key body landmarks are visible."""
    frame = request.files.get("frame")
    if frame is None:
        return jsonify({"error": "frame is required"}), 400

    service = _service()
    file_id, file_path = save_filestorage(frame, service.uploads_dir, "frame")
    runtime = current_app.extensions.get("swing_runtime", {})
    model_manager = runtime.get("model_manager")
    if model_manager is None:
        return jsonify({"error": "model manager is unavailable"}), 503
    check = model_manager.check_body_visibility(file_path)
    return jsonify({"upload_id": file_id, "file_name": Path(file_path).name, "check": check})


@dashboard_bp.post("/api/record-swing")
def record_swing() -> Response:
    """Accept a recorded swing video blob and save it for analysis."""
    file_obj = request.files.get("video")
    if file_obj is None:
        return jsonify({"error": "video file is required"}), 400

    service = _service()
    file_id, file_path = save_filestorage(file_obj, service.uploads_dir, "recorded")
    uploaded_files[file_id] = file_path
    return jsonify({"upload_id": file_id, "file_name": Path(file_path).name, "preview_url": f"/uploads/{Path(file_path).name}"})


@dashboard_bp.post("/api/analyze-swing")
def analyze_swing_api() -> Response:
    payload = request.get_json(silent=True) or {}
    video_upload_id = payload.get("video_upload_id")
    club_category = payload.get("club_category")

    if not video_upload_id:
        return jsonify({"error": "video_upload_id is required"}), 400

    video_path = uploaded_files.get(video_upload_id)
    if not video_path:
        return jsonify({"error": "video upload not found"}), 404

    result = _service().run_analysis(
        AnalysisContext(video_path=video_path, club_image_path=None, manual_club_category=club_category)
    )
    status_code = 200 if result.get("status") == "success" else 422
    return jsonify(result), status_code


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
    status_code = 200 if result.get("status") == "success" else 422
    return jsonify(result), status_code


@dashboard_bp.get("/api/results/<analysis_id>")
def get_results(analysis_id: str) -> Response:
    result = _service().get_result(analysis_id)
    if result is None:
        return jsonify({"error": "analysis_id not found"}), 404
    return jsonify(result)


@dashboard_bp.get("/api/artifacts/<analysis_id>")
def get_artifacts(analysis_id: str) -> Response:
    result = _service().get_result(analysis_id)
    if result is None:
        return jsonify({"error": "analysis_id not found"}), 404

    return jsonify(
        {
            "analysis_id": analysis_id,
            "overlay_files": result.get("overlay_files", []),
            "advanced_metrics": result.get("advanced_metrics", {}),
        }
    )


@dashboard_bp.get("/api/analysis-assets/<analysis_id>")
def analysis_assets(analysis_id: str) -> Response:
    result = _service().get_result(analysis_id)
    if result is None:
        return jsonify({"error": "analysis_id not found"}), 404
    return jsonify(
        {
            "analysis_id": analysis_id,
            "overlay_files": result.get("overlay_files", []),
            "advanced_metrics": result.get("advanced_metrics", {}),
            "summary": result.get("summary", {}),
        }
    )


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
    file_path = (Path(service.uploads_dir) / filename).resolve()
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "uploaded file not found"}), 404
    mimetype, _ = mimetypes.guess_type(file_path.name)
    current_app.logger.info("API SERVING upload file path=%s mimetype=%s", file_path, mimetype or "application/octet-stream")
    return send_file(file_path, mimetype=mimetype or "application/octet-stream", conditional=True)


@dashboard_bp.get("/outputs/<path:filename>")
def output_file(filename: str) -> Response:
    service = _service()
    file_path = (Path(service.outputs_dir) / filename).resolve()
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "output file not found"}), 404
    current_app.logger.info("API SERVING overlay file path=%s mimetype=%s", file_path, "video/mp4")
    return send_file(file_path, mimetype="video/mp4", conditional=True)


@dashboard_bp.get("/reports/<path:filename>")
def report_file(filename: str) -> Response:
    service = _service()
    return send_from_directory(service.reports_dir, filename, as_attachment=True)
