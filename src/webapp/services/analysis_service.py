from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from webapp.inference.pipeline_components import (
    build_feedback,
    classify_club_category,
    estimate_pose,
    extract_swing_metrics,
    recognize_loft_or_number,
    render_outputs,
    run_club_detection,
    score_swing,
    track_body_landmarks,
)
from webapp.services.report_service import generate_pdf_report, generate_word_report
from webapp.utils.storage import ensure_dir


@dataclass
class AnalysisContext:
    video_path: str
    club_image_path: Optional[str]
    manual_club_category: Optional[str]


class AnalysisService:
    """Local-only orchestration for SwingSight dashboard analysis."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self._results: Dict[str, Dict] = {}

        paths = self.config.setdefault("paths", {})
        self.uploads_dir = ensure_dir(paths.get("uploads_dir", "uploads"))
        self.outputs_dir = ensure_dir(paths.get("outputs_dir", "outputs"))
        self.reports_dir = ensure_dir(paths.get("reports_dir", "reports"))

    def run_analysis(self, context: AnalysisContext) -> Dict:
        analysis_id = uuid4().hex

        club_detection = run_club_detection(context.club_image_path, self.config)
        club_classification = classify_club_category(
            context.club_image_path,
            context.manual_club_category or club_detection["category"],
            self.config,
        )
        loft_recognition = recognize_loft_or_number(context.club_image_path, self.config)

        final_club_category = context.manual_club_category or club_classification["category"]

        pose_frames = estimate_pose(context.video_path, self.config)
        body_tracking = track_body_landmarks(pose_frames)
        metrics = extract_swing_metrics(pose_frames, self.config)
        score = score_swing(metrics, final_club_category, self.config)
        feedback = build_feedback(metrics, final_club_category, self.config)
        rendered = render_outputs(context.video_path, pose_frames, str(self.outputs_dir))

        detected_club = None
        if isinstance(club_detection, dict):
            recognition = club_detection.get("recognition") or {}
            detected_club = recognition.get("predicted_club") or club_detection.get("category")

        summary = self._build_summary(detected_club, score, feedback)
        advanced = {
            "metrics": metrics,
            "pose_frame_count": len(pose_frames),
            "body_tracking": body_tracking,
            "model_outputs": {
                "club_detection": club_detection,
                "club_classification": club_classification,
                "loft_recognition": loft_recognition,
                "score": score,
            },
            "outputs": rendered,
        }

        result = {
            "analysis_id": analysis_id,
            "club_detection": club_detection,
            "club_classification": club_classification,
            "loft_recognition": loft_recognition,
            "final_club_category": final_club_category,
            "detected_club": detected_club,
            "pose_frame_count": len(pose_frames),
            "body_tracking": body_tracking,
            "metrics": metrics,
            "score": score,
            "feedback": feedback,
            "outputs": rendered,
            "input_video": context.video_path,
            "summary": summary,
            "advanced": advanced,
        }

        self._results[analysis_id] = result
        self._persist_result(analysis_id, result)
        return result

    @staticmethod
    def _build_summary(detected_club: Optional[str], score: Dict, feedback: object) -> Dict:
        overall_score = score.get("overall_score") if isinstance(score, dict) else None

        takeaways = []
        focus = None
        if isinstance(feedback, dict):
            key = feedback.get("key_suggestions") or []
            body = feedback.get("body_position") or []
            path = feedback.get("club_path") or []
            takeaways = [*key, *body, *path][:3]
            focus_candidates = [*body, *path, *key]
            focus = focus_candidates[0] if focus_candidates else None
        elif isinstance(feedback, list):
            takeaways = [str(item) for item in feedback[:3]]
            focus = str(feedback[0]) if feedback else None

        if not focus:
            focus = "Keep your tempo smooth and finish balanced."

        return {
            "overall_score": overall_score,
            "confirmed_club": detected_club or "Unknown",
            "takeaways": takeaways,
            "focus": focus,
        }

    def get_result(self, analysis_id: str) -> Optional[Dict]:
        if analysis_id in self._results:
            return self._results[analysis_id]

        file_path = self.outputs_dir / f"analysis_{analysis_id}.json"
        if not file_path.exists():
            return None

        with file_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        self._results[analysis_id] = result
        return result

    def generate_report(self, analysis_id: str, report_format: str) -> Optional[str]:
        result = self.get_result(analysis_id)
        if result is None:
            return None

        if report_format == "pdf":
            return generate_pdf_report(result, str(self.reports_dir))
        if report_format == "docx":
            return generate_word_report(result, str(self.reports_dir))
        return None

    def _persist_result(self, analysis_id: str, result: Dict) -> None:
        file_path = self.outputs_dir / f"analysis_{analysis_id}.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
