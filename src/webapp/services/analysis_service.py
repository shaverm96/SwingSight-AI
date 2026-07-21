from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from flask import current_app

from webapp.services.gemini_coaching_service import GeminiCoachingService
from webapp.services.report_service import generate_pdf_report, generate_word_report
from webapp.utils.storage import ensure_dir


LOGGER = logging.getLogger(__name__)


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
        self.gemini_coaching = GeminiCoachingService(self.config.get("gemini", {}))

    def run_analysis(self, context: AnalysisContext) -> Dict:
        analysis_id = uuid4().hex
        runtime = self._runtime()
        model_manager = runtime.get("model_manager")
        coaching_engine = runtime.get("coaching_engine")

        if model_manager is None:
            result = {
                "status": "error",
                "analysis_id": analysis_id,
                "video_processed": False,
                "message": "Model manager is unavailable.",
            }
            self._results[analysis_id] = result
            self._persist_result(analysis_id, result)
            return result

        if coaching_engine is None:
            result = {
                "status": "error",
                "analysis_id": analysis_id,
                "video_processed": False,
                "message": "Coaching engine is unavailable.",
            }
            self._results[analysis_id] = result
            self._persist_result(analysis_id, result)
            return result

        try:
            analysis = model_manager.analyze_swing(
                context.video_path,
                club_image_path=context.club_image_path,
                manual_club_category=context.manual_club_category,
            )
        except ValueError as exc:
            result = {
                "status": "error",
                "analysis_id": analysis_id,
                "video_processed": False,
                "message": str(exc),
            }
            self._results[analysis_id] = result
            self._persist_result(analysis_id, result)
            return result
        except Exception as exc:
            LOGGER.exception("Unexpected analysis failure for %s", context.video_path)
            result = {
                "status": "error",
                "analysis_id": analysis_id,
                "video_processed": False,
                "message": "Unable to analyze the uploaded video. Please try a clearer clip.",
                "debug": {"error_type": type(exc).__name__},
            }
            self._results[analysis_id] = result
            self._persist_result(analysis_id, result)
            return result

        technical_metrics = analysis.get("advanced_metrics", {}) or {}
        coaching = coaching_engine.coach(
            technical_metrics,
            club=analysis.get("club"),
            video_processed=analysis.get("video_processed", False),
            fallback_used=analysis.get("fallback_used", False),
            club_note=analysis.get("club_note"),
        )

        gemini_result = self.gemini_coaching.coach(analysis, coaching)
        gemini_coaching = gemini_result.get("coaching") or {}
        if gemini_coaching.get("next_focus"):
            coaching["next_focus"] = gemini_coaching["next_focus"]
        if gemini_coaching.get("overall_score") is not None:
            coaching["swing_score"] = gemini_coaching["overall_score"]
            coaching["score_source"] = "gemini"
            coaching["score_rationale"] = gemini_coaching.get("score_rationale")
        else:
            coaching["score_source"] = "local_cv"

        swing_score = coaching.get("swing_score")
        score_label = self._score_label(swing_score, analysis.get("video_processed", False), analysis.get("tracking", {}))
        club_name = analysis.get("club", "Not detected") or "Not detected"
        club_note = analysis.get("club_note") or ("Club detection needs a closer view of the club head or club end." if club_name == "Not detected" else None)

        summary = {
            "analysis_id": analysis_id,
            "club": club_name,
            "swing_score": swing_score,
            "next_focus": coaching.get("next_focus", "Record from a clear side angle with your full body in frame."),
        }

        visualization = analysis.get("visualization", {}) or {}
        overlay_validation = analysis.get("overlay_validation", {}) or {}
        overlay_variants = analysis.get("overlay_variants", {}) or visualization.get("overlay_variants", {}) or {}

        result = {
            "status": "success",
            "analysis_id": analysis_id,
            "video_processed": bool(analysis.get("video_processed", False)),
            "club": club_name,
            "club_note": club_note,
            "swing_score": swing_score,
            "score_label": score_label,
            "score_source": coaching.get("score_source", "local_cv"),
            "score_rationale": coaching.get("score_rationale"),
            "strengths": coaching.get("strengths", []) or ["Your swing was uploaded successfully."],
            "improvements": coaching.get("improvements", []) or ["Record with your full body visible from setup through finish."],
            "next_focus": coaching.get("next_focus", "Record from a clear side angle with your full body in frame."),
            "advanced_metrics": technical_metrics,
            "tracking": analysis.get("tracking", {}),
            "model_outputs": analysis.get("model_outputs", {}),
            "gemini_analysis": gemini_coaching or None,
            "gemini": {key: value for key, value in gemini_result.items() if key != "coaching"},
            "overlay_files": analysis.get("overlay_files", []),
            "overlay_validation": overlay_validation,
            "overlay_variants": overlay_variants,
            "visualization": visualization,
            "video_metadata": analysis.get("video_metadata", {}),
            "warnings": analysis.get("warnings", []),
            "fallback_used": bool(analysis.get("fallback_used", False)),
            "debug": analysis.get("debug", {}),
            "summary": summary,
            "advanced": technical_metrics,
            "detected_club": club_name,
            "input_video": context.video_path,
        }

        self._results[analysis_id] = result
        self._persist_result(analysis_id, result)
        return result

    def _score_label(self, swing_score: Optional[int], video_processed: bool, tracking: Dict) -> str:
        pose_detected = int(
            tracking.get("pose_frames_detected", 0)
            or tracking.get("frames_with_pose", 0)
            or (tracking.get("quality_metrics", {}) or {}).get("frames_with_pose", 0)
            or 0
        )

        if swing_score is None:
            if not video_processed:
                return "Analysis incomplete"
            return "Review ready" if pose_detected > 0 else "Needs clearer video"

        if pose_detected == 0:
            return "Needs clearer video"

        if swing_score >= 85:
            return "Strong swing"
        if swing_score >= 70:
            return "Improving"
        if swing_score >= 50:
            return "Needs practice"
        return "Needs practice"

    def _runtime(self) -> Dict:
        try:
            return current_app.extensions.get("swing_runtime", {})
        except RuntimeError:
            return {}

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