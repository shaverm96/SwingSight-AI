from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from flask import current_app

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
        runtime = self._runtime()
        model_manager = runtime.get("model_manager")

        if model_manager is not None:
            analysis = model_manager.analyze_swing(context.video_path)
            club = analysis.get("club", "Unknown")
            swing_score = analysis.get("swing_score", 0)
            strengths = analysis.get("strengths", [])
            improvements = analysis.get("improvements", [])
            next_focus = analysis.get("next_focus", "Keep your tempo smooth and repeat this swing")
            advanced_metrics = analysis.get("advanced_metrics", {})
            overlay_files = analysis.get("overlay_files", [])
        else:
            club = "Unknown"
            swing_score = 0
            strengths = ["You completed the swing"]
            improvements = ["Collect notebook artifacts to enable richer feedback"]
            next_focus = "Keep your tempo smooth and repeat this swing"
            advanced_metrics = {}
            overlay_files = []

        summary = {
            "analysis_id": analysis_id,
            "club": club,
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
        }

        result = {
            "analysis_id": analysis_id,
            "club": club,
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
            "advanced_metrics": advanced_metrics,
            "overlay_files": overlay_files,
            "summary": summary,
            "advanced": advanced_metrics,
            "detected_club": club,
            "input_video": context.video_path,
        }

        self._results[analysis_id] = result
        self._persist_result(analysis_id, result)
        return result

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