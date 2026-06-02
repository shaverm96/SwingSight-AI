from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class CoachingResult:
    swing_score: int
    strengths: List[str]
    improvements: List[str]
    next_focus: str


class CoachingEngine:
    """Convert technical model outputs into golfer-friendly coaching feedback."""

    def build_feedback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        swing_score = int(round(self._score(inputs)))
        strengths = self._strengths(inputs)
        improvements = self._improvements(inputs)
        next_focus = self._next_focus(inputs, improvements)
        return {
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
        }

    def _score(self, inputs: Dict[str, Any]) -> float:
        score = 72.0
        score += self._score_from_metric(inputs.get("head_stability"), 18.0, good_high=True)
        score += self._score_from_metric(inputs.get("hip_rotation"), 15.0, good_high=True)
        score += self._score_from_metric(inputs.get("shoulder_rotation"), 12.0, good_high=True)
        score += self._score_from_metric(inputs.get("spine_angle"), 10.0, good_high=False, ideal=18.0)
        score += self._club_path_adjustment(inputs.get("club_path"))
        score += self._weight_shift_adjustment(inputs.get("weight_shift"))
        return max(0.0, min(100.0, score))

    @staticmethod
    def _score_from_metric(value: Any, weight: float, *, good_high: bool, ideal: float | None = None) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if ideal is not None:
            return max(0.0, weight - abs(numeric - ideal) * 1.5)
        if good_high:
            return min(weight, numeric / 100.0 * weight)
        return max(0.0, weight - abs(numeric - 18.0) * 0.5)

    @staticmethod
    def _club_path_adjustment(value: Any) -> float:
        if not value:
            return 0.0
        text = str(value).lower()
        if text in {"neutral", "square", "in-to-out"}:
            return 8.0
        if text == "outside_in":
            return -8.0
        return 0.0

    @staticmethod
    def _weight_shift_adjustment(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(-5.0, min(5.0, (numeric - 50.0) / 10.0))

    def _strengths(self, inputs: Dict[str, Any]) -> List[str]:
        strengths: List[str] = []
        if self._metric_high(inputs.get("head_stability"), 80):
            strengths.append("Good balance throughout the swing")
        if self._metric_high(inputs.get("hip_rotation"), 65):
            strengths.append("Solid hip turn")
        if self._metric_high(inputs.get("shoulder_rotation"), 65):
            strengths.append("Comfortable shoulder rotation")
        if not strengths:
            strengths.append("You made a controlled swing")
        return strengths[:3]

    def _improvements(self, inputs: Dict[str, Any]) -> List[str]:
        improvements: List[str] = []
        if self._club_path_is_outside_in(inputs.get("club_path")):
            improvements.append("Reduce your outside-in club path")
        if self._metric_low(inputs.get("hip_rotation"), 60):
            improvements.append("Improve hip rotation through impact")
        if self._metric_low(inputs.get("head_stability"), 70):
            improvements.append("Keep your head more stable through impact")
        if not improvements:
            improvements.append("Keep the same tempo and repeat this motion")
        return improvements[:3]

    @staticmethod
    def _metric_high(value: Any, threshold: float) -> bool:
        try:
            return float(value) >= threshold
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _metric_low(value: Any, threshold: float) -> bool:
        try:
            return float(value) < threshold
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _club_path_is_outside_in(value: Any) -> bool:
        if not value:
            return False
        return str(value).lower() == "outside_in"

    def _next_focus(self, inputs: Dict[str, Any], improvements: Iterable[str]) -> str:
        improvement_list = list(improvements)
        if improvement_list:
            first = improvement_list[0].lower()
            if "outside-in" in first:
                return "Work on shallowing the club during the downswing"
            if "hip rotation" in first:
                return "Turn your hips a little more freely through the strike"
            if "head" in first:
                return "Keep your head quieter as you swing through the ball"
        club_path = inputs.get("club_path")
        if club_path:
            return "Keep your path consistent and finish balanced"
        return "Keep your tempo smooth and repeat this swing"
