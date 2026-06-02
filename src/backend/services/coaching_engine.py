from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except Exception:
        return None


@dataclass(frozen=True)
class CoachingResult:
    club: str
    swing_score: int
    strengths: List[str]
    improvements: List[str]
    next_focus: str


class CoachingEngine:
    """Convert technical swing metrics into coach-friendly feedback."""

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}

    def coach(self, technical_metrics: Dict[str, object], club: Optional[str] = None) -> Dict:
        metrics = technical_metrics or {}
        club_name = club or str(metrics.get("club", "Unknown"))

        swing_score = self._score(metrics)
        strengths, improvements = self._build_feedback(metrics)
        next_focus = self._choose_next_focus(metrics, improvements)

        return {
            "club": club_name,
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
        }

    def _score(self, metrics: Dict[str, object]) -> int:
        values: List[float] = []

        head_stability = _to_float(metrics.get("head_stability"))
        if head_stability is not None:
            values.append(_clamp(head_stability))

        hip_rotation = _to_float(metrics.get("hip_rotation"))
        if hip_rotation is not None:
            values.append(_clamp(hip_rotation))

        shoulder_rotation = _to_float(metrics.get("shoulder_rotation"))
        if shoulder_rotation is not None:
            values.append(_clamp(shoulder_rotation))

        weight_shift = _to_float(metrics.get("weight_shift"))
        if weight_shift is not None:
            values.append(_clamp(weight_shift))

        spine_angle = _to_float(metrics.get("spine_angle"))
        if spine_angle is not None:
            values.append(_clamp(100.0 - abs(spine_angle - 18.0) * 2.5))

        if not values:
            return 72

        return int(round(sum(values) / len(values)))

    def _build_feedback(self, metrics: Dict[str, object]) -> tuple[List[str], List[str]]:
        strengths: List[str] = []
        improvements: List[str] = []

        head_stability = _to_float(metrics.get("head_stability"))
        if head_stability is None or head_stability < 80:
            improvements.append("Try keeping your head more stable through impact")
        else:
            strengths.append("Consistent head position")

        hip_rotation = _to_float(metrics.get("hip_rotation"))
        if hip_rotation is None or hip_rotation < 70:
            improvements.append("Improve hip rotation through impact")
        else:
            strengths.append("Good hip turn through the swing")

        shoulder_rotation = _to_float(metrics.get("shoulder_rotation"))
        if shoulder_rotation is None or shoulder_rotation < 70:
            improvements.append("Let your shoulders turn a little more freely")
        else:
            strengths.append("Strong shoulder turn")

        spine_angle = _to_float(metrics.get("spine_angle"))
        if spine_angle is None or abs(spine_angle - 18.0) > 8.0:
            improvements.append("Keep your posture more steady")
        else:
            strengths.append("Your setup looks balanced")

        club_path = str(metrics.get("club_path", "")).lower()
        if club_path == "outside_in":
            improvements.append("Reduce your outside-in club path")
        elif club_path == "inside_out":
            strengths.append("Nice inside-out path")

        if not strengths:
            strengths.append("Solid swing foundation")

        if not improvements:
            improvements.append("Keep repeating the same smooth motion")

        return strengths[:3], improvements[:3]

    def _choose_next_focus(self, metrics: Dict[str, object], improvements: Iterable[str]) -> str:
        club_path = str(metrics.get("club_path", "")).lower()
        if club_path == "outside_in":
            return "Work on shallowing the club during the downswing"

        head_stability = _to_float(metrics.get("head_stability"))
        if head_stability is None or head_stability < 80:
            return "Keep your head steady through impact"

        improvement_list = list(improvements)
        if improvement_list:
            return improvement_list[0]

        return "Keep your tempo smooth and finish balanced"
