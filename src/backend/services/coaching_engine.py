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

    def coach(
        self,
        technical_metrics: Dict[str, object],
        club: Optional[str] = None,
        *,
        video_processed: bool = False,
        fallback_used: bool = False,
        club_note: Optional[str] = None,
    ) -> Dict:
        metrics = technical_metrics or {}
        club_name = club or str(metrics.get("club", "Not detected"))

        swing_score = self._score(metrics, video_processed=video_processed, fallback_used=fallback_used)
        strengths, improvements = self._build_feedback(
            metrics,
            video_processed=video_processed,
            fallback_used=fallback_used,
            club_note=club_note,
        )
        next_focus = self._choose_next_focus(metrics, improvements, fallback_used=fallback_used)

        return {
            "club": club_name,
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
        }

    def _metric(self, metrics: Dict[str, object], *names: str) -> Optional[float]:
        for name in names:
            value = _to_float(metrics.get(name))
            if value is not None:
                return value
        return None

    def _score_component(self, name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None

        if name in {"balance_score", "head_movement_cm", "hip_turn_proxy", "shoulder_turn_proxy", "tempo_estimate"}:
            return _clamp(value)

        if name == "spine_angle_deg":
            return _clamp(100.0 - abs(value - 10.0) * 3.5)

        if name == "knee_flex_deg":
            return _clamp(100.0 - abs(value - 145.0) * 1.1)

        return None

    def _score(self, metrics: Dict[str, object], *, video_processed: bool, fallback_used: bool) -> Optional[int]:
        values: List[float] = []

        component_sources = [
            ("head_movement_cm", self._metric(metrics, "head_movement_cm", "head_stability")),
            ("hip_turn_proxy", self._metric(metrics, "hip_turn_proxy", "hip_rotation")),
            ("shoulder_turn_proxy", self._metric(metrics, "shoulder_turn_proxy", "shoulder_rotation")),
            ("balance_score", self._metric(metrics, "balance_score", "weight_shift")),
            ("tempo_estimate", self._metric(metrics, "tempo_estimate")),
            ("spine_angle_deg", self._metric(metrics, "spine_angle_deg", "spine_angle")),
            ("knee_flex_deg", self._metric(metrics, "knee_flex_deg")),
        ]

        for name, value in component_sources:
            component = self._score_component(name, value)
            if component is not None:
                values.append(component)

        if not values:
            return None if video_processed or fallback_used else 72

        completeness = len(values) / 5.0
        average = sum(values) / len(values)
        adjusted = average * (0.5 + 0.5 * completeness)
        if completeness < 0.5:
            adjusted = max(adjusted, 50.0)
        return int(round(max(0.0, min(100.0, adjusted))))

    def _build_feedback(
        self,
        metrics: Dict[str, object],
        *,
        video_processed: bool,
        fallback_used: bool,
        club_note: Optional[str],
    ) -> tuple[List[str], List[str]]:
        strengths: List[str] = []
        improvements: List[str] = []

        if video_processed and fallback_used:
            strengths.append("Your swing was uploaded successfully.")
            strengths.append("We were able to process the video preview.")

        if fallback_used:
            improvements.append("Try recording from a clear side angle with your full body visible.")
            improvements.append("Keep the camera still and avoid zooming.")
        else:
            tempo_estimate = self._metric(metrics, "tempo_estimate")
            if tempo_estimate is None or tempo_estimate < 70:
                improvements.append("Let the backswing and downswing share the same rhythm—no need to rush the good part.")

        head_stability = self._metric(metrics, "head_movement_cm", "head_stability")
        if head_stability is None:
            if not fallback_used:
                improvements.append("Keep your head nice and quiet through impact—let the club do the traveling.")
        elif head_stability < 80:
            improvements.append("Keep your head nice and quiet through impact—let the club do the traveling.")
        else:
            strengths.append("Nice quiet head through the swing—steady is plenty.")

        hip_rotation = self._metric(metrics, "hip_turn_proxy", "hip_rotation")
        if hip_rotation is None:
            if not fallback_used:
                improvements.append("Give those hips a little more turn through impact—let them join the party.")
        elif hip_rotation < 70:
            improvements.append("Give those hips a little more turn through impact—let them join the party.")
        else:
            strengths.append("Good hip turn through the swing—you're clearing space nicely.")

        shoulder_rotation = self._metric(metrics, "shoulder_turn_proxy", "shoulder_rotation")
        if shoulder_rotation is None:
            if not fallback_used:
                improvements.append("Let your shoulders make a fuller turn—no need to keep them on a short leash.")
        elif shoulder_rotation < 70:
            improvements.append("Let your shoulders make a fuller turn—no need to keep them on a short leash.")
        else:
            strengths.append("Strong shoulder turn—keep that coil working for you.")

        spine_angle = self._metric(metrics, "spine_angle_deg", "spine_angle")
        if spine_angle is None:
            if not fallback_used:
                improvements.append("Hold your posture a touch steadier—athletic, not stiff as a coat rack.")
        elif abs(spine_angle - 10.0) > 8.0:
            improvements.append("Hold your posture a touch steadier—athletic, not stiff as a coat rack.")
        else:
            strengths.append("Your setup looks balanced—an athletic place to start.")

        balance_score = self._metric(metrics, "balance_score", "weight_shift")
        if balance_score is not None and balance_score >= 70:
            strengths.append("Your lower body stays nicely balanced—good work keeping the floor under you.")

        tempo_estimate = self._metric(metrics, "tempo_estimate")
        if tempo_estimate is not None and tempo_estimate >= 70:
            strengths.append("Your tempo stays smooth—keep that easy rhythm rolling.")

        knee_flex = self._metric(metrics, "knee_flex_deg")
        if knee_flex is not None and knee_flex < 135:
            improvements.append("Give your knees a little more flex at setup—athletic, not waiting in a checkout line.")

        club_path = str(metrics.get("club_path", "")).lower()
        if club_path == "outside_in":
            improvements.append("Work the club a little less outside-in—let's keep it from cutting across the ball.")
        elif club_path == "inside_out":
            strengths.append("Nice inside-out path—keep swinging out toward your target.")

        if club_note and club_note not in improvements:
            improvements.append(club_note)

        if not strengths:
            strengths.append("You've got a solid base here—now let's give it a little polish.")

        if not improvements:
            improvements.append("Keep repeating that smooth motion—boring practice is where the good golf lives.")

        return strengths[:3], improvements[:3]

    def _choose_next_focus(self, metrics: Dict[str, object], improvements: Iterable[str], *, fallback_used: bool) -> str:
        if fallback_used:
            return "Give me a clear side view with your whole body in frame, and we'll have plenty to work with."

        club_path = str(metrics.get("club_path", "")).lower()
        if club_path == "outside_in":
            return "Let's shallow the club a touch in the downswing and give it room to approach from the inside."

        head_stability = self._metric(metrics, "head_movement_cm", "head_stability")
        if head_stability is None or head_stability < 80:
            return "Keep that head quiet through impact—let the club be the one making all the noise."

        improvement_list = list(improvements)
        if improvement_list:
            return improvement_list[0]

        return "Keep the rhythm smooth and finish in balance—you want to pose for the camera, not chase the ball."
