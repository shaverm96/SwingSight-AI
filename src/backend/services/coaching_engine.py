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
        coach_observations = self._build_coach_observations(
            metrics,
            strengths=strengths,
            improvements=improvements,
            next_focus=next_focus,
            fallback_used=fallback_used,
        )
        coach_summary = self._build_coach_summary(coach_observations, fallback_used=fallback_used)

        return {
            "club": club_name,
            "swing_score": swing_score,
            "strengths": strengths,
            "improvements": improvements,
            "next_focus": next_focus,
            "coach_observations": coach_observations,
            "coach_summary": coach_summary,
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

    def _build_coach_observations(
        self,
        metrics: Dict[str, object],
        *,
        strengths: Iterable[str],
        improvements: Iterable[str],
        next_focus: str,
        fallback_used: bool,
    ) -> List[Dict[str, str]]:
        """Create the three most important, player-facing coaching conversations."""

        if fallback_used:
            return [
                {
                    "title": "Give me a clean side-on view",
                    "description": "Put your whole body in frame from setup through finish. That gives us a real swing to coach instead of asking the camera to play detective.",
                },
                {
                    "title": "Keep the camera still",
                    "description": "Set the phone down and leave the zoom alone. A steady picture lets us follow your setup, turn, and finish without guessing.",
                },
                {
                    "title": "Make your normal swing",
                    "description": "No need to manufacture a perfect one for the camera. Give us your usual motion and we can start finding the moves that will actually help on the course.",
                },
            ]

        observations: List[Dict[str, str]] = []
        head_stability = self._metric(metrics, "head_movement_cm", "head_stability")
        hip_rotation = self._metric(metrics, "hip_turn_proxy", "hip_rotation")
        shoulder_rotation = self._metric(metrics, "shoulder_turn_proxy", "shoulder_rotation")
        spine_angle = self._metric(metrics, "spine_angle_deg", "spine_angle")
        knee_flex = self._metric(metrics, "knee_flex_deg")
        balance_score = self._metric(metrics, "balance_score", "weight_shift")
        tempo_estimate = self._metric(metrics, "tempo_estimate")
        club_path = str(metrics.get("club_path", "")).lower()

        if knee_flex is not None and knee_flex < 135:
            observations.append({
                "title": "Give yourself a more athletic setup",
                "description": "You are getting a little tall over the ball. Add a soft bend in the knees so your legs can support the turn and help you move into the shot instead of just standing over it.",
            })
        if head_stability is not None and head_stability < 80:
            observations.append({
                "title": "Let your head stay quieter through impact",
                "description": "Your head is doing a bit too much traveling as the club comes through. Keep your eyes on the ball and let your chest rotate around that steady center—your strike will have a much better chance to repeat.",
            })
        elif head_stability is not None:
            observations.append({
                "title": "Keep that quiet head working for you",
                "description": "Your head stays nicely centered through the swing. That is a great anchor point, so keep building the rest of the motion around that steady base.",
            })
        if hip_rotation is not None and hip_rotation < 70:
            observations.append({
                "title": "Let your lower body lead the way",
                "description": "Your hips look a little quiet through the strike. Start the downswing from the ground up and let your belt buckle begin turning toward the target before the arms try to take over.",
            })
        elif hip_rotation is not None:
            observations.append({
                "title": "Keep clearing space with your hips",
                "description": "You are getting a good hip turn through the ball. Keep that move going so your arms have room to swing through instead of getting trapped behind you.",
            })
        if shoulder_rotation is not None and shoulder_rotation < 70:
            observations.append({
                "title": "Finish the shoulder turn",
                "description": "There is a little more turn available in your upper body. Let your lead shoulder work under your chin on the way back, then keep rotating through so the swing does not run out of room.",
            })
        if spine_angle is not None and abs(spine_angle - 10.0) > 8.0:
            observations.append({
                "title": "Stay in your posture a touch longer",
                "description": "Your posture is changing more than we want as the swing moves toward impact. Feel your chest stay over the ball while you rotate—athletic and balanced, not frozen in place.",
            })
        if balance_score is not None and balance_score < 70:
            observations.append({
                "title": "Move into your lead side with more intent",
                "description": "Your weight shift needs a little more purpose through the ball. Let your pressure arrive on the lead foot as you turn, then finish tall and balanced like you could hold the pose for a photo.",
            })
        if tempo_estimate is not None and tempo_estimate < 70:
            observations.append({
                "title": "Take the hurry out of the transition",
                "description": "The change from backswing to downswing is getting a little quick. Give yourself one smooth beat at the top, then let the lower body start the move before the club chases after it.",
            })
        if club_path == "outside_in":
            observations.append({
                "title": "Give the club a little more room from the inside",
                "description": "The club is working across the ball from outside-in. Feel the handle and hands drop closer to your trail pocket first, then swing out toward the target instead of cutting across it.",
            })

        if len(observations) < 3:
            for note in [*improvements, *strengths]:
                if len(observations) == 3:
                    break
                observations.append({
                    "title": "Keep building this move",
                    "description": str(note),
                })

        while len(observations) < 3:
            observations.append({
                "title": "Trust the rhythm you have",
                "description": "Keep your setup athletic, let the body turn through the ball, and finish in balance. Good golf does not need to look rushed to be powerful.",
            })

        return observations[:3]

    def _build_coach_summary(self, observations: Iterable[Dict[str, str]], *, fallback_used: bool) -> str:
        if fallback_used:
            return "Give me that clear side view on the next swing, and we can trade camera advice for the good stuff—your actual golf swing."

        notes = list(observations)
        if not notes:
            return "You have a useful foundation here. Keep the motion athletic, give the body room to turn, and take one simple feel to the next ball."

        return "You do not need to rebuild the whole swing today. Pick the first move, rehearse it slowly a few times, then let it show up in a normal swing—one good adjustment beats five thoughts over the ball."

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
