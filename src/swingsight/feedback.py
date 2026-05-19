from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeedbackItem:
    message: str
    severity: str = "info"


class FeedbackEngine:
    """Rule-based feedback generator for the MVP."""

    def __init__(self, config: Dict) -> None:
        self.config = config

    def generate(self, metrics: Dict[str, float], club_category: str) -> List[FeedbackItem]:
        feedback: List[FeedbackItem] = []
        if club_category == "driver_wood":
            feedback.append(FeedbackItem("Focus on a stable head position and smooth tempo."))
        elif club_category == "hybrid":
            feedback.append(FeedbackItem("Maintain a balanced finish and controlled rotation."))
        else:
            feedback.append(FeedbackItem("Keep posture steady and avoid early extension."))

        if metrics.get("balance_score", 0.0) < 0.5:
            feedback.append(FeedbackItem("Work on finish stability to improve balance."))

        return feedback


def build_feedback_engine(config: Dict) -> FeedbackEngine:
    return FeedbackEngine(config)


def generate_club_feedback(metrics: Dict[str, float], club_category: str, config: Dict) -> List[str]:
    """Generate club-aware feedback based on metrics.

    Returns a list of strings for easy display.
    """
    engine = build_feedback_engine(config)
    return [item.message for item in engine.generate(metrics, club_category)]
