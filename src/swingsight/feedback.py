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


def generate_feedback_sections(metrics: Dict[str, float], club_category: str, config: Dict) -> Dict[str, List[str]]:
    """Return structured feedback sections for the UI."""
    engine = build_feedback_engine(config)
    base_messages = [item.message for item in engine.generate(metrics, club_category)]

    body_position: List[str] = []
    club_path: List[str] = []
    key_suggestions: List[str] = []

    balance_score = metrics.get("balance_score", 1.0)
    if balance_score < 0.5:
        body_position.append("Balance looks unstable. Focus on a steady finish posture.")
    else:
        body_position.append("Posture stays stable through the swing and finish.")

    if club_category in {"driver_wood", "hybrid"}:
        club_path.append("Work on a shallow approach through impact for woods.")
    else:
        club_path.append("Maintain a crisp, descending strike path for irons and wedges.")

    key_suggestions.extend(base_messages)
    if not key_suggestions:
        key_suggestions.append("Keep tempo smooth and finish balanced.")

    return {
        "body_position": body_position,
        "club_path": club_path,
        "key_suggestions": key_suggestions,
    }


def generate_club_feedback(metrics: Dict[str, float], club_category: str, config: Dict) -> List[str]:
    """Generate club-aware feedback as a flat list for reports."""
    sections = generate_feedback_sections(metrics, club_category, config)
    flattened: List[str] = []
    for items in sections.values():
        flattened.extend(items)
    return flattened
