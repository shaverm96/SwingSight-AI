from __future__ import annotations

from typing import Dict, List


def generate_club_feedback(metrics: Dict[str, float], club_category: str, config: Dict) -> List[str]:
    """Generate club-aware feedback based on metrics.

    Expand these rules as metrics mature.
    """
    _ = config

    feedback = []
    if club_category == "driver_wood":
        feedback.append("Focus on a stable head position and smooth tempo.")
    elif club_category == "hybrid":
        feedback.append("Maintain a balanced finish and controlled rotation.")
    else:
        feedback.append("Keep posture steady and avoid early extension.")

    if metrics.get("balance_score", 0.0) < 0.5:
        feedback.append("Work on finish stability to improve balance.")

    return feedback
