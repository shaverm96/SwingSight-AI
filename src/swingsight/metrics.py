from __future__ import annotations

from typing import Dict, List


def compute_swing_metrics(landmarks: List[Dict], config: Dict) -> Dict[str, float]:
    """Compute basic swing metrics from landmarks.

    Replace these placeholders with real geometry-based calculations.
    """
    _ = landmarks
    _ = config

    return {
        "spine_angle_deg": 0.0,
        "head_movement_cm": 0.0,
        "knee_flex_deg": 0.0,
        "shoulder_turn_proxy": 0.0,
        "hip_turn_proxy": 0.0,
        "balance_score": 0.0,
        "tempo_estimate": 0.0,
    }
