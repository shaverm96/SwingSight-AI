from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SwingMetrics:
    spine_angle_deg: float
    head_movement_cm: float
    knee_flex_deg: float
    shoulder_turn_proxy: float
    hip_turn_proxy: float
    balance_score: float
    tempo_estimate: float

    def to_dict(self) -> Dict[str, float]:
        return dict(asdict(self))


def _compute_from_landmarks(landmarks: List[Dict], config: Dict) -> SwingMetrics:
    """Compute metrics from per-frame landmark dicts.

    Replace with geometry-based calculations when pose inference is ready.
    """
    _ = landmarks
    _ = config
    return SwingMetrics(
        spine_angle_deg=0.0,
        head_movement_cm=0.0,
        knee_flex_deg=0.0,
        shoulder_turn_proxy=0.0,
        hip_turn_proxy=0.0,
        balance_score=0.0,
        tempo_estimate=0.0,
    )


def compute_swing_metrics(landmarks: List[Dict], config: Dict) -> Dict[str, float]:
    """Compute basic swing metrics from landmarks.

    Returns a dict for easy JSON serialization.
    """
    metrics = _compute_from_landmarks(landmarks, config)
    return metrics.to_dict()
