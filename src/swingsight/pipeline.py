from __future__ import annotations

from typing import Dict, Optional

from swingsight.feedback import generate_club_feedback
from swingsight.metrics import compute_swing_metrics
from swingsight.pose_estimation import run_pose_estimation
from swingsight.visualization import render_annotated_video


def analyze_swing(
    club_category: str,
    video_path: str,
    config: Dict,
    club_image_path: Optional[str] = None,
) -> Dict:
    """Run the end-to-end swing analysis pipeline."""
    _ = club_image_path

    landmarks = run_pose_estimation(video_path, config)
    metrics = compute_swing_metrics(landmarks, config)
    feedback = generate_club_feedback(metrics, club_category, config)
    annotated_video_path = render_annotated_video(
        video_path, landmarks, config["paths"]["outputs_dir"]
    )

    return {
        "club_category": club_category,
        "metrics": metrics,
        "feedback": feedback,
        "annotated_video_path": annotated_video_path,
    }
