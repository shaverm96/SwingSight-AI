from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from swingsight.feedback import build_feedback_engine
from swingsight.metrics import compute_swing_metrics
from swingsight.pose_estimation import build_pose_estimator
from swingsight.visualization import build_visualizer


@dataclass(frozen=True)
class SwingAnalysisResult:
    club_category: str
    metrics: Dict[str, float]
    feedback: list[str]
    annotated_video_path: Optional[str]

    def to_dict(self) -> Dict:
        return dict(asdict(self))


class SwingAnalyzer:
    """Orchestrates the end-to-end swing analysis workflow."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.pose_estimator = build_pose_estimator(config)
        self.feedback_engine = build_feedback_engine(config)
        self.visualizer = build_visualizer()

    def analyze(
        self,
        club_category: str,
        video_path: str,
        club_image_path: Optional[str] = None,
    ) -> SwingAnalysisResult:
        _ = club_image_path

        frames = self.pose_estimator.infer(video_path)
        serialized_frames = [
            {
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "landmarks": {
                    name: {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for name, lm in frame.landmarks.items()
                },
            }
            for frame in frames
        ]
        metrics = compute_swing_metrics(serialized_frames, self.config)
        feedback_items = self.feedback_engine.generate(metrics, club_category)
        feedback = [item.message for item in feedback_items]
        render_result = self.visualizer.render(
            video_path, serialized_frames, self.config["paths"]["outputs_dir"]
        )

        return SwingAnalysisResult(
            club_category=club_category,
            metrics=metrics,
            feedback=feedback,
            annotated_video_path=render_result.annotated_video_path,
        )


def analyze_swing(
    club_category: str,
    video_path: str,
    config: Dict,
    club_image_path: Optional[str] = None,
) -> Dict:
    """Run the end-to-end swing analysis pipeline."""
    analyzer = SwingAnalyzer(config)
    result = analyzer.analyze(club_category, video_path, club_image_path)
    return result.to_dict()
