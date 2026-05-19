from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RenderResult:
    annotated_video_path: Optional[str]
    preview_frames: List[str]


class Visualizer:
    """Rendering scaffold for annotated outputs."""

    def render(self, video_path: str, landmarks: List[Dict], output_dir: str) -> RenderResult:
        _ = video_path
        _ = landmarks
        _ = output_dir
        return RenderResult(annotated_video_path=None, preview_frames=[])


def build_visualizer() -> Visualizer:
    return Visualizer()


def render_annotated_video(video_path: str, landmarks: List[Dict], output_dir: str) -> Optional[str]:
    """Render annotated frames or video.

    Returns the annotated video path when available.
    """
    visualizer = build_visualizer()
    result = visualizer.render(video_path, landmarks, output_dir)
    return result.annotated_video_path
