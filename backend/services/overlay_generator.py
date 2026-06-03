from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_src_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "src" / "backend" / "services" / "overlay_generator.py"
    spec = importlib.util.spec_from_file_location("swing_sight_src_backend_overlay_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load src backend overlay_generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_src_module = _load_src_module()
generate_pose_overlay = _src_module.generate_pose_overlay
draw_skeleton = _src_module.draw_skeleton
draw_motion_trails = _src_module.draw_motion_trails
save_overlay_video = _src_module.save_overlay_video
load_pose_landmarks = _src_module.load_pose_landmarks
validate_overlay_video = _src_module.validate_overlay_video

__all__ = [
    "generate_pose_overlay",
    "draw_skeleton",
    "draw_motion_trails",
    "save_overlay_video",
    "load_pose_landmarks",
    "validate_overlay_video",
]