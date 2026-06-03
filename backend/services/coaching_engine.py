from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_src_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "src" / "backend" / "services" / "coaching_engine.py"
    spec = importlib.util.spec_from_file_location("swing_sight_src_backend_coaching_engine", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load src backend coaching_engine from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_src_module = _load_src_module()
CoachingEngine = _src_module.CoachingEngine
CoachingResult = getattr(_src_module, "CoachingResult", None)

__all__ = ["CoachingEngine", "CoachingResult"]
