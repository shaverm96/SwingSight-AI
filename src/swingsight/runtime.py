from __future__ import annotations

from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def select_yolo_device() -> int | str:
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass

        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
        except Exception:
            pass

    return "cpu"
