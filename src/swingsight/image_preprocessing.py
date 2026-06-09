from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _clahe_settings(clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> cv2.CLAHE:
    try:
        grid_x = int(tile_grid_size[0])
        grid_y = int(tile_grid_size[1])
    except Exception:
        grid_x, grid_y = 8, 8
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(max(1, grid_x), max(1, grid_y)))


def apply_adaptive_contrast_bgr(frame: np.ndarray, *, enabled: bool = True, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if not enabled or frame is None:
        return frame

    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = _clahe_settings(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def apply_adaptive_contrast_rgb(image: Image.Image, *, enabled: bool = True, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    if not enabled:
        return image

    rgb_frame = np.array(image.convert("RGB"))
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        return image

    lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = _clahe_settings(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)
