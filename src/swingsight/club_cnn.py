"""Checkpoint-backed CNNs used by the staged club-recognition pipeline.

Each checkpoint is intentionally self-contained.  The training utility stores
the architecture version, class names, input size, normalization values, and
weights together, so inference does not rely on class ordering hidden in a
separate file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

try:  # Keep the local dashboard usable when optional ML dependencies are absent.
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - exercised only in minimal installations
    torch = None
    Tensor = Any
    nn = None


CHECKPOINT_FORMAT = "swingsight_club_cnn_v1"
SUPPORTED_TASKS = {"broad_category", "iron_number", "wood_type", "club_type_5way"}
DEFAULT_INPUT_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CnnPrediction:
    """The result of one CNN stage, including an actionable unavailable reason."""

    label: Optional[str]
    confidence: float
    probabilities: dict[str, float]
    source: str
    reasoning: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.label is not None


if nn is not None:

    class ClubCnnModel(nn.Module):
        """A compact CNN that is practical to train on a small, focused dataset."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(192, num_classes),
            )

        def forward(self, batch: Tensor) -> Tensor:
            return self.classifier(self.features(batch))

else:  # pragma: no cover - allows importing this module without torch

    class ClubCnnModel:  # type: ignore[no-redef]
        def __init__(self, num_classes: int) -> None:
            _ = num_classes
            raise RuntimeError("PyTorch is required to create a ClubCnnModel.")


@dataclass(frozen=True)
class _LoadedCheckpoint:
    model: Any
    class_names: tuple[str, ...]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    device: Any


def classify_image(
    image: Image.Image,
    *,
    checkpoint_path: str | Path | None,
    expected_task: str,
    minimum_confidence: float = 0.0,
) -> CnnPrediction:
    """Classify one RGB image with a validated stage checkpoint.

    A missing, corrupt, or mismatched checkpoint produces an explicit
    unavailable result.  It never falls back to an image-shape heuristic.
    """

    if expected_task not in SUPPORTED_TASKS:
        return _unavailable("cnn_invalid_task", f"Unsupported CNN task: {expected_task}.")
    if torch is None:
        return _unavailable("cnn_unavailable", "PyTorch is not installed.")
    if not checkpoint_path:
        return _unavailable("cnn_missing", f"No {expected_task} CNN checkpoint is configured.")

    path = Path(checkpoint_path).expanduser()
    if not path.exists() or not path.is_file():
        return _unavailable("cnn_missing", f"{expected_task} CNN checkpoint was not found: {path}.")

    try:
        loaded = _load_checkpoint(str(path.resolve()), expected_task)
    except Exception as exc:
        return _unavailable("cnn_invalid", f"Unable to load {expected_task} CNN checkpoint: {exc}")

    try:
        batch = image_to_tensor(image, loaded.input_size, loaded.mean, loaded.std).unsqueeze(0).to(loaded.device)
        with torch.inference_mode():
            scores = loaded.model(batch)
            probabilities = torch.softmax(scores, dim=1)[0].detach().cpu().numpy()
    except Exception as exc:
        return _unavailable("cnn_inference_error", f"{expected_task} CNN inference failed: {exc}")

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    probability_map = {
        label: round(float(probability), 6)
        for label, probability in zip(loaded.class_names, probabilities.tolist())
    }
    label = loaded.class_names[best_index]
    if confidence < minimum_confidence:
        return CnnPrediction(
            label=None,
            confidence=confidence,
            probabilities=probability_map,
            source="cnn",
            reasoning=(
                f"{expected_task} CNN confidence {confidence:.2f} is below "
                f"the required {minimum_confidence:.2f}."
            ),
        )

    return CnnPrediction(
        label=label,
        confidence=confidence,
        probabilities=probability_map,
        source="cnn",
        reasoning=f"{expected_task} CNN predicted {label}.",
    )


def image_to_tensor(
    image: Image.Image,
    input_size: int = DEFAULT_INPUT_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> Tensor:
    """Apply the exact deterministic preprocessing recorded by training."""

    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    size = max(16, int(input_size))
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose((2, 0, 1))))
    mean_tensor = torch.tensor(tuple(float(value) for value in mean), dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(tuple(max(1e-6, float(value)) for value in std), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def clear_classifier_cache() -> None:
    """Clear cached checkpoints, useful after replacing model weights in-process."""

    _load_checkpoint.cache_clear()


@lru_cache(maxsize=6)
def _load_checkpoint(path_string: str, expected_task: str) -> _LoadedCheckpoint:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    checkpoint = _torch_load(Path(path_string))
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint is not a mapping")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"expected format {CHECKPOINT_FORMAT}")
    if checkpoint.get("task") != expected_task:
        raise ValueError(f"checkpoint task is {checkpoint.get('task')!r}, expected {expected_task!r}")

    raw_names = checkpoint.get("class_names")
    if not isinstance(raw_names, (list, tuple)) or len(raw_names) < 2:
        raise ValueError("checkpoint must contain at least two class_names")
    class_names = tuple(str(name).strip() for name in raw_names)
    if any(not name for name in class_names) or len(set(class_names)) != len(class_names):
        raise ValueError("checkpoint class_names must be unique, non-empty values")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing state_dict")

    input_size = max(16, int(checkpoint.get("input_size", DEFAULT_INPUT_SIZE)))
    mean = _numeric_triplet(checkpoint.get("mean", DEFAULT_MEAN), "mean")
    std = _numeric_triplet(checkpoint.get("std", DEFAULT_STD), "std")
    device = _torch_device()
    model = ClubCnnModel(len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return _LoadedCheckpoint(
        model=model,
        class_names=class_names,
        input_size=input_size,
        mean=mean,
        std=std,
        device=device,
    )


def _torch_load(path: Path) -> Mapping[str, Any]:
    """Use weights-only loading on newer PyTorch releases when available."""

    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not have weights_only.
        return torch.load(path, map_location="cpu")


def _torch_device() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _numeric_triplet(values: object, field_name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"checkpoint {field_name} must have three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _unavailable(source: str, reasoning: str) -> CnnPrediction:
    return CnnPrediction(label=None, confidence=0.0, probabilities={}, source=source, reasoning=reasoning)
