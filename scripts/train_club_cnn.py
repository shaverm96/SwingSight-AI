"""Train one checkpoint for the staged SwingSight club-recognition CNN.

Example:
    python scripts/train_club_cnn.py --task broad_category \
      --data-dir data/club_cnn/broad_category \
      --output models/trained/club_broad_cnn.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swingsight.club_cnn import (  # noqa: E402
    CHECKPOINT_FORMAT,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    ClubCnnModel,
    image_to_tensor,
)


TASK_CLASSES = {
    "broad_category": ("iron", "wood"),
    "iron_number": tuple(str(number) for number in range(1, 10)),
    "wood_type": ("driver", "wood", "hybrid"),
    "club_type_5way": ("driver", "wood", "hybrid", "iron", "wedge"),
    # Exact sole/face markings after the five-way model selects Iron or Wedge.
    "club_marking": (
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "p", "a", "g", "s", "l",
        "50", "52", "54", "56", "58", "60",
    ),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SwingSight club-recognition CNN stage.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_CLASSES))
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing train/ and val/ class folders.")
    parser.add_argument("--output", required=True, type=Path, help="Destination .pt checkpoint path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on Windows unless dataset loading is slow.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ClubImageDataset:
    """Small dependency-free ImageFolder-style dataset for deterministic class ordering."""

    def __init__(
        self,
        root: Path,
        class_names: Sequence[str],
        *,
        input_size: int,
        augment: bool,
    ) -> None:
        self.root = root
        self.class_names = tuple(class_names)
        self.input_size = input_size
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for class_index, class_name in enumerate(self.class_names):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise ValueError(f"Missing class folder: {class_dir}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((image_path, class_index))

        if not self.samples:
            raise ValueError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        image_path, class_index = self.samples[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if self.augment:
            image = _augment(image)
        tensor = image_to_tensor(image, self.input_size, DEFAULT_MEAN, DEFAULT_STD)
        return tensor, torch.tensor(class_index, dtype=torch.long)


def _augment(image: Image.Image) -> Image.Image:
    """Use lighting-only augmentation; never mirror a number on an iron marking."""

    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.85, 1.15)
    return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(brightness)).enhance(contrast)


def validate_layout(data_dir: Path, task: str) -> tuple[str, ...]:
    expected = TASK_CLASSES[task]
    for split_name in ("train", "val"):
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            raise ValueError(f"Missing required split folder: {split_dir}")
        actual = tuple(sorted(path.name for path in split_dir.iterdir() if path.is_dir()))
        if set(actual) != set(expected):
            raise ValueError(
                f"{split_dir} must contain exactly these class folders: {', '.join(expected)}. "
                f"Found: {', '.join(actual) or 'none'}."
            )
    return expected


def select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets).item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.size(0))
    return (total_loss / max(1, total), correct / max(1, total))


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.input_size < 16:
        raise ValueError("epochs and batch-size must be positive; input-size must be at least 16.")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class_names = validate_layout(args.data_dir, args.task)
    train_dataset = ClubImageDataset(args.data_dir / "train", class_names, input_size=args.input_size, augment=True)
    val_dataset = ClubImageDataset(args.data_dir / "val", class_names, input_size=args.input_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = select_device()
    model = ClubCnnModel(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.task} on {device} with {len(train_dataset)} train and {len(val_dataset)} validation images.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * targets.size(0)
            train_total += int(targets.size(0))

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss / max(1, train_total):.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.3f}"
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "task": args.task,
                "architecture": "club_cnn_small_v1",
                "class_names": list(class_names),
                "input_size": int(args.input_size),
                "mean": list(DEFAULT_MEAN),
                "std": list(DEFAULT_STD),
                "validation_accuracy": round(float(val_accuracy), 6),
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            }
            torch.save(checkpoint, args.output)

    print(f"Saved best checkpoint ({best_accuracy:.3f} validation accuracy) to {args.output}")


if __name__ == "__main__":
    main()
