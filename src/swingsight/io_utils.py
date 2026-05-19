from __future__ import annotations

from pathlib import Path

from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_uploaded_file(uploaded_file, output_dir: str, prefix: str) -> str:
    """Save a Streamlit UploadedFile to disk and return the path."""
    output_path = ensure_dir(output_dir)
    suffix = Path(uploaded_file.name).suffix or ""
    file_path = output_path / f"{prefix}{suffix}"

    with file_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())

    return str(file_path)


def load_image(path: str) -> Image.Image:
    """Load an image from disk."""
    return Image.open(path)
