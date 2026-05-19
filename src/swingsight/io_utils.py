from __future__ import annotations

from pathlib import Path


def save_uploaded_file(uploaded_file, output_dir: str, prefix: str) -> str:
    """Save a Streamlit UploadedFile to disk and return the path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ""
    file_path = output_path / f"{prefix}{suffix}"

    with file_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())

    return str(file_path)
