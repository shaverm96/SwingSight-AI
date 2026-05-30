from __future__ import annotations

from pathlib import Path
from typing import IO
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_filestorage(file_obj: FileStorage, destination_dir: str | Path, prefix: str) -> tuple[str, str]:
    """Persist an uploaded file and return (id, absolute path)."""
    directory = ensure_dir(destination_dir)
    safe_name = secure_filename(file_obj.filename or "upload.bin")
    suffix = Path(safe_name).suffix
    file_id = uuid4().hex
    filename = f"{prefix}_{file_id}{suffix}"
    destination = directory / filename
    file_obj.save(destination)
    return file_id, str(destination)


def copy_local_file(handle: IO[bytes], destination_dir: str | Path, filename_hint: str) -> str:
    """Copy a binary stream into destination_dir and return absolute path."""
    directory = ensure_dir(destination_dir)
    suffix = Path(filename_hint).suffix
    destination = directory / f"copied_{uuid4().hex}{suffix}"
    with destination.open("wb") as out_file:
        out_file.write(handle.read())
    return str(destination)
