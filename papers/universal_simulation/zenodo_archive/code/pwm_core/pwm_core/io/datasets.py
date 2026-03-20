"""pwm_core.io.datasets

Dataset loading policies:
- copy vs reference
- checksum
- manifest writing
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


@dataclass
class DatasetRef:
    mode: str  # "copy" | "reference"
    path: str
    checksum: Optional[str] = None
    size_mb: Optional[float] = None


def prepare_input_dataset(path: str, rb_dir: str, mode: str = "auto", copy_threshold_mb: int = 100) -> DatasetRef:
    """Return DatasetRef and copy into runbundle if needed."""
    size = file_size_mb(path)
    checksum = sha256_file(path)

    if mode == "auto":
        mode = "copy" if size <= copy_threshold_mb else "reference"

    if mode == "copy":
        dst_dir = os.path.join(rb_dir, "artifacts", "input_data")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy2(path, dst)
        return DatasetRef(mode="copy", path=dst, checksum=checksum, size_mb=size)

    return DatasetRef(mode="reference", path=path, checksum=checksum, size_mb=size)
