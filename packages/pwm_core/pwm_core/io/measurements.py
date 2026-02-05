"""pwm_core.io.measurements

I/O for measured y (and metadata).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pwm_core.io.formats import load_npz, save_npz


@dataclass
class Measurement:
    y_path: str
    meta: Dict[str, Any]

    def load(self) -> np.ndarray:
        if self.y_path.endswith(".npz"):
            return load_npz(self.y_path)
        if self.y_path.endswith(".npy"):
            return np.load(self.y_path)
        raise ValueError(f"Unsupported measurement format: {self.y_path}")


def load_measurement(y_path: str, meta_path: Optional[str] = None) -> Measurement:
    meta: Dict[str, Any] = {}
    if meta_path:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return Measurement(y_path=y_path, meta=meta)


def save_measurement(y: np.ndarray, out_dir: str, name: str = "y") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{name}.npz")
    save_npz(path, y)
    return path
