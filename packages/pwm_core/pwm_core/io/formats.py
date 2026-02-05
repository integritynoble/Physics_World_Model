"""pwm_core.io.formats

Minimal format helpers. Extend as needed:
- OME-TIFF
- NIfTI
- HDF5 / Zarr
- NPZ / PT
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def load_npz(path: str, key: str = "arr_0") -> np.ndarray:
    d = np.load(path)
    return d[key]


def save_npz(path: str, arr: np.ndarray, key: str = "arr_0") -> None:
    np.savez_compressed(path, **{key: arr})
