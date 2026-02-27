"""Dataset upload handling and metadata extraction.

Supports .npy, .npz, .mat (v5 and v7.3), .h5/.hdf5 files.
Saves uploads to static/uploads/{session_id}/ and extracts shape/dtype/stats.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

_ALLOWED_EXTENSIONS = {".npy", ".npz", ".mat", ".h5", ".hdf5"}
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@dataclass
class DatasetMetadata:
    """Metadata extracted from an uploaded dataset file."""

    file_path: str
    original_filename: str
    file_format: str            # "npy", "mat", "h5", "npz"
    data_key: str               # Array key within container files
    shape: tuple
    dtype: str
    stats: dict                 # {min, max, mean, std}
    file_size_bytes: int
    role: str                   # "measurement", "sensing_matrix", "ground_truth"
    available_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(d["shape"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetMetadata:
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


async def save_uploaded_file(
    file_content: bytes,
    original_filename: str,
    session_id: str,
    role: str,
) -> str:
    """Save uploaded file bytes to disk.

    Returns the full file path.
    """
    ext = Path(original_filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    if len(file_content) > _MAX_FILE_SIZE:
        raise ValueError(
            f"File too large ({len(file_content) / 1024 / 1024:.1f} MB). "
            f"Maximum: {_MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
        )

    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{role}{ext}"
    file_path = session_dir / filename
    file_path.write_bytes(file_content)

    logger.info("Saved upload: %s (%d bytes)", file_path, len(file_content))
    return str(file_path)


def extract_dataset_metadata(file_path: str, role: str) -> DatasetMetadata:
    """Read a saved file and extract metadata (shape, dtype, stats)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    file_size = path.stat().st_size

    if ext == ".npy":
        arr = np.load(str(path))
        return _build_meta(
            arr, file_path, path.name, "npy", "__array__", role, file_size, []
        )

    if ext == ".npz":
        npz = np.load(str(path))
        keys = list(npz.keys())
        # Pick largest array
        best_key = max(keys, key=lambda k: npz[k].size) if keys else keys[0]
        arr = npz[best_key]
        return _build_meta(
            arr, file_path, path.name, "npz", best_key, role, file_size, keys
        )

    if ext == ".mat":
        return _load_mat_metadata(file_path, path.name, role, file_size)

    if ext in (".h5", ".hdf5"):
        return _load_h5_metadata(file_path, path.name, role, file_size)

    raise ValueError(f"Unsupported extension: {ext}")


def load_array_from_upload(meta: DatasetMetadata) -> np.ndarray:
    """Load the actual numpy array from an upload based on its metadata."""
    path = meta.file_path
    ext = meta.file_format

    if ext == "npy":
        return np.load(path)

    if ext == "npz":
        return np.load(path)[meta.data_key]

    if ext == "mat":
        return _load_mat_array(path, meta.data_key)

    if ext in ("h5", "hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return np.array(f[meta.data_key])

    raise ValueError(f"Unsupported format: {ext}")


# ── Internal helpers ──────────────────────────────────────────────────────


def _build_meta(
    arr: np.ndarray,
    file_path: str,
    filename: str,
    fmt: str,
    key: str,
    role: str,
    file_size: int,
    available_keys: list[str],
) -> DatasetMetadata:
    arr_float = arr.astype(np.float64)
    stats = {
        "min": float(np.nanmin(arr_float)),
        "max": float(np.nanmax(arr_float)),
        "mean": float(np.nanmean(arr_float)),
        "std": float(np.nanstd(arr_float)),
    }
    return DatasetMetadata(
        file_path=file_path,
        original_filename=filename,
        file_format=fmt,
        data_key=key,
        shape=arr.shape,
        dtype=str(arr.dtype),
        stats=stats,
        file_size_bytes=file_size,
        role=role,
        available_keys=available_keys,
    )


def _load_mat_metadata(
    file_path: str, filename: str, role: str, file_size: int,
) -> DatasetMetadata:
    """Load .mat file — try scipy first (v5), fall back to h5py (v7.3)."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(file_path)
        # Filter internal keys
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError("No data arrays found in .mat file")
        best_key = max(keys, key=lambda k: mat[k].size)
        arr = np.asarray(mat[best_key])
        return _build_meta(
            arr, file_path, filename, "mat", best_key, role, file_size, keys
        )
    except NotImplementedError:
        # v7.3 format — use h5py
        return _load_h5_metadata(file_path, filename, role, file_size, fmt="mat")


def _load_h5_metadata(
    file_path: str,
    filename: str,
    role: str,
    file_size: int,
    fmt: str = "h5",
) -> DatasetMetadata:
    import h5py
    with h5py.File(file_path, "r") as f:
        keys = _list_h5_datasets(f)
        if not keys:
            raise ValueError("No datasets found in HDF5 file")
        best_key = max(keys, key=lambda k: np.prod(f[k].shape))
        arr = np.array(f[best_key])
    return _build_meta(
        arr, file_path, filename, fmt, best_key, role, file_size, keys
    )


def _load_mat_array(file_path: str, key: str) -> np.ndarray:
    try:
        import scipy.io
        mat = scipy.io.loadmat(file_path)
        return np.asarray(mat[key])
    except NotImplementedError:
        import h5py
        with h5py.File(file_path, "r") as f:
            return np.array(f[key])


def _list_h5_datasets(group, prefix: str = "") -> list[str]:
    """Recursively list dataset paths in an HDF5 group."""
    import h5py
    result = []
    for key in group.keys():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(group[key], h5py.Dataset):
            result.append(path)
        elif isinstance(group[key], h5py.Group):
            result.extend(_list_h5_datasets(group[key], path))
    return result
