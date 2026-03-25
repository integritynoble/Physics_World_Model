"""pwm_core.data.ingest

Backend for ``pwm ingest`` — ingests external datasets into the PWM format.

Public API
----------
    ingest_dataset(
        modality: str,
        source: str,
        fmt: str = "auto",
        out_dir: str | Path = "data/ingested",
        n_samples: int | None = None,
    ) -> Path

Supported source formats:
    npy / npz / hdf5 / h5 / png / tiff / tif / dicom / dcm / nifti / nii / gz
    GCS paths: gs://bucket/path/...  (requires google-cloud-storage)

Each ingested sample is written as .npz with keys: x_true, params_json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def ingest_dataset(
    modality: str,
    source: str,
    fmt: str = "auto",
    out_dir: str | Path = "data/ingested",
    n_samples: Optional[int] = None,
) -> Path:
    """Ingest an external dataset and write it to the PWM .npz format.

    Parameters
    ----------
    modality:
        Physics modality label (e.g. "ct", "mri").
    source:
        Path or GCS URI to the dataset file or directory.
    fmt:
        File format hint: "auto" | "npy" | "npz" | "hdf5" | "png" | "tiff" |
        "dicom" | "nifti" | "gs"
    out_dir:
        Root output directory; files land in ``<out_dir>/<modality>/``.
    n_samples:
        Maximum number of samples to ingest (None = all).

    Returns
    -------
    Path
        The directory where .npz files were written.
    """
    out_path = Path(out_dir) / modality
    out_path.mkdir(parents=True, exist_ok=True)

    if fmt == "auto":
        fmt = _detect_format(source)

    logger.info("ingesting %s (fmt=%s) → %s", source, fmt, out_path)

    loader = _FORMAT_LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format: {fmt!r}. Supported: {sorted(_FORMAT_LOADERS)}")

    count = 0
    for idx, (x_true, meta) in enumerate(loader(source)):
        if n_samples is not None and idx >= n_samples:
            break
        meta["modality"] = modality
        meta["source"] = source
        out_file = out_path / f"sample_{idx:04d}.npz"
        np.savez_compressed(
            out_file,
            x_true=x_true.astype(np.float32),
            params_json=json.dumps(meta, default=str),
        )
        count += 1
        logger.debug("wrote %s", out_file)

    logger.info("ingested %d samples → %s", count, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(source: str) -> str:
    if source.startswith("gs://"):
        return "gs"
    p = Path(source)
    ext = p.suffix.lower()
    if p.is_dir():
        # Check contents
        exts = {f.suffix.lower() for f in p.iterdir() if f.is_file()}
        if ".dcm" in exts:
            return "dicom"
        if ".png" in exts:
            return "png"
        if ".tif" in exts or ".tiff" in exts:
            return "tiff"
        if ".nii" in exts or ".gz" in exts:
            return "nifti"
        return "npy"
    return {
        ".npy": "npy",
        ".npz": "npz",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".png": "png",
        ".tif": "tiff",
        ".tiff": "tiff",
        ".dcm": "dicom",
        ".nii": "nifti",
        ".gz": "nifti",
    }.get(ext, "npy")


# ---------------------------------------------------------------------------
# Format loaders — each yields (x_true: np.ndarray, meta: dict)
# ---------------------------------------------------------------------------

def _load_npy(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    arr = np.load(source)
    if arr.ndim == 2:
        yield arr, {"shape": list(arr.shape)}
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            yield arr[i], {"index": i, "shape": list(arr[i].shape)}
    else:
        yield arr, {"shape": list(arr.shape)}


def _load_npz(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    data = np.load(source)
    for key in data.files:
        arr = data[key]
        if arr.ndim >= 2:
            yield arr, {"npz_key": key, "shape": list(arr.shape)}


def _load_hdf5(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 ingestion: pip install h5py")

    def _walk(group, prefix=""):
        for key in group:
            item = group[key]
            path = f"{prefix}/{key}"
            if hasattr(item, "shape") and item.ndim >= 2:
                arr = item[()]
                if arr.ndim == 2:
                    yield arr, {"h5_path": path, "shape": list(arr.shape)}
                elif arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        yield arr[i], {"h5_path": path, "index": i, "shape": list(arr[i].shape)}
            elif hasattr(item, "keys"):
                yield from _walk(item, path)

    with h5py.File(source, "r") as f:
        yield from _walk(f)


def _load_png(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for PNG ingestion: pip install Pillow")

    p = Path(source)
    files = sorted(p.glob("*.png")) if p.is_dir() else [p]
    for fpath in files:
        img = Image.open(fpath).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        yield arr, {"filename": fpath.name, "shape": list(arr.shape)}


def _load_tiff(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile required: pip install tifffile")

    p = Path(source)
    files = sorted(p.glob("*.tif*")) if p.is_dir() else [p]
    for fpath in files:
        arr = tifffile.imread(str(fpath)).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        if arr.ndim == 2:
            yield arr, {"filename": fpath.name, "shape": list(arr.shape)}
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                yield arr[i], {"filename": fpath.name, "slice": i, "shape": list(arr[i].shape)}


def _load_dicom(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom required for DICOM ingestion: pip install pydicom")

    p = Path(source)
    files = sorted(p.glob("*.dcm")) if p.is_dir() else [p]
    for fpath in files:
        ds = pydicom.dcmread(str(fpath))
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        meta: Dict[str, Any] = {"filename": fpath.name, "shape": list(arr.shape)}
        if hasattr(ds, "PatientID"):
            meta["patient_id"] = str(ds.PatientID)
        if hasattr(ds, "Modality"):
            meta["dicom_modality"] = str(ds.Modality)
        yield arr, meta


def _load_nifti(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required for NIfTI ingestion: pip install nibabel")

    p = Path(source)
    files: List[Path] = []
    if p.is_dir():
        files = sorted(p.glob("*.nii")) + sorted(p.glob("*.nii.gz"))
    else:
        files = [p]

    for fpath in files:
        img = nib.load(str(fpath))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.max() > 1.0:
            data = data / data.max()
        if data.ndim == 2:
            yield data, {"filename": fpath.name, "shape": list(data.shape)}
        elif data.ndim == 3:
            for i in range(data.shape[2]):
                yield data[:, :, i], {"filename": fpath.name, "slice": i, "shape": list(data[:, :, i].shape)}


def _load_gcs(source: str) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    """Download from GCS and ingest based on file extension."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required: pip install google-cloud-storage")

    import tempfile

    # Parse gs://bucket/path
    without_prefix = source[5:]  # strip "gs://"
    bucket_name, blob_path = without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    suffix = Path(blob_path).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        local_fmt = _detect_format(tmp.name)
        local_loader = _FORMAT_LOADERS.get(local_fmt, _load_npy)
        yield from local_loader(tmp.name)


_FORMAT_LOADERS = {
    "npy": _load_npy,
    "npz": _load_npz,
    "hdf5": _load_hdf5,
    "h5": _load_hdf5,
    "png": _load_png,
    "tiff": _load_tiff,
    "tif": _load_tiff,
    "dicom": _load_dicom,
    "dcm": _load_dicom,
    "nifti": _load_nifti,
    "nii": _load_nifti,
    "gz": _load_nifti,
    "gs": _load_gcs,
}
