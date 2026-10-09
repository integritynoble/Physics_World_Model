"""Bounded public data access and EDF curve extraction for IPWM-24."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(os.environ.get("IPWM24_DATA", "/tmp/ipwm24-data"))
SAXS_DIR = DATA_DIR / "saxs"
SAXS_URL = "https://archive.researchdata.leeds.ac.uk/1010/1/SAXS_patterns.zip"
SAXS_FILES = [f"beam_0_{i:05d}.edf" for i in range(46, 66)]


def fetch_saxs() -> dict[str, dict[str, Any]]:
    """Fetch only the 20 one-second SiO2 frames used by the benchmark.

    The archive itself is temporary and is removed after selected frames have
    been extracted. Only retained EDF frame hashes are returned.
    """
    SAXS_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "saxs_download.zip"
    missing = [name for name in SAXS_FILES if not (SAXS_DIR / name).is_file()]
    if missing:
        with urllib.request.urlopen(SAXS_URL, timeout=120) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            for name in missing:
                (SAXS_DIR / name).write_bytes(zf.read("SAXS patterns/" + name))
        archive.unlink(missing_ok=True)
    manifest = {}
    for name in SAXS_FILES:
        raw = (SAXS_DIR / name).read_bytes()
        manifest[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return manifest


def read_edf(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the signed-int32 EDF subset and relevant acquisition metadata."""
    raw = Path(path).read_bytes()
    marker = raw.find(b"\x0c\x1a")
    if marker < 0:
        marker = raw.find(b"}") + 1
    if marker <= 0:
        raise ValueError(f"{path}: EDF header terminator not found")
    header = raw[:marker].decode("latin1", errors="strict")

    def value(name: str, cast):
        match = re.search(rf"\b{name}\s*=\s*([^;]+)\s*;", header)
        if match is None:
            raise ValueError(f"{path}: missing EDF field {name}")
        return cast(match.group(1).strip())

    size = value("EDF_HeaderSize", int)
    width, height = value("Dim_1", int), value("Dim_2", int)
    data = np.frombuffer(raw[size:], dtype="<i4")
    if data.size != width * height:
        raise ValueError(f"{path}: expected {width * height} int32 values, found {data.size}")
    meta = {
        "center_px": [value("Center_1", float), value("Center_2", float)],
        "pixel_size_m": [value("PSize_1", float), value("PSize_2", float)],
        "wavelength_m": value("WaveLength", float),
        "exposure_s": value("ExposureTime", float),
        "sample_distance_header_m": value("SampleDistance", float),
        "shape": [height, width],
        "dummy": value("Dummy", float),
    }
    return data.reshape(height, width).copy(), meta


def radial_curve(path: str | Path, n_bins: int = 29,
                 r_min_px: float = 60.0, r_max_px: float = 350.0) -> tuple[np.ndarray, np.ndarray]:
    """Return mean background-corrected intensity by detector-radius bin.

    Curve x coordinates are detector radius in pixels. Absolute q conversion
    is intentionally omitted because source documentation and EDF headers
    disagree on detector distance for these frames.
    """
    image, meta = read_edf(path)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    cx, cy = meta["center_px"]
    radius = np.hypot(xx - cx, yy - cy)
    edges = np.linspace(r_min_px, r_max_px, n_bins + 1)
    bins = np.digitize(radius.ravel(), edges) - 1
    vals = image.ravel().astype(np.float64)
    valid = (bins >= 0) & (bins < n_bins) & (vals > meta["dummy"])
    sums = np.bincount(bins[valid], weights=vals[valid], minlength=n_bins)
    counts = np.bincount(bins[valid], minlength=n_bins)
    if np.any(counts == 0):
        raise ValueError("empty detector-radius bins")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, sums / counts


def canonical_curve_bytes(curve: Any) -> bytes:
    """Stable byte encoding used to detect exact copies of public curves."""
    return np.ascontiguousarray(np.asarray(curve, dtype="<f8")).tobytes()


def curve_sha256(curve: Any) -> str:
    return hashlib.sha256(canonical_curve_bytes(curve)).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

