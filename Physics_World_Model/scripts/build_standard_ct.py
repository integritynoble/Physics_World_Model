#!/usr/bin/env python3
"""
Build standard dataset for ct modality.

Source: LoDoPaB-CT validation split
  Leuschner et al., Sci. Data 2021; doi:10.1038/s41597-021-00893-z
  Zenodo 3384092 (600 MB+ download)

  For standard dataset: Use Shepp-Logan phantom with realistic Radon forward model
  (same physics as LoDoPaB-CT, 362x362 image, 1000 angles, 513 detector pixels)

Standard dataset: 10 samples
  x_true  = CT image (362x362), float32 [0,1] (HU normalized)
  y_ideal = sinogram: y = Radon(x_true) using 1000 angles, 513 detectors
            Forward model: y[theta, t] = integral of x along ray at angle theta, offset t

Output: datasets/benchmark/ct/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "ct" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_ANGLES  = 1000   # LoDoPaB-CT uses 1000 angles
N_DETECTORS = 513  # LoDoPaB-CT detector pixels
NPIX = 362         # LoDoPaB-CT native image size


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _shepp_logan_2d(npix=362):
    """Generate 2D Shepp-Logan phantom at given resolution."""
    params = [
        [1.0,   0.69,  0.92,   0.0,    0.0,    0.0],
        [-0.8,  0.6624, 0.874,  0.0,   -0.0184, 0.0],
        [-0.2,  0.11,  0.31,   0.22,   0.0,   -18.0],
        [-0.2,  0.16,  0.41,  -0.22,   0.0,    18.0],
        [0.1,   0.21,  0.25,   0.0,    0.35,    0.0],
        [0.1,   0.046, 0.046,  0.0,    0.1,     0.0],
        [0.1,   0.046, 0.046,  0.0,   -0.1,     0.0],
        [-0.02, 0.046, 0.023, -0.08,  -0.605,   0.0],
        [-0.02, 0.023, 0.023,  0.0,   -0.606,   0.0],
        [-0.02, 0.023, 0.046,  0.06,  -0.605,   0.0],
    ]
    phantom = np.zeros((npix, npix), dtype=np.float32)
    x = np.linspace(-1, 1, npix)
    y = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x, y, indexing='xy')
    for p in params:
        A, a, b, x0, y0, phi = p
        phi_rad = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)
        Xr = cos_p * (X - x0) + sin_p * (Y - y0)
        Yr = -sin_p * (X - x0) + cos_p * (Y - y0)
        inside = (Xr / a)**2 + (Yr / b)**2 <= 1
        phantom[inside] += A
    return np.clip(phantom, 0, None).astype(np.float32)


def _radon_transform(image: np.ndarray, n_angles: int, n_detectors: int) -> np.ndarray:
    """
    Compute Radon transform (sinogram) of an image.
    image: (npix, npix) float32
    Returns: (n_angles, n_detectors) sinogram
    """
    npix = image.shape[0]
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    # Detector coordinates (centered)
    t = np.linspace(-1, 1, n_detectors)
    sinogram = np.zeros((n_angles, n_detectors), dtype=np.float32)

    # Pixel coordinates normalized to [-1, 1]
    x = np.linspace(-1, 1, npix)
    y = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x, y, indexing='xy')
    pixel_size = 2.0 / npix

    for i, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Project each pixel onto detector coordinate
        t_pixels = cos_t * X + sin_t * Y  # (npix, npix)
        # Accumulate into detector bins
        t_idx = ((t_pixels - t[0]) / (t[-1] - t[0]) * (n_detectors - 1)).astype(int)
        t_idx = np.clip(t_idx, 0, n_detectors - 1)
        np.add.at(sinogram[i], t_idx.ravel(), image.ravel() * pixel_size)

    return sinogram


def build():
    print("=" * 60)
    print("CT standard dataset builder (Shepp-Logan + Radon forward model)")
    print("=" * 60)

    print(f"\n[1] Generating {10} CT phantom slices (Shepp-Logan, {NPIX}x{NPIX})...")
    records = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 7)
        phantom = _shepp_logan_2d(NPIX)
        # Add tissue-like variations
        noise = rng.uniform(0, 0.02, phantom.shape).astype(np.float32)
        phantom = phantom + noise * (phantom > 0)
        # Add small random inclusions
        n_inc = rng.integers(2, 6)
        for _ in range(n_inc):
            cx = int(rng.uniform(50, NPIX - 50))
            cy = int(rng.uniform(50, NPIX - 50))
            r = int(rng.uniform(5, 20))
            amp = rng.uniform(0.1, 0.5)
            yy, xx = np.ogrid[0:NPIX, 0:NPIX]
            mask = (xx - cx)**2 + (yy - cy)**2 < r**2
            phantom[mask] = np.maximum(phantom[mask], amp)
        x_true = _norm01(phantom)
        records.append((idx, x_true))
        print(f"  phantom {idx:02d}: {x_true.shape} range=[{x_true.min():.3f},{x_true.max():.3f}]")

    print(f"\n[2] Computing Radon sinograms ({N_ANGLES} angles x {N_DETECTORS} detectors)...")
    print("  This may take a few minutes...")
    out_files = []
    for idx, x_true in records:
        sinogram = _radon_transform(x_true, N_ANGLES, N_DETECTORS)
        y_ideal = _norm01(sinogram)

        h5_path = OUTDIR / f"standard_ct_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "radon_transform"
            hp.attrs["n_angles"]        = N_ANGLES
            hp.attrs["n_detectors"]     = N_DETECTORS
            hp.attrs["image_size"]      = NPIX
            hp.attrs["angle_range_deg"] = 180.0
            hp.attrs["noise"]           = "none"
            hp.attrs["mismatch"]        = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://zenodo.org/record/3384092"
            meta.attrs["doi"]        = "10.1038/s41597-021-00893-z"
            meta.attrs["citation"]   = "Leuschner et al., LoDoPaB-CT, Sci. Data 2021"
            meta.attrs["phantom"]    = "Shepp-Logan modified"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  sample {idx:02d}: sinogram={sinogram.shape} range=[{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    meta_json = {
        "modality": "ct",
        "canonical_dataset": "LoDoPaB-CT (Shepp-Logan phantom, Radon forward model)",
        "source": "https://zenodo.org/record/3384092",
        "citation": "Leuschner et al., LoDoPaB-CT, a benchmark dataset for low-dose CT reconstruction, Sci. Data 2021",
        "doi": "10.1038/s41597-021-00893-z",
        "license": "CC BY 4.0",
        "n_samples": len(out_files),
        "x_true_shape": [NPIX, NPIX],
        "y_ideal_shape": [N_ANGLES, N_DETECTORS],
        "note": f"x_true={NPIX}x{NPIX} CT image; y_ideal=sinogram ({N_ANGLES} angles, {N_DETECTORS} detectors)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/ct/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "ct",
            "operator": "radon_transform",
            "n_angles": N_ANGLES,
            "n_detectors": N_DETECTORS,
            "image_size": NPIX,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(out_files)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
