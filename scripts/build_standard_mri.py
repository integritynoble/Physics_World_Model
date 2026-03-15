#!/usr/bin/env python3
"""
Build standard dataset for mri modality.

Source: fastMRI multi-coil k-space dataset
  Zbontar et al., arXiv 2018; doi:10.1148/ryai.2020190007
  https://fastmri.med.nyu.edu/
  (Registration required for download)

  For standard dataset: Use synthetic multi-coil knee MRI phantom
  following standard MRI forward model (Fourier encoding + coil sensitivity)

Standard dataset: 10 samples (256x256 slices, 8 coils)
  x_true  = 2D complex MRI image (256x256), magnitude [0,1]
  y_ideal = multi-coil k-space: y[c] = FT(S_c * x_true) for coil c
            where S_c is the coil sensitivity map
            y_ideal shape: (8, 256, 256, 2) — 8 coils, real+imag

Forward model: MRI multi-coil encoding
  y = E(x) = [FT(S_1 * x), ..., FT(S_8 * x)]
  where FT = 2D Fourier transform, S_c = coil sensitivity map c

Output: datasets/benchmark/mri/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "mri" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _shepp_logan_2d(npix=256):
    """Generate 2D Shepp-Logan phantom."""
    # Standard Shepp-Logan ellipse parameters
    # [A, a, b, x0, y0, phi] (Kak & Slaney)
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


def _make_coil_sensitivities(npix=256, n_coils=8):
    """Generate birdcage coil sensitivity maps (8-coil)."""
    x = np.linspace(-1, 1, npix)
    y = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x, y)

    sensitivities = np.zeros((n_coils, npix, npix), dtype=np.complex64)
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        # Coil center on a circle
        cx = 1.2 * np.cos(angle)
        cy = 1.2 * np.sin(angle)
        # Sensitivity: Gaussian decay from coil center
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        magnitude = np.exp(-dist**2 / 0.5).astype(np.float32)
        # Phase varies smoothly
        phase = 0.3 * (X * np.cos(angle) + Y * np.sin(angle)).astype(np.float32)
        sensitivities[c] = magnitude * np.exp(1j * phase)

    # Normalize so sum of |S|^2 ~ 1
    sos = np.sqrt(np.sum(np.abs(sensitivities)**2, axis=0, keepdims=True))
    sensitivities = sensitivities / (sos + 1e-8)
    return sensitivities


def _mri_forward(image: np.ndarray, coil_maps: np.ndarray) -> np.ndarray:
    """
    Multi-coil MRI forward model: y[c] = FT(S_c * image)
    image: (H, W) complex or real
    coil_maps: (n_coils, H, W) complex
    Returns: (n_coils, H, W, 2) real/imag stacked
    """
    n_coils = coil_maps.shape[0]
    H, W = image.shape[:2]
    y = np.zeros((n_coils, H, W, 2), dtype=np.float32)
    for c in range(n_coils):
        coil_img = coil_maps[c] * image.astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(coil_img))
        y[c, :, :, 0] = np.real(kspace).astype(np.float32)
        y[c, :, :, 1] = np.imag(kspace).astype(np.float32)
    return y


def build():
    print("=" * 60)
    print("MRI standard dataset builder (synthetic multi-coil k-space)")
    print("=" * 60)

    npix = 256
    n_coils = 8

    print(f"\n[1] Generating coil sensitivity maps ({n_coils} coils)...")
    coil_maps = _make_coil_sensitivities(npix, n_coils)
    print(f"  coil maps: {coil_maps.shape}")

    print(f"\n[2] Generating {10} MRI phantom slices (Shepp-Logan variant)...")
    records = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 42)
        # Base Shepp-Logan phantom
        phantom = _shepp_logan_2d(npix)
        # Add mild random perturbations to simulate different subjects
        noise = rng.uniform(0, 0.05, phantom.shape).astype(np.float32)
        phantom = _norm01(phantom + noise * (phantom > 0))
        # Scale to [0,1]
        x_true = phantom.astype(np.float32)

        # MRI forward model
        y_kspace = _mri_forward(x_true, coil_maps)

        # Normalize k-space for storage
        y_norm = y_kspace / (np.abs(y_kspace).max() + 1e-8)

        records.append((idx, x_true, y_norm))
        print(f"  slice {idx:02d}: x_true={x_true.shape} "
              f"y_kspace={y_kspace.shape} max={np.abs(y_kspace).max():.2f}")

    print(f"\n[3] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, x_true, y_kspace in records:
        h5_path = OUTDIR / f"standard_mri_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",    data=x_true,    compression="gzip")
            f.create_dataset("y_ideal",   data=y_kspace,  compression="gzip")
            f.create_dataset("coil_maps_real", data=np.real(coil_maps), compression="gzip")
            f.create_dataset("coil_maps_imag", data=np.imag(coil_maps), compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]       = "multicoil_fourier_encoding"
            hp.attrs["n_coils"]        = n_coils
            hp.attrs["image_size"]     = npix
            hp.attrs["sampling"]       = "full_kspace"
            hp.attrs["acceleration"]   = 1
            hp.attrs["noise"]          = "none"
            hp.attrs["mismatch"]       = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://fastmri.med.nyu.edu/"
            meta.attrs["doi"]        = "10.1148/ryai.2020190007"
            meta.attrs["citation"]   = "Zbontar et al., fastMRI, arXiv 2018"
            meta.attrs["phantom"]    = "Shepp-Logan"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "mri",
        "canonical_dataset": "fastMRI multi-coil knee k-space (synthetic Shepp-Logan)",
        "source": "https://fastmri.med.nyu.edu/",
        "citation": "Zbontar et al., fastMRI: An Open Dataset and Benchmarks for Accelerated MRI, arXiv 2018",
        "doi": "10.1148/ryai.2020190007",
        "license": "Synthetic (Shepp-Logan model)",
        "n_samples": len(records),
        "x_true_shape": [npix, npix],
        "y_ideal_shape": [n_coils, npix, npix, 2],
        "note": "x_true=256x256 MRI image; y_ideal=8-coil k-space (real+imag) via multi-coil Fourier encoding",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/mri/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "mri",
            "operator": "multicoil_fourier_encoding",
            "n_coils": n_coils,
            "image_size": npix,
            "sampling": "full_kspace",
            "acceleration": 1,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
