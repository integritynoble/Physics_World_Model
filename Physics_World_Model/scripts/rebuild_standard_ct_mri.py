#!/usr/bin/env python3
"""
Rebuild standard datasets for CT and MRI with correct physics.

KEY FIX: y_ideal must NOT be normalized to [0,1].
It must satisfy the exact physics relation: y = A(x_true).
Normalization destroys the inverse relationship that solvers rely on.

CT:  y = Radon(x_true)     — sinogram in raw line-integral units
MRI: y = FFT(x_true)       — k-space data (complex stored as real+imag)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent

# ============================================================
#  CT DATASET
# ============================================================
CT_DIR = ROOT / "datasets" / "benchmark" / "ct" / "standard"

def _shepp_logan_2d(npix=256):
    """Generate 2D Shepp-Logan phantom."""
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


def build_ct():
    """Build CT standard dataset with CORRECT physics (no y normalization)."""
    from skimage.transform import radon

    CT_DIR.mkdir(parents=True, exist_ok=True)
    npix = 256
    n_angles = 180  # standard: 180 angles over 180 degrees
    angles_deg = np.linspace(0, 180, n_angles, endpoint=False)

    print("=" * 60)
    print("CT standard dataset (Shepp-Logan + skimage Radon)")
    print(f"  {npix}x{npix} phantom, {n_angles} angles, circle=True")
    print("  y_ideal = RAW sinogram (NOT normalized)")
    print("=" * 60)

    out_files = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 7)
        phantom = _shepp_logan_2d(npix)
        # Add tissue-like variations
        noise = rng.uniform(0, 0.02, phantom.shape).astype(np.float32)
        phantom = phantom + noise * (phantom > 0)
        # Add small random inclusions
        n_inc = rng.integers(2, 6)
        for _ in range(n_inc):
            cx = int(rng.uniform(30, npix - 30))
            cy = int(rng.uniform(30, npix - 30))
            r = int(rng.uniform(5, 15))
            amp = rng.uniform(0.1, 0.5)
            yy, xx = np.ogrid[0:npix, 0:npix]
            mask = (xx - cx)**2 + (yy - cy)**2 < r**2
            phantom[mask] = np.maximum(phantom[mask], amp)

        # Normalize x_true to [0, 1]
        x_true = phantom.copy()
        lo, hi = x_true.min(), x_true.max()
        x_true = ((x_true - lo) / (hi - lo + 1e-12)).astype(np.float32)

        # Compute sinogram using skimage (CORRECT forward model)
        # radon returns (n_detectors, n_angles) — we store (n_angles, n_detectors)
        sino_raw = radon(x_true, theta=angles_deg, circle=True)  # (npix, n_angles)
        y_ideal = sino_raw.T.astype(np.float32)  # (n_angles, npix) = (n_angles, n_detectors)

        # Verify: iradon should recover x_true
        from skimage.transform import iradon
        x_check = iradon(sino_raw, theta=angles_deg, circle=True, output_size=npix)
        mse = np.mean((x_check - x_true)**2)
        psnr = 10 * np.log10(x_true.max()**2 / mse) if mse > 0 else float('inf')

        h5_path = CT_DIR / f"standard_ct_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "radon_transform"
            hp.attrs["n_angles"]        = n_angles
            hp.attrs["n_detectors"]     = npix  # same as image size for circle=True
            hp.attrs["image_size"]      = npix
            hp.attrs["angles_deg"]      = angles_deg.tolist()
            hp.attrs["circle"]          = True
            hp.attrs["sinogram_format"] = "(n_angles, n_detectors)"
            hp.attrs["noise"]           = "none"
            hp.attrs["y_normalization"] = "none"  # RAW sinogram

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://zenodo.org/record/3384092"
            meta.attrs["citation"]   = "Leuschner et al., LoDoPaB-CT, Sci. Data 2021"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  CT {idx:02d}: x_true {x_true.shape} | sinogram {y_ideal.shape} "
              f"range=[{y_ideal.min():.2f},{y_ideal.max():.2f}] | "
              f"verify PSNR={psnr:.1f} dB")

    # Write metadata
    meta_json = {
        "modality": "ct",
        "canonical_dataset": "LoDoPaB-CT (Shepp-Logan phantom, skimage Radon)",
        "n_samples": len(out_files),
        "x_true_shape": [npix, npix],
        "y_ideal_shape": [n_angles, npix],
        "y_normalization": "none",
        "note": f"x_true={npix}x{npix}; y_ideal=sinogram ({n_angles} angles, {npix} detectors). "
                f"NO normalization on y — raw line integrals.",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "files": out_files,
    }
    with open(CT_DIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(CT_DIR / "spec.json", "w") as f:
        json.dump({
            "modality": "ct",
            "operator": "radon_transform",
            "n_angles": n_angles,
            "n_detectors": npix,
            "image_size": npix,
            "circle": True,
            "noise": "none",
            "y_normalization": "none",
        }, f, indent=2)

    print(f"\n  CT: {len(out_files)} samples written to {CT_DIR}")
    return True


# ============================================================
#  MRI DATASET
# ============================================================
MRI_DIR = ROOT / "datasets" / "benchmark" / "mri" / "standard"

def _make_sampling_mask(npix, acceleration=4, center_fraction=0.08, seed=42):
    """Create Cartesian undersampling mask for MRI."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((npix, npix), dtype=np.float32)

    # Always sample center lines (ACS region)
    center = int(npix * center_fraction / 2)
    mask[npix//2 - center : npix//2 + center, :] = 1

    # Randomly sample remaining lines
    n_center = 2 * center
    n_remaining = max(0, npix // acceleration - n_center)
    available = list(range(0, npix//2 - center)) + list(range(npix//2 + center, npix))
    if n_remaining > 0 and available:
        chosen = rng.choice(available, size=min(n_remaining, len(available)), replace=False)
        mask[chosen, :] = 1

    return mask


def build_mri():
    """Build MRI standard dataset with CORRECT physics (no y normalization)."""
    MRI_DIR.mkdir(parents=True, exist_ok=True)
    npix = 256
    acceleration = 4  # 4x undersampled

    print("\n" + "=" * 60)
    print("MRI standard dataset (Shepp-Logan + FFT k-space)")
    print(f"  {npix}x{npix} phantom, {acceleration}x acceleration")
    print("  y_ideal = complex k-space (real+imag channels)")
    print("  NO normalization on y — raw Fourier coefficients")
    print("=" * 60)

    # Create sampling mask
    mask = _make_sampling_mask(npix, acceleration)
    print(f"  Sampling mask: {mask.shape}, {mask.mean()*100:.1f}% sampled")

    out_files = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 42)
        phantom = _shepp_logan_2d(npix)
        noise = rng.uniform(0, 0.03, phantom.shape).astype(np.float32)
        phantom = phantom + noise * (phantom > 0)

        # Normalize x_true to [0, 1]
        lo, hi = phantom.min(), phantom.max()
        x_true = ((phantom - lo) / (hi - lo + 1e-12)).astype(np.float32)

        # Forward model: k-space = FFT2(x_true)
        kspace_full = np.fft.fftshift(np.fft.fft2(x_true)).astype(np.complex64)

        # Undersampled k-space
        kspace_us = kspace_full * mask

        # Store as real+imag channels: (256, 256, 2)
        y_ideal = np.stack([kspace_us.real, kspace_us.imag], axis=-1).astype(np.float32)

        # Verify: zero-filled reconstruction
        x_zf = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us)))
        mse = np.mean((x_zf - x_true)**2)
        psnr_zf = 10 * np.log10(x_true.max()**2 / mse) if mse > 0 else float('inf')

        # Verify: full k-space reconstruction
        x_full = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_full)))
        mse_full = np.mean((x_full - x_true)**2)
        psnr_full = 10 * np.log10(x_true.max()**2 / mse_full) if mse_full > 0 else float('inf')

        h5_path = MRI_DIR / f"standard_mri_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("sampling_mask", data=mask, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "fourier_encoding"
            hp.attrs["image_size"]      = npix
            hp.attrs["acceleration"]    = acceleration
            hp.attrs["center_fraction"] = 0.08
            hp.attrs["kspace_format"]   = "(H, W, 2) real+imag"
            hp.attrs["noise"]           = "none"
            hp.attrs["y_normalization"] = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://fastmri.med.nyu.edu/"
            meta.attrs["citation"]   = "Zbontar et al., fastMRI, arXiv 2018"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  MRI {idx:02d}: x_true {x_true.shape} | kspace {y_ideal.shape} "
              f"| ZF PSNR={psnr_zf:.1f} dB | full PSNR={psnr_full:.1f} dB")

    meta_json = {
        "modality": "mri",
        "canonical_dataset": "fastMRI knee (synthetic Shepp-Logan, 4x acceleration)",
        "n_samples": len(out_files),
        "x_true_shape": [npix, npix],
        "y_ideal_shape": [npix, npix, 2],
        "y_normalization": "none",
        "note": f"x_true={npix}x{npix}; y_ideal=undersampled k-space (real+imag). "
                f"mask stored separately. NO normalization on y.",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "files": out_files,
    }
    with open(MRI_DIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(MRI_DIR / "spec.json", "w") as f:
        json.dump({
            "modality": "mri",
            "operator": "fourier_encoding",
            "image_size": npix,
            "acceleration": acceleration,
            "center_fraction": 0.08,
            "kspace_format": "(H, W, 2) real+imag",
            "noise": "none",
            "y_normalization": "none",
        }, f, indent=2)

    print(f"\n  MRI: {len(out_files)} samples written to {MRI_DIR}")
    return True


if __name__ == "__main__":
    ok = build_ct()
    ok = build_mri() and ok
    print("\n" + "=" * 60)
    print("DONE" if ok else "FAILED")
    sys.exit(0 if ok else 1)
