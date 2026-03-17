#!/usr/bin/env python3
"""
Rebuild CASSI standard dataset using REAL KAIST data (not synthetic).

Source: KAIST 10 test scenes from TSA-Net mirror (already in _raw_src/).
Uses binarized KAIST mask with step=2 dispersion.

This should match reference GAP-TV PSNR (24.4 dB) and MST-L (34.9 dB).
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import scipy.io

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
SRCDIR = OUTDIR / "_raw_src"

N_BANDS = 28
STEP    = 2
NPIX    = 256
W_MEAS  = NPIX + (N_BANDS - 1) * STEP  # 310


def cassi_forward(x, mask, n_bands, step):
    """CASSI forward: y[i, j + k*step] += mask[i,j] * x[i,j,k]."""
    h, w, _ = x.shape
    w_ext = w + (n_bands - 1) * step
    y = np.zeros((h, w_ext), dtype=np.float64)
    for k in range(n_bands):
        y[:, k*step:k*step+w] += mask * x[:, :, k]
    return y.astype(np.float32)


def psnr(x_hat, x_true):
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64))**2)
    if mse < 1e-15: return float('inf')
    return float(10 * np.log10(float(x_true.max())**2 / mse))


def build():
    print("=" * 60)
    print("CASSI standard dataset — REAL KAIST data")
    print(f"  {NPIX}x{NPIX}x{N_BANDS}, step={STEP}, y={NPIX}x{W_MEAS}")
    print("=" * 60)

    # Load KAIST mask
    mask_path = SRCDIR / "mask.mat"
    if not mask_path.exists():
        print(f"  ERROR: {mask_path} not found")
        return False
    mat = scipy.io.loadmat(str(mask_path))
    mask_raw = mat["mask"].astype(np.float32)
    mask_bin = (mask_raw > 0.5).astype(np.float32)
    print(f"  Mask: {mask_raw.shape}, raw range [{mask_raw.min():.4f}, {mask_raw.max():.4f}]")
    print(f"  Binarized fill rate: {mask_bin.mean():.4f}")

    # Wavelengths (KAIST: 450-650nm, 28 bands)
    wavelengths = np.linspace(450, 650, N_BANDS).astype(np.float32)

    # Build samples from real scenes
    out_files = []
    for idx in range(10):
        scene_name = f"scene{idx+1:02d}"
        scene_path = SRCDIR / f"{scene_name}.mat"
        if not scene_path.exists():
            print(f"  SKIP {scene_name}: not found")
            continue

        mat = scipy.io.loadmat(str(scene_path))
        x_true = mat["img"].astype(np.float32)  # (256, 256, 28)

        # CASSI forward model (no normalization!)
        y_ideal = cassi_forward(x_true, mask_bin, N_BANDS, STEP)

        # Verify forward model
        y_check = cassi_forward(x_true, mask_bin, N_BANDS, STEP)
        fwd_err = np.abs(y_ideal - y_check).max()

        h5_path = OUTDIR / f"standard_cassi_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask", data=mask_bin, compression="gzip")
            f.create_dataset("wavelength", data=wavelengths, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"] = "cassi_spectral_dispersion"
            hp.attrs["n_bands"] = N_BANDS
            hp.attrs["step"] = STEP
            hp.attrs["image_size"] = NPIX
            hp.attrs["meas_width"] = W_MEAS
            hp.attrs["mask_fill"] = float(mask_bin.mean())
            hp.attrs["noise"] = "none"
            hp.attrs["y_normalization"] = "none"

            meta = f.create_group("metadata")
            meta.attrs["scene_name"] = scene_name
            meta.attrs["source"] = "KAIST Hyperspectral via TSA-Net mirror"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  {scene_name}: x={x_true.shape} [{x_true.min():.3f},{x_true.max():.3f}] "
              f"y={y_ideal.shape} [{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    # Write metadata
    meta_json = {
        "modality": "cassi",
        "canonical_dataset": "KAIST Hyperspectral - 10 scenes (REAL)",
        "source_repo": "https://github.com/mengziyi64/TSA-Net",
        "n_samples": len(out_files),
        "x_true_shape": [NPIX, NPIX, N_BANDS],
        "y_ideal_shape": [NPIX, W_MEAS],
        "n_bands": N_BANDS,
        "step": STEP,
        "mask_type": "binarized_kaist",
        "mask_fill": float(mask_bin.mean()),
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f_meta:
        json.dump(meta_json, f_meta, indent=2)
    with open(OUTDIR / "spec.json", "w") as f_spec:
        json.dump({
            "modality": "cassi",
            "operator": "cassi_spectral_dispersion",
            "n_bands": N_BANDS,
            "step": STEP,
            "image_size": NPIX,
            "meas_width": W_MEAS,
            "y_normalization": "none",
        }, f_spec, indent=2)

    print(f"\n  Built {len(out_files)} samples from REAL KAIST data")

    # Quick verification
    print("\n  Verifying on sample 00...")
    f = h5py.File(str(OUTDIR / "standard_cassi_00.h5"), 'r')
    y0 = np.array(f['y_ideal'], dtype=np.float32)
    x0 = np.array(f['x_true'], dtype=np.float32)
    m0 = np.array(f['mask'], dtype=np.float32)
    f.close()

    # Spectral correlation
    corrs = [np.corrcoef(x0[:,:,k].ravel(), x0[:,:,k+1].ravel())[0,1]
             for k in range(N_BANDS-1)]
    print(f"  Spectral correlation: mean={np.mean(corrs):.4f}")

    # Back-projection PSNR
    h, w = m0.shape
    w_ext = W_MEAS
    Phi_sum = np.zeros((h, w_ext), dtype=np.float32)
    for k in range(N_BANDS):
        Phi_sum[:, k*STEP:k*STEP+w] += m0**2
    Phi_sum = np.maximum(Phi_sum, 1e-6)
    x_bp = np.zeros_like(x0)
    for k in range(N_BANDS):
        x_bp[:,:,k] = m0 * y0[:, k*STEP:k*STEP+w] / Phi_sum[:, k*STEP:k*STEP+w]
    print(f"  Back-projection PSNR: {psnr(x_bp, x0):.2f} dB")

    # Oracle PSNR (mask=1 perfect, mask=0 zero)
    x_oracle = np.zeros_like(x0)
    for k in range(N_BANDS):
        x_oracle[:,:,k] = m0 * x0[:,:,k]
    print(f"  Oracle PSNR: {psnr(x_oracle, x0):.2f} dB")

    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
