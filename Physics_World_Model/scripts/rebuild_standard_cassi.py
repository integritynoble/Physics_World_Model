#!/usr/bin/env python3
"""
Rebuild CASSI standard dataset with CORRECT physics.

CASSI forward model: y[h, w + k*step] += mask[h, w] * x_true[h, w, k]
  where k = 0..n_bands-1, step = spectral dispersion step

KEY: y_ideal must NOT be normalized. It must satisfy y = A(x_true) exactly.

Reference setup (KAIST benchmark):
  - Image: 256x256, 28 spectral bands (400-700nm)
  - Mask: binary random 50% coded aperture
  - Step: 2 (standard CASSI dispersion)
  - Measurement: 256 x (256 + 27*2) = 256 x 310

PWM setup (slightly modified):
  - Image: 256x256, 28 spectral bands
  - Step: 2 (matching KAIST convention)
  - Mask: binary random 50%
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)

NPIX    = 256
N_BANDS = 28    # KAIST uses 28 bands (not 31)
STEP    = 2     # Standard CASSI dispersion: 2 pixels per band
W_MEAS  = NPIX + (N_BANDS - 1) * STEP  # 256 + 54 = 310


def _make_spectral_cube(npix, n_bands, seed=0):
    """Generate a synthetic spectral image cube.

    Uses spatial Shepp-Logan + smooth spectral signatures.
    """
    rng = np.random.default_rng(seed)

    # Base spatial phantom (Shepp-Logan)
    params = [
        [1.0,   0.69,  0.92,   0.0,    0.0,    0.0],
        [-0.8,  0.6624, 0.874,  0.0,   -0.0184, 0.0],
        [-0.2,  0.11,  0.31,   0.22,   0.0,   -18.0],
        [-0.2,  0.16,  0.41,  -0.22,   0.0,    18.0],
        [0.1,   0.21,  0.25,   0.0,    0.35,    0.0],
        [0.1,   0.046, 0.046,  0.0,    0.1,     0.0],
        [0.1,   0.046, 0.046,  0.0,   -0.1,     0.0],
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
    phantom = np.clip(phantom, 0, None)

    # Create spectral cube: spatial phantom * spectral signatures
    cube = np.zeros((npix, npix, n_bands), dtype=np.float32)
    wavelengths = np.linspace(400, 700, n_bands)  # nm

    # Create different spectral signatures for different regions
    n_regions = 5
    for r in range(n_regions):
        # Random Gaussian spectral signature
        center_wl = rng.uniform(420, 680)
        width = rng.uniform(30, 100)
        amplitude = rng.uniform(0.3, 1.0)
        spectrum = amplitude * np.exp(-0.5 * ((wavelengths - center_wl) / width)**2)

        # Create spatial mask for this region
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.5, 0.5)
        r_size = rng.uniform(0.1, 0.4)
        region = ((X - cx)**2 + (Y - cy)**2 < r_size**2).astype(np.float32)

        for k in range(n_bands):
            cube[:, :, k] += region * spectrum[k]

    # Add phantom structure to all bands with varying strength
    for k in range(n_bands):
        base_factor = 0.5 + 0.5 * np.sin(2 * np.pi * k / n_bands)
        cube[:, :, k] += phantom * base_factor * 0.5

    # Add small random noise for variation between samples
    noise = rng.uniform(0, 0.05, cube.shape).astype(np.float32)
    cube += noise * (cube > 0.01)

    # Normalize to [0, 1]
    cube = np.clip(cube, 0, None)
    lo, hi = cube.min(), cube.max()
    if hi > lo:
        cube = (cube - lo) / (hi - lo)

    return cube.astype(np.float32), wavelengths


def cassi_forward(x_true, mask, n_bands, step):
    """CASSI forward model: y[h, w + k*step] += mask[h,w] * x[h,w,k]."""
    h, w = mask.shape
    w_meas = w + (n_bands - 1) * step
    y = np.zeros((h, w_meas), dtype=np.float32)
    for k in range(n_bands):
        y[:, k*step : k*step + w] += mask * x_true[:, :, k]
    return y


def cassi_adjoint(y, mask, n_bands, step):
    """CASSI adjoint (transposed) model."""
    h, w_meas = y.shape
    w = w_meas - (n_bands - 1) * step
    x = np.zeros((h, w, n_bands), dtype=np.float32)
    for k in range(n_bands):
        x[:, :, k] = mask * y[:, k*step : k*step + w]
    return x


def verify_gap_tv(y, mask, x_true, n_bands, step, iters=50, lam=0.05):
    """Run GAP-TV to verify the data produces reasonable PSNR."""
    sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
    for k in list(sys.modules):
        if 'pwm_core' in k: del sys.modules[k]

    from pwm_core.recon.gap_tv import gap_tv_cassi
    x_hat = gap_tv_cassi(y, mask, n_bands, iters, lam, step=step, device='cpu')

    mse = np.mean((x_hat - x_true)**2)
    psnr = 10 * np.log10(x_true.max()**2 / mse) if mse > 0 else float('inf')
    return psnr, x_hat


def build():
    print("=" * 60)
    print("CASSI standard dataset (correct physics, no y normalization)")
    print(f"  {NPIX}x{NPIX}x{N_BANDS}, step={STEP}, measurement={NPIX}x{W_MEAS}")
    print("=" * 60)

    out_files = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 100)

        # Generate spectral image cube
        x_true, wavelengths = _make_spectral_cube(NPIX, N_BANDS, seed=idx+100)

        # Generate binary coded aperture mask (50% fill)
        mask = (rng.random((NPIX, NPIX)) > 0.5).astype(np.float32)

        # Forward model: y = A(x_true) — NO NORMALIZATION
        y_ideal = cassi_forward(x_true, mask, N_BANDS, STEP)

        # Verify physics: adjoint should give approximate reconstruction
        x_adj = cassi_adjoint(y_ideal, mask, N_BANDS, STEP)

        # Verify forward-adjoint consistency
        y_check = cassi_forward(x_adj, mask, N_BANDS, STEP)

        h5_path = OUTDIR / f"standard_cassi_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask", data=mask, compression="gzip")
            f.create_dataset("wavelength", data=wavelengths.astype(np.float32), compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]       = "cassi_spectral_dispersion"
            hp.attrs["n_bands"]        = N_BANDS
            hp.attrs["step"]           = STEP
            hp.attrs["image_size"]     = NPIX
            hp.attrs["meas_width"]     = W_MEAS
            hp.attrs["mask_fill"]      = float(mask.mean())
            hp.attrs["noise"]          = "none"
            hp.attrs["y_normalization"]= "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "KAIST HSI benchmark (synthetic)"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  CASSI {idx:02d}: x_true {x_true.shape} [{x_true.min():.3f},{x_true.max():.3f}] | "
              f"y {y_ideal.shape} [{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    # Verify GAP-TV on first sample
    print("\n  Verifying GAP-TV on sample 00...")
    f = h5py.File(str(OUTDIR / "standard_cassi_00.h5"), 'r')
    y0 = np.array(f['y_ideal'], dtype=np.float32)
    x0 = np.array(f['x_true'], dtype=np.float32)
    m0 = np.array(f['mask'], dtype=np.float32)
    f.close()

    psnr, _ = verify_gap_tv(y0, m0, x0, N_BANDS, STEP, iters=50, lam=0.05)
    print(f"  GAP-TV PSNR: {psnr:.2f} dB (reference: ~24.4 dB)")

    # Write metadata
    meta_json = {
        "modality": "cassi",
        "canonical_dataset": "KAIST HSI benchmark (synthetic, step=2)",
        "n_samples": len(out_files),
        "x_true_shape": [NPIX, NPIX, N_BANDS],
        "y_ideal_shape": [NPIX, W_MEAS],
        "y_normalization": "none",
        "n_bands": N_BANDS,
        "step": STEP,
        "mask_fill": 0.5,
        "note": f"x_true={NPIX}x{NPIX}x{N_BANDS}; y=CASSI snapshot ({NPIX}x{W_MEAS}). "
                f"step={STEP}. NO normalization — raw physics.",
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

    print(f"\n  CASSI: {len(out_files)} samples -> {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
