#!/usr/bin/env python3
"""Regenerate the CACTI public-tier H5 with InverseNet paper parameters.

Fixes two issues in the original dataset:
  1. Noise model: old used poisson_alpha=1.0 (~0 dB SNR).
     Paper uses peak_photon=10000 (~40 dB SNR).
  2. Forward model: old used continuous warped mask.
     Paper binarizes warped mask before computing measurement.

Mismatch parameters match validate_cacti_inversenet.py exactly:
  mask_dx=0.5, mask_dy=0.3, mask_rotation=0.1, mask_blur=0.0,
  gain_drift=1.02, offset_drift=0.002, clock_offset=0.05

Output: datasets/benchmark/cacti/public/cacti_challenge_public.h5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import shift, rotate as sp_rotate, gaussian_filter

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Try multiple known locations for CACTI simulation .mat files
_SIM_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "CACTI" / "simulation",
    PROJECT_ROOT / "repos" / "EfficientSCI" / "test_datasets" / "simulation",
    PROJECT_ROOT / "hisvit_temp" / "data" / "testdata" / "gray_256",
]
SIM_DIR = next((p for p in _SIM_CANDIDATES if p.exists()), _SIM_CANDIDATES[0])
OUT_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "cacti" / "public"
OUT_H5  = OUT_DIR / "cacti_challenge_public.h5"

# ── InverseNet paper parameters ──────────────────────────────────────────────
# From validate_cacti_inversenet.py lines 53-63
TRUE_SPEC = {
    "mask_dx": 0.50,
    "mask_dy": 0.30,
    "mask_rotation": 0.10,
    "mask_blur": 0.0,
    "clock_offset": 0.05,
    "gain_drift": 1.02,
    "offset_drift": 0.002,
}

SPEC_RANGES = [
    {"name": "mask_dx",       "min": 0.2,   "max": 0.8,  "unit": "px"},
    {"name": "mask_dy",       "min": 0.1,   "max": 0.5,  "unit": "px"},
    {"name": "mask_rotation", "min": 0.0,   "max": 0.3,  "unit": "deg"},
    {"name": "mask_blur",     "min": 0.0,   "max": 0.5,  "unit": "px"},
    {"name": "clock_offset",  "min": -0.1,  "max": 0.1,  "unit": "frames"},
    {"name": "gain_drift",    "min": 0.95,  "max": 1.05, "unit": ""},
    {"name": "offset_drift",  "min": -0.02, "max": 0.02, "unit": ""},
]

NOISE_PEAK = 10000   # photons at signal peak (~40 dB measurement SNR)
NOISE_SIGMA = 1.0    # Gaussian read noise std (in photon units)
SEED = 1001


# ── Physics ──────────────────────────────────────────────────────────────────

def warp_mask(mask: np.ndarray, dx, dy, rot_deg, blur_sigma=0.0):
    """Apply spatial mismatch to mask (H, W, T)."""
    H, W, T = mask.shape
    out = mask.copy()
    for t in range(T):
        m = out[:, :, t]
        m = shift(m, [dy, dx], order=1, mode='constant')
        if abs(rot_deg) > 1e-6:
            m = sp_rotate(m, rot_deg, reshape=False, order=1, mode='constant')
        if blur_sigma > 0:
            m = gaussian_filter(m, sigma=blur_sigma)
        out[:, :, t] = m
    return out


def add_noise(y, rng, peak=NOISE_PEAK, sigma=NOISE_SIGMA):
    """Poisson + Gaussian noise matching InverseNet paper (peak=10000, sigma=1.0).

    Scales measurement to peak photon count, applies Poisson shot noise,
    adds Gaussian read noise, then scales back.
    """
    y = np.maximum(y, 0).astype(np.float64)
    y_max = y.max()
    if y_max < 1e-10:
        return y.astype(np.float32)
    y_scaled = y / y_max * peak
    y_noisy = rng.poisson(np.maximum(y_scaled, 0).astype(np.int64)).astype(np.float64)
    y_noisy += rng.normal(0, sigma, y_noisy.shape)
    y_noisy = y_noisy / peak * y_max
    return np.maximum(y_noisy, 0).astype(np.float64)


def generate_measurement(x, mask, true_spec, rng):
    """Forward model: binarized warped mask + gain/offset + noise.

    Key difference from old code: mask is binarized after warping,
    matching InverseNet paper (validate_cacti_inversenet.py lines 199-200).
    """
    dx = true_spec["mask_dx"]
    dy = true_spec["mask_dy"]
    rot = true_spec["mask_rotation"]
    blur = true_spec["mask_blur"]
    gain = true_spec["gain_drift"]
    offset = true_spec["offset_drift"]

    mask_warped = warp_mask(mask, dx, dy, rot, blur)
    mask_warped_bin = (mask_warped > 0.5).astype(np.float64)  # binarize!

    y = gain * np.sum(x * mask_warped_bin, axis=2) + offset
    y = add_noise(y, rng)
    return y


# ── Data loading (same 20 samples as build_cacti_package.py) ─────────────────

def load_public_samples():
    """Load all T=8 measurement groups from 6 CACTI simulation scenes."""
    files = [
        ("kobe_cacti.mat", "kobe"),
        ("traffic_cacti.mat", "traffic"),
        ("runner8_cacti.mat", "runner"),
        ("drop8_cacti.mat", "drop"),
        ("crash32_cacti.mat", "crash"),
        ("aerial32_cacti.mat", "aerial"),
    ]
    T_mask = 8
    samples = []
    for fname, name in files:
        mat_path = SIM_DIR / fname
        if not mat_path.exists():
            print(f"  ERROR: {mat_path} not found")
            sys.exit(1)
        mat = sio.loadmat(str(mat_path))
        orig = np.array(mat["orig"], dtype=np.float64)
        mask = np.array(mat["mask"], dtype=np.float64)
        if orig.max() > 1.0:
            orig = orig / 255.0
        orig = np.clip(orig, 0, 1)
        mask = mask[:, :, :T_mask] / max(mask[:, :, :T_mask].max(), 1e-8)

        T_total = orig.shape[2]
        n_groups = T_total // T_mask
        print(f"  {name}: {T_total} frames -> {n_groups} group(s)")
        for g in range(n_groups):
            t0 = g * T_mask
            x = orig[:, :, t0 : t0 + T_mask]
            label = f"{name}_g{g}" if n_groups > 1 else name
            samples.append({"scene": label, "x": x, "mask": mask.copy()})
    return samples


# ── H5 generation ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Regenerating CACTI public H5 with InverseNet paper parameters")
    print("=" * 60)
    print(f"\nNoise model: peak_photon={NOISE_PEAK}, gaussian_sigma={NOISE_SIGMA}")
    print(f"True spec: {TRUE_SPEC}")
    print(f"Forward model: binarized warped mask")
    print(f"Output: {OUT_H5}\n")

    samples = load_public_samples()
    print(f"\nTotal: {len(samples)} samples\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    with h5py.File(str(OUT_H5), "w") as f:
        f.attrs["variant"] = "cacti"
        f.attrs["tier"] = "public"
        f.attrs["version"] = "3.0"
        f.attrs["noise_model"] = "poisson_gaussian"
        f.attrs["noise_params"] = json.dumps({
            "peak_photon": NOISE_PEAK,
            "gaussian_sigma_photon": NOISE_SIGMA,
        })

        for i, s in enumerate(samples):
            x = s["x"]
            mask = s["mask"]
            y = generate_measurement(x, mask, TRUE_SPEC, rng)

            H, W, T = x.shape
            print(f"  sample_{i:02d} ({s['scene']}): ({H},{W},{T})  "
                  f"y range [{y.min():.4f}, {y.max():.4f}]")

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("y", data=y, compression="gzip", compression_opts=4)
            grp.create_dataset("H_ideal", data=mask, compression="gzip",
                               compression_opts=4)
            grp.create_dataset("x_true", data=x, compression="gzip",
                               compression_opts=4)
            grp.attrs["spec_ranges"] = json.dumps(SPEC_RANGES)
            grp.attrs["true_spec"] = json.dumps(TRUE_SPEC)
            grp.attrs["metadata"] = json.dumps({
                "scene": s["scene"],
                "shape": list(x.shape),
                "T": T,
                "noise_model": "poisson_gaussian",
                "noise_params": {
                    "peak_photon": NOISE_PEAK,
                    "gaussian_sigma_photon": NOISE_SIGMA,
                },
                "source": "CACTI simulation",
                "forward_model": "binarized_warped_mask",
            })

    size_mb = OUT_H5.stat().st_size / 1024 / 1024
    print(f"\nDone! {OUT_H5} ({size_mb:.1f} MB)")
    print(f"  {len(samples)} samples, version 3.0")
    print(f"  Noise: peak_photon={NOISE_PEAK} (~40 dB SNR)")
    print(f"  Forward: binarized warped mask (matching InverseNet paper)")


if __name__ == "__main__":
    main()
