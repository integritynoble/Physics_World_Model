#!/usr/bin/env python3
"""Generate synthetic CASSI challenge data for 2026-W10.

Produces a small hyperspectral test cube and coded-aperture snapshot measurement.
All data is generated on-the-fly from NumPy operations -- no large files are
committed to the repository.

Usage:
    python generate_data.py --output ./data
    python generate_data.py --output ./data --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def generate_spectral_cube(
    H: int = 64, W: int = 64, L: int = 8, seed: int = 42
) -> np.ndarray:
    """Generate a synthetic hyperspectral cube with smooth spectral features."""
    rng = np.random.RandomState(seed)
    # Create a base image with smooth spatial structure
    x = np.zeros((H, W, L), dtype=np.float64)
    # Add a few Gaussian blobs at different spectral bands
    for _ in range(5):
        cy, cx = rng.randint(10, H - 10), rng.randint(10, W - 10)
        sigma_s = rng.uniform(5, 15)
        yy, xx = np.mgrid[0:H, 0:W]
        spatial = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_s ** 2))
        # Smooth spectral profile
        center_l = rng.uniform(1, L - 2)
        sigma_l = rng.uniform(1, 3)
        spectral = np.exp(-((np.arange(L) - center_l) ** 2) / (2 * sigma_l ** 2))
        intensity = rng.uniform(0.3, 1.0)
        x += intensity * spatial[:, :, None] * spectral[None, None, :]
    # Normalize to [0, 1]
    x = x / (x.max() + 1e-8)
    return x


def generate_cassi_measurement(
    x_gt: np.ndarray, seed: int = 42, read_noise_sigma: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate CASSI coded-aperture snapshot measurement.

    Returns (y, mask) where y is the 2D detector measurement.
    """
    H, W, L = x_gt.shape
    rng = np.random.RandomState(seed + 1)
    # Generate binary coded aperture mask
    mask = (rng.rand(H, W) > 0.5).astype(np.float64)
    # Simulate spectral dispersion: shift each band by its index
    y = np.zeros((H, W + L - 1), dtype=np.float64)
    for l_idx in range(L):
        shifted = np.zeros((H, W + L - 1), dtype=np.float64)
        shifted[:, l_idx : l_idx + W] = mask * x_gt[:, :, l_idx]
        y += shifted
    # Scale to photon counts and add noise
    max_photons = 800.0
    y_scaled = y * max_photons
    # Poisson shot noise
    y_noisy = rng.poisson(np.clip(y_scaled, 0, None)).astype(np.float64)
    # Read noise
    y_noisy += rng.normal(0, read_noise_sigma, y_noisy.shape)
    # Normalize back
    y_noisy = y_noisy / max_photons
    return y_noisy, mask


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def main():
    parser = argparse.ArgumentParser(description="Generate CASSI challenge data")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating CASSI data (seed={args.seed})...")
    x_gt = generate_spectral_cube(H=64, W=64, L=8, seed=args.seed)
    y, mask = generate_cassi_measurement(x_gt, seed=args.seed)

    # Save arrays
    np.save(out / "x_gt.npy", x_gt)
    np.save(out / "y.npy", y)
    np.save(out / "mask.npy", mask)

    # Compute hashes
    hashes = {
        "x_gt": sha256_file(out / "x_gt.npy"),
        "y": sha256_file(out / "y.npy"),
        "mask": sha256_file(out / "mask.npy"),
    }

    manifest = {
        "challenge": "2026-W10",
        "modality": "cassi",
        "seed": args.seed,
        "dims": {"H": 64, "W": 64, "L": 8},
        "files": ["x_gt.npy", "y.npy", "mask.npy"],
        "hashes": hashes,
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated data in {out}/")
    print(f"  x_gt: {x_gt.shape}, range [{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"  y:    {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")
    print(f"  mask: {mask.shape}, {mask.mean():.1%} fill")


if __name__ == "__main__":
    main()
