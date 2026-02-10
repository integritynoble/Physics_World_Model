#!/usr/bin/env python3
"""Generate synthetic CT challenge data for 2026-W12.

Produces a phantom image and sparse-angle sinogram measurement.

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


def generate_phantom(N: int = 128, seed: int = 42) -> np.ndarray:
    """Generate a Shepp-Logan-like phantom with ellipses."""
    rng = np.random.RandomState(seed)
    phantom = np.zeros((N, N), dtype=np.float64)
    yy, xx = np.mgrid[0:N, 0:N]
    yy = (yy - N / 2) / (N / 2)
    xx = (xx - N / 2) / (N / 2)

    # Background ellipse
    mask = (xx ** 2 / 0.69 ** 2 + yy ** 2 / 0.92 ** 2) <= 1.0
    phantom[mask] = 1.0

    # Internal structures
    ellipses = [
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18, -0.2),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.2),
        (0.0, 0.35, 0.21, 0.25, 0, 0.1),
        (0.0, 0.1, 0.046, 0.046, 0, 0.1),
        (0.0, -0.1, 0.046, 0.046, 0, 0.1),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.1),
        (0.0, -0.605, 0.023, 0.023, 0, 0.1),
        (0.06, -0.605, 0.023, 0.046, 0, 0.1),
    ]

    for cx, cy, a, b, angle, intensity in ellipses:
        theta_rad = np.radians(angle)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_rot = cos_t * (xx - cx) + sin_t * (yy - cy)
        y_rot = -sin_t * (xx - cx) + cos_t * (yy - cy)
        mask = (x_rot ** 2 / (a ** 2 + 1e-10) + y_rot ** 2 / (b ** 2 + 1e-10)) <= 1.0
        phantom[mask] += intensity

    phantom = np.clip(phantom, 0, 1)
    return phantom


def parallel_beam_project(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Simple parallel-beam projection (sum along rays)."""
    N = image.shape[0]
    n_angles = len(angles)
    n_det = int(np.ceil(N * np.sqrt(2)))
    sinogram = np.zeros((n_angles, n_det), dtype=np.float64)

    center = N / 2.0
    det_center = n_det / 2.0

    for i, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for y_idx in range(N):
            for x_idx in range(N):
                # Project pixel center onto detector
                t = (x_idx - center) * cos_a + (y_idx - center) * sin_a
                det_idx = t + det_center
                d0 = int(np.floor(det_idx))
                d1 = d0 + 1
                w1 = det_idx - d0
                w0 = 1.0 - w1
                if 0 <= d0 < n_det:
                    sinogram[i, d0] += w0 * image[y_idx, x_idx]
                if 0 <= d1 < n_det:
                    sinogram[i, d1] += w1 * image[y_idx, x_idx]

    return sinogram


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def main():
    parser = argparse.ArgumentParser(description="Generate CT challenge data")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    N = 128
    n_angles = 30

    print(f"Generating CT data (seed={args.seed})...")
    x_gt = generate_phantom(N=N, seed=args.seed)

    # Sparse angles
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    y = parallel_beam_project(x_gt, angles)

    # Add Poisson noise
    rng = np.random.RandomState(args.seed + 1)
    max_photons = 500.0
    y_scaled = y * max_photons
    y_noisy = rng.poisson(np.clip(y_scaled, 0, None)).astype(np.float64) / max_photons

    np.save(out / "x_gt.npy", x_gt)
    np.save(out / "y.npy", y_noisy)
    np.save(out / "angles.npy", angles)

    hashes = {
        "x_gt": sha256_file(out / "x_gt.npy"),
        "y": sha256_file(out / "y.npy"),
        "angles": sha256_file(out / "angles.npy"),
    }

    manifest = {
        "challenge": "2026-W12",
        "modality": "ct",
        "seed": args.seed,
        "dims": {"H": N, "W": N},
        "n_angles": n_angles,
        "files": ["x_gt.npy", "y.npy", "angles.npy"],
        "hashes": hashes,
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated data in {out}/")
    print(f"  x_gt:   {x_gt.shape}, range [{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"  y:      {y_noisy.shape}")
    print(f"  angles: {len(angles)} projections")


if __name__ == "__main__":
    main()
