#!/usr/bin/env python3
"""Generate synthetic SPC challenge data for 2026-W11.

Produces a test image and compressive single-pixel measurements using
Hadamard patterns at 25% sampling ratio.

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


def hadamard_matrix(n: int) -> np.ndarray:
    """Generate a Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    h_half = hadamard_matrix(n // 2)
    return np.block([[h_half, h_half], [h_half, -h_half]])


def generate_test_image(H: int = 64, W: int = 64, seed: int = 42) -> np.ndarray:
    """Generate a synthetic test image with geometric features."""
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    # Add geometric shapes
    for _ in range(4):
        cy, cx = rng.randint(10, H - 10), rng.randint(10, W - 10)
        r = rng.uniform(5, 15)
        intensity = rng.uniform(0.3, 1.0)
        circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        x[circle] = intensity
    # Add smooth background
    freq = rng.uniform(0.02, 0.08, size=2)
    bg = 0.15 * np.sin(2 * np.pi * freq[0] * yy) * np.cos(2 * np.pi * freq[1] * xx)
    x += bg
    x = np.clip(x, 0, 1)
    return x


def generate_spc_measurement(
    x_gt: np.ndarray, sampling_ratio: float = 0.25, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate single-pixel camera measurement with Hadamard patterns."""
    H, W = x_gt.shape
    N = H * W
    rng = np.random.RandomState(seed + 1)

    # Number of measurements
    M = int(N * sampling_ratio)

    # Use random subset of Hadamard rows (or random Gaussian if N is large)
    if N <= 4096:
        n_had = 1
        while n_had < N:
            n_had *= 2
        H_full = hadamard_matrix(n_had)[:N, :N] / np.sqrt(N)
        indices = rng.choice(N, M, replace=False)
        indices.sort()
        Phi = H_full[indices]
    else:
        # Gaussian random for larger sizes
        Phi = rng.randn(M, N) / np.sqrt(M)

    # Measurement
    x_vec = x_gt.flatten()
    y_clean = Phi @ x_vec

    # Poisson-like noise (scale to photon counts)
    max_photons = 1000.0
    y_scaled = np.abs(y_clean) * max_photons
    y_noisy_scaled = rng.poisson(np.clip(y_scaled, 0, None)).astype(np.float64)
    y_noisy = np.sign(y_clean) * y_noisy_scaled / max_photons

    return y_noisy, Phi


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def main():
    parser = argparse.ArgumentParser(description="Generate SPC challenge data")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating SPC data (seed={args.seed})...")
    x_gt = generate_test_image(H=64, W=64, seed=args.seed)
    y, Phi = generate_spc_measurement(x_gt, sampling_ratio=0.25, seed=args.seed)

    np.save(out / "x_gt.npy", x_gt)
    np.save(out / "y.npy", y)
    np.save(out / "Phi.npy", Phi)

    hashes = {
        "x_gt": sha256_file(out / "x_gt.npy"),
        "y": sha256_file(out / "y.npy"),
        "Phi": sha256_file(out / "Phi.npy"),
    }

    manifest = {
        "challenge": "2026-W11",
        "modality": "spc",
        "seed": args.seed,
        "dims": {"H": 64, "W": 64},
        "sampling_ratio": 0.25,
        "files": ["x_gt.npy", "y.npy", "Phi.npy"],
        "hashes": hashes,
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated data in {out}/")
    print(f"  x_gt: {x_gt.shape}, range [{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"  y:    {y.shape} ({len(y)} measurements)")
    print(f"  Phi:  {Phi.shape}")


if __name__ == "__main__":
    main()
