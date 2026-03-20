#!/usr/bin/env python3
"""Generate synthetic widefield low-dose challenge data for 2026-W13.

Produces a fluorescence test image and a blurred, noisy measurement with
low photon count and high background.

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
from numpy.fft import fft2, ifft2, fftshift


def generate_fluorescence_image(
    H: int = 128, W: int = 128, seed: int = 42
) -> np.ndarray:
    """Generate a synthetic fluorescence microscopy image."""
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]

    # Simulate fluorescent beads / structures
    n_beads = 15
    for _ in range(n_beads):
        cy, cx = rng.randint(5, H - 5), rng.randint(5, W - 5)
        sigma = rng.uniform(1.5, 4.0)
        intensity = rng.uniform(0.3, 1.0)
        bead = intensity * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        )
        x += bead

    # Add filamentary structures
    for _ in range(3):
        t = np.linspace(0, 1, 200)
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(30, 60)
        width = rng.uniform(1.5, 3.0)
        intensity = rng.uniform(0.2, 0.6)
        for ti in t:
            py = int(cy + length * ti * np.sin(angle))
            px = int(cx + length * ti * np.cos(angle))
            if 0 <= py < H and 0 <= px < W:
                r2 = (yy - py) ** 2 + (xx - px) ** 2
                x += intensity * 0.01 * np.exp(-r2 / (2 * width ** 2))

    x = np.clip(x, 0, None)
    x = x / (x.max() + 1e-8)
    return x


def gaussian_psf(H: int, W: int, sigma: float = 2.5) -> np.ndarray:
    """Generate a 2D Gaussian PSF."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    psf = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    psf = psf / psf.sum()
    return psf


def convolve_fft(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve image with PSF using FFT."""
    return np.real(ifft2(fft2(image) * fft2(fftshift(psf))))


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate widefield low-dose challenge data"
    )
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    H, W = 128, 128
    psf_sigma = 2.5
    max_photons = 100.0  # Low photon budget
    bg_level = 0.15  # High background

    print(f"Generating widefield low-dose data (seed={args.seed})...")
    x_gt = generate_fluorescence_image(H=H, W=W, seed=args.seed)
    psf = gaussian_psf(H, W, sigma=psf_sigma)

    # Forward model: convolve + background + noise
    y_clean = convolve_fft(x_gt, psf) + bg_level

    # Poisson noise
    rng = np.random.RandomState(args.seed + 1)
    y_scaled = y_clean * max_photons
    y_noisy = rng.poisson(np.clip(y_scaled, 0, None)).astype(np.float64) / max_photons

    # Read noise
    read_noise_sigma = 3.0
    y_noisy += rng.normal(0, read_noise_sigma / max_photons, y_noisy.shape)

    np.save(out / "x_gt.npy", x_gt)
    np.save(out / "y.npy", y_noisy)
    np.save(out / "psf.npy", psf)

    hashes = {
        "x_gt": sha256_file(out / "x_gt.npy"),
        "y": sha256_file(out / "y.npy"),
        "psf": sha256_file(out / "psf.npy"),
    }

    manifest = {
        "challenge": "2026-W13",
        "modality": "widefield",
        "seed": args.seed,
        "dims": {"H": H, "W": W},
        "psf_sigma": psf_sigma,
        "max_photons": max_photons,
        "bg_level": bg_level,
        "files": ["x_gt.npy", "y.npy", "psf.npy"],
        "hashes": hashes,
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated data in {out}/")
    print(f"  x_gt: {x_gt.shape}, range [{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"  y:    {y_noisy.shape}, range [{y_noisy.min():.3f}, {y_noisy.max():.3f}]")
    print(f"  psf:  {psf.shape}, sigma={psf_sigma}")


if __name__ == "__main__":
    main()
