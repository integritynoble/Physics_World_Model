#!/usr/bin/env python3
"""Generate complex synthetic scenes for SPC-Kronecker dev/hidden tiers.

Creates unique, non-traceable 256x256 grayscale images by:
  1. Multi-source blending  — mix 3-5 random BSDS400 crops with random weights
  2. Perlin-like noise      — multi-octave gradient noise for natural texture
  3. Geometric features     — edges, curves, fine-grained structure
  4. Frequency manipulation — swap/blend frequency bands across sources
  5. Elastic warping        — non-linear spatial distortion
  6. Non-linear grading     — local contrast, gamma, histogram shaping

Each image is deterministically generated from a seed, reproducible but unique.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    uniform_filter,
)
from scipy.signal import fftconvolve


# ── Perlin-like noise generator ──────────────────────────────────────────────

def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin_2d(shape: tuple[int, int], res: tuple[int, int],
               rng: np.random.RandomState) -> np.ndarray:
    """Generate a single octave of Perlin-like noise."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Random gradient vectors at grid points
    angles = 2 * np.pi * rng.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Tile indices
    t = np.arange(shape[0]) // d
    s = np.arange(shape[1]) // d

    g00 = gradients[t[:, None], s[None, :]]
    g10 = gradients[t[:, None] + 1, s[None, :]]
    g01 = gradients[t[:, None], s[None, :] + 1]
    g11 = gradients[t[:, None] + 1, s[None, :] + 1]

    # Dot products
    n00 = np.sum(grid * g00, axis=-1)
    n10 = np.sum((grid - np.array([1, 0])) * g10, axis=-1)
    n01 = np.sum((grid - np.array([0, 1])) * g01, axis=-1)
    n11 = np.sum((grid - np.array([1, 1])) * g11, axis=-1)

    fx = _fade(grid[:, :, 0])
    fy = _fade(grid[:, :, 1])

    n0 = n00 * (1 - fx) + n10 * fx
    n1 = n01 * (1 - fx) + n11 * fx
    return n0 * (1 - fy) + n1 * fy


def multi_octave_noise(shape: tuple[int, int], rng: np.random.RandomState,
                       octaves: int = 6, persistence: float = 0.5) -> np.ndarray:
    """Multi-octave Perlin noise with fractal detail."""
    result = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    freq = 4  # Starting frequency

    for _ in range(octaves):
        if freq > min(shape):
            break
        try:
            noise = _perlin_2d(shape, (freq, freq), rng)
            result += amplitude * noise
        except (ValueError, IndexError):
            # Fallback: Gaussian-filtered random noise
            raw = rng.randn(*shape)
            sigma = max(1, shape[0] / freq / 2)
            result += amplitude * gaussian_filter(raw, sigma=sigma)
        total_amp += amplitude
        amplitude *= persistence
        freq *= 2

    return result / max(total_amp, 1e-8)


# ── Source image loading ─────────────────────────────────────────────────────

def _load_source_images(source_dir: Path, fmt: str = "jpg",
                        max_images: int = 400) -> list[np.ndarray]:
    """Load source images as float64 grayscale arrays."""
    ext_map = {"jpg": ["*.jpg", "*.jpeg"], "png": ["*.png"], "tif": ["*.tif", "*.tiff"]}
    patterns = ext_map.get(fmt, [f"*.{fmt}"])

    files = []
    for pat in patterns:
        files.extend(sorted(source_dir.glob(pat)))

    images = []
    for f in files[:max_images]:
        try:
            img = Image.open(f).convert("L")
            arr = np.array(img, dtype=np.float64) / 255.0
            images.append(arr)
        except Exception:
            continue
    return images


def _random_crop(img: np.ndarray, size: int,
                 rng: np.random.RandomState) -> np.ndarray:
    """Extract a random square crop from an image, resizing if needed."""
    h, w = img.shape
    if h < size or w < size:
        # Resize up
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        scale = max(size / h, size / w) * 1.1
        new_h, new_w = int(h * scale), int(w * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img = np.array(pil_img, dtype=np.float64) / 255.0
        h, w = img.shape

    y0 = rng.randint(0, h - size + 1)
    x0 = rng.randint(0, w - size + 1)
    return img[y0:y0 + size, x0:x0 + size]


# ── Frequency-domain mixing ─────────────────────────────────────────────────

def _frequency_blend(imgs: list[np.ndarray], weights: np.ndarray,
                     rng: np.random.RandomState) -> np.ndarray:
    """Blend images in frequency domain with random band mixing."""
    h, w = imgs[0].shape
    result_fft = np.zeros((h, w), dtype=np.complex128)

    # Create frequency distance map
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq_dist = np.sqrt(fy ** 2 + fx ** 2)

    # Random frequency band boundaries
    n_bands = 4
    boundaries = sorted(rng.uniform(0.01, 0.5, n_bands - 1))
    boundaries = [0.0] + list(boundaries) + [1.0]

    for band_idx in range(n_bands):
        lo, hi = boundaries[band_idx], boundaries[band_idx + 1]
        mask = ((freq_dist >= lo) & (freq_dist < hi)).astype(np.float64)
        # Smooth the mask edges
        mask = gaussian_filter(mask, sigma=1)

        # Pick a weighted random source for this band
        band_weights = rng.dirichlet(weights * 3 + 0.1)
        band_fft = np.zeros((h, w), dtype=np.complex128)
        for i, img in enumerate(imgs):
            band_fft += band_weights[i] * np.fft.fft2(img)
        result_fft += mask * band_fft

    result = np.real(np.fft.ifft2(result_fft))
    return result


# ── Elastic warping ──────────────────────────────────────────────────────────

def _elastic_warp(img: np.ndarray, rng: np.random.RandomState,
                  alpha: float = 20.0, sigma: float = 5.0) -> np.ndarray:
    """Apply elastic deformation to an image."""
    h, w = img.shape
    dx = gaussian_filter(rng.randn(h, w) * alpha, sigma=sigma)
    dy = gaussian_filter(rng.randn(h, w) * alpha, sigma=sigma)

    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    return map_coordinates(img, coords, order=3, mode="reflect")


# ── Swirl distortion ─────────────────────────────────────────────────────────

def _swirl(img: np.ndarray, rng: np.random.RandomState,
           strength: float = 2.0, radius: float = 80.0) -> np.ndarray:
    """Apply a localized swirl distortion."""
    h, w = img.shape
    cy, cx = rng.randint(h // 4, 3 * h // 4), rng.randint(w // 4, 3 * w // 4)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dy, dx = y - cy, x - cx
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = strength * np.exp(-r ** 2 / (2 * radius ** 2))
    new_x = cx + dx * np.cos(theta) - dy * np.sin(theta)
    new_y = cy + dx * np.sin(theta) + dy * np.cos(theta)
    coords = [np.clip(new_y, 0, h - 1), np.clip(new_x, 0, w - 1)]
    return map_coordinates(img, coords, order=3, mode="reflect")


# ── Local contrast manipulation ──────────────────────────────────────────────

def _local_contrast(img: np.ndarray, rng: np.random.RandomState,
                    kernel_size: int = 31) -> np.ndarray:
    """Apply CLAHE-like local contrast enhancement with random parameters."""
    local_mean = uniform_filter(img, size=kernel_size)
    local_sq_mean = uniform_filter(img ** 2, size=kernel_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + 1e-6)

    contrast_gain = rng.uniform(0.5, 2.0)
    enhanced = local_mean + contrast_gain * (img - local_mean) / (local_std + 0.1)
    return np.clip(enhanced, 0, 1)


# ── Edge/structure injection ─────────────────────────────────────────────────

def _inject_edges(img: np.ndarray, rng: np.random.RandomState,
                  n_curves: int = 15) -> np.ndarray:
    """Add random bezier curves and fine geometric structures."""
    h, w = img.shape
    result = img.copy()

    for _ in range(n_curves):
        # Random cubic Bezier curve
        pts = rng.rand(4, 2) * np.array([h, w])
        t = np.linspace(0, 1, 500)
        t2 = t[:, None]
        curve = ((1 - t2) ** 3 * pts[0] +
                 3 * (1 - t2) ** 2 * t2 * pts[1] +
                 3 * (1 - t2) * t2 ** 2 * pts[2] +
                 t2 ** 3 * pts[3])

        width = rng.uniform(0.3, 2.0)
        intensity = rng.uniform(0.2, 0.9)
        sign = rng.choice([-1, 1])

        for yi, xi in curve.astype(int):
            if 0 <= yi < h and 0 <= xi < w:
                r = max(1, int(width))
                y_lo, y_hi = max(0, yi - r), min(h, yi + r + 1)
                x_lo, x_hi = max(0, xi - r), min(w, xi + r + 1)
                result[y_lo:y_hi, x_lo:x_hi] += sign * intensity * 0.1

    return np.clip(result, 0, 1)


# ── Micro-texture generation ─────────────────────────────────────────────────

def _micro_texture(shape: tuple[int, int],
                   rng: np.random.RandomState) -> np.ndarray:
    """Generate fine-grained natural-looking micro-texture."""
    h, w = shape

    # Start with multi-scale Gabor-like filters
    texture = np.zeros((h, w), dtype=np.float64)

    for _ in range(8):
        freq = rng.uniform(0.05, 0.4)
        angle = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.05, 0.2)

        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        texture += amplitude * np.sin(2 * np.pi * freq * x_rot + phase)

    # Add some high-frequency noise
    texture += 0.05 * rng.randn(h, w)

    return texture


# ── Main synthesis pipeline ──────────────────────────────────────────────────

def generate_synthetic_scene(
    source_images: list[np.ndarray],
    target_size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Generate one complex synthetic scene.

    Pipeline:
      1. Select 3-5 random source crops
      2. Frequency-domain blending across random bands
      3. Add multi-octave Perlin noise texture
      4. Inject fine geometric structures (Bezier curves)
      5. Apply elastic warping + optional swirl
      6. Local contrast manipulation
      7. Micro-texture overlay
      8. Non-linear intensity mapping
      9. Final sharpening + normalize
    """
    rng = np.random.RandomState(seed)
    size = target_size

    # 1. Select and crop 3-5 source images
    n_sources = rng.randint(3, min(6, len(source_images) + 1))
    source_indices = rng.choice(len(source_images), n_sources, replace=False)
    crops = [_random_crop(source_images[idx], size, rng) for idx in source_indices]

    # 2. Frequency-domain blending
    weights = rng.dirichlet(np.ones(n_sources) * 2)
    blended = _frequency_blend(crops, weights, rng)

    # Also do spatial blending with different weights
    spatial_weights = rng.dirichlet(np.ones(n_sources) * 1.5)
    spatial_blend = sum(w * c for w, c in zip(spatial_weights, crops))

    # Mix frequency-blended and spatial-blended
    mix_ratio = rng.uniform(0.3, 0.7)
    img = mix_ratio * blended + (1 - mix_ratio) * spatial_blend

    # Normalize to [0, 1]
    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

    # 3. Add multi-octave Perlin noise
    noise_strength = rng.uniform(0.08, 0.25)
    perlin = multi_octave_noise((size, size), rng, octaves=6, persistence=0.5)
    perlin = (perlin - perlin.min()) / max(perlin.max() - perlin.min(), 1e-8)
    img = img * (1 - noise_strength) + perlin * noise_strength

    # 4. Inject fine geometric structures
    if rng.rand() > 0.3:
        n_curves = rng.randint(5, 20)
        img = _inject_edges(img, rng, n_curves=n_curves)

    # 5. Elastic warping
    warp_alpha = rng.uniform(10, 40)
    warp_sigma = rng.uniform(3, 8)
    img = _elastic_warp(img, rng, alpha=warp_alpha, sigma=warp_sigma)

    # Optional mild swirl
    if rng.rand() > 0.6:
        swirl_strength = rng.uniform(0.3, 1.5)
        img = _swirl(img, rng, strength=swirl_strength, radius=rng.uniform(50, 120))

    # 6. Local contrast manipulation
    if rng.rand() > 0.3:
        kernel = rng.choice([21, 31, 51])
        img = _local_contrast(img, rng, kernel_size=kernel)

    # 7. Micro-texture overlay
    micro_strength = rng.uniform(0.03, 0.12)
    micro = _micro_texture((size, size), rng)
    micro = (micro - micro.min()) / max(micro.max() - micro.min(), 1e-8)
    img = img * (1 - micro_strength) + micro * micro_strength

    # 8. Non-linear intensity mapping (random gamma + S-curve)
    gamma = rng.uniform(0.6, 1.6)
    img = np.clip(img, 0, 1)
    img = img ** gamma

    # S-curve contrast (mild — avoid pushing to binary)
    if rng.rand() > 0.4:
        midpoint = rng.uniform(0.4, 0.6)
        steepness = rng.uniform(4, 8)
        img = 1 / (1 + np.exp(-steepness * (img - midpoint)))

    # 9. Final normalization and mild sharpening
    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

    # If image is too bimodal (extreme contrast), apply histogram equalization
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    # Check if >40% of pixels are in the top/bottom 10% bins
    extreme_frac = (hist[:26].sum() + hist[230:].sum()) / hist.sum()
    if extreme_frac > 0.40:
        # Adaptive histogram equalization
        cdf = np.cumsum(hist).astype(np.float64)
        cdf = cdf / cdf[-1]
        img_flat = (img * 255).astype(int).clip(0, 255)
        img = cdf[img_flat]

    # Mild unsharp masking
    blurred = gaussian_filter(img, sigma=1.0)
    img = img + 0.3 * (img - blurred)
    img = np.clip(img, 0, 1)

    return img


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic challenge scenes")
    parser.add_argument("--n-dev", type=int, default=20,
                        help="Number of dev scenes to generate")
    parser.add_argument("--n-hidden", type=int, default=20,
                        help="Number of hidden scenes to generate")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dev-seed", type=int, default=77777)
    parser.add_argument("--hidden-seed", type=int, default=99999)
    parser.add_argument("--preview", action="store_true",
                        help="Save preview PNG images")
    parser.add_argument("--output-dir", type=str, default="/tmp/synthetic_scenes")
    args = parser.parse_args()

    # Load source images (BSDS400 + BrainImages as raw material)
    datasets_root = Path(__file__).resolve().parent.parent.parent / "datasets"
    bsds_dir = datasets_root / "SPC" / "BSDS400"
    brain_dir = datasets_root / "SPC" / "BrainImages_test"

    print("Loading source images...")
    sources = []
    if bsds_dir.exists():
        sources.extend(_load_source_images(bsds_dir, "jpg"))
        print(f"  BSDS400: {len(sources)} images")
    if brain_dir.exists():
        brain_imgs = _load_source_images(brain_dir, "png")
        sources.extend(brain_imgs)
        print(f"  BrainImages: {len(brain_imgs)} images")

    if len(sources) < 5:
        print("ERROR: Need at least 5 source images")
        sys.exit(1)

    print(f"Total source pool: {len(sources)} images")

    out_dir = Path(args.output_dir)

    for tier_name, n_scenes, base_seed in [
        ("dev", args.n_dev, args.dev_seed),
        ("hidden", args.n_hidden, args.hidden_seed),
    ]:
        print(f"\n{'='*60}")
        print(f"  Generating {n_scenes} {tier_name} scenes (seed={base_seed})")
        print(f"{'='*60}")

        tier_dir = out_dir / tier_name
        tier_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_scenes):
            scene_seed = base_seed + i * 137  # Spread seeds
            img = generate_synthetic_scene(sources, args.size, scene_seed)
            np.save(tier_dir / f"scene_{i:02d}.npy", img)

            if args.preview:
                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
                pil_img.save(tier_dir / f"scene_{i:02d}.png")

            std = img.std()
            entropy = -np.sum(
                np.histogram(img, bins=256, range=(0, 1), density=True)[0]
                * np.log(np.histogram(img, bins=256, range=(0, 1), density=True)[0] + 1e-10)
            )
            print(f"  scene_{i:02d}: std={std:.3f}, entropy={entropy:.1f}, "
                  f"range=[{img.min():.3f}, {img.max():.3f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
