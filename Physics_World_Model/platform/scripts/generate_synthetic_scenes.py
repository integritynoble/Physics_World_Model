#!/usr/bin/env python3
"""Generate complex synthetic scenes for SPC-Kronecker dev/hidden tiers.

Creates unique, non-traceable 256×256 grayscale images through heavy
multi-pass computational fusion of 970+ source images from 5 modalities:
  - BSDS400 natural images (400)
  - BrainImages MRI test (50)
  - TSA hyperspectral slices (280 = 10 scenes × 28 bands)
  - CACTI video frames (192 = 6 videos × 32 frames)
  - Real MRI reconstruction slices (~54)

Pipeline per scene (3 iterative rounds):
  1. Select 7-12 random source crops from diverse modalities
  2. Multi-scale patch quilting — stitch patches at 3 scales
  3. Frequency-domain band mixing with 8 random bands
  4. Phase scrambling — randomize Fourier phase, keep magnitudes
  5. Gradient field mixing — blend ∇x/∇y from different images, integrate
  6. Haar wavelet coefficient shuffling across sources
  7. Multi-octave Perlin noise injection
  8. Dense Bezier curve network (15-30 curves)
  9. Double elastic warp + multi-swirl distortion
  10. Random convolutional filter bank
  11. Local contrast + Perona-Malik nonlinear diffusion
  12. Micro-texture overlay (Gabor + noise)
  13. Non-linear intensity mapping (gamma, S-curve, polynomial)
  14. Histogram equalization fallback + unsharp masking

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

    angles = 2 * np.pi * rng.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    t = np.arange(shape[0]) // d
    s = np.arange(shape[1]) // d

    g00 = gradients[t[:, None], s[None, :]]
    g10 = gradients[t[:, None] + 1, s[None, :]]
    g01 = gradients[t[:, None], s[None, :] + 1]
    g11 = gradients[t[:, None] + 1, s[None, :] + 1]

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
    freq = 4

    for _ in range(octaves):
        if freq > min(shape):
            break
        try:
            noise = _perlin_2d(shape, (freq, freq), rng)
            result += amplitude * noise
        except (ValueError, IndexError):
            raw = rng.randn(*shape)
            sigma = max(1, shape[0] / freq / 2)
            result += amplitude * gaussian_filter(raw, sigma=sigma)
        total_amp += amplitude
        amplitude *= persistence
        freq *= 2

    return result / max(total_amp, 1e-8)


# ── Source image loading ─────────────────────────────────────────────────────

def _load_source_images(source_dir: Path, fmt: str = "jpg",
                        max_images: int = 500) -> list[np.ndarray]:
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


def _load_mat_slices(mat_path: Path, key: str) -> list[np.ndarray]:
    """Load 2D slices from a 3D .mat array."""
    try:
        import scipy.io as sio
        data = sio.loadmat(str(mat_path))
        arr = data[key]
        slices = []
        for k in range(arr.shape[2]):
            s = arr[:, :, k].astype(np.float64)
            if s.max() > 0:
                s = s / s.max()
            slices.append(s)
        return slices
    except Exception:
        return []


def _load_h5_mri_slices(h5_path: Path) -> list[np.ndarray]:
    """Load MRI reconstruction slices from HDF5."""
    try:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            rss = f["reconstruction_rss"][:]  # (N, H, W)
        slices = []
        for k in range(rss.shape[0]):
            s = rss[k].astype(np.float64)
            if s.max() > 0:
                s = s / s.max()
            slices.append(s)
        return slices
    except Exception:
        return []


def load_all_sources(datasets_root: Path) -> list[np.ndarray]:
    """Load the full multi-domain source pool (~970 images)."""
    sources: list[np.ndarray] = []

    # 1. BSDS400 natural images
    bsds_dir = datasets_root / "SPC" / "BSDS400"
    if bsds_dir.exists():
        bsds = _load_source_images(bsds_dir, "jpg")
        print(f"  BSDS400: {len(bsds)} images")
        sources.extend(bsds)

    # 2. BrainImages
    brain_dir = datasets_root / "SPC" / "BrainImages_test"
    if brain_dir.exists():
        brain = _load_source_images(brain_dir, "png")
        print(f"  BrainImages: {len(brain)} images")
        sources.extend(brain)

    # 3. TSA hyperspectral slices (10 scenes × 28 spectral bands = 280)
    tsa_dir = datasets_root / "TSA_simu_data" / "Truth"
    if tsa_dir.exists():
        import glob as _glob
        tsa_files = sorted(_glob.glob(str(tsa_dir / "*.mat")))
        tsa_count = 0
        for fp in tsa_files:
            slices = _load_mat_slices(Path(fp), "img")
            sources.extend(slices)
            tsa_count += len(slices)
        print(f"  TSA hyperspectral: {tsa_count} slices")

    # 4. CACTI video frames (6 videos × 32 frames = 192)
    cacti_dir = datasets_root / "CACTI" / "simulation"
    if cacti_dir.exists():
        import glob as _glob
        cacti_files = sorted(_glob.glob(str(cacti_dir / "*.mat")))
        cacti_count = 0
        for fp in cacti_files:
            slices = _load_mat_slices(Path(fp), "orig")
            sources.extend(slices)
            cacti_count += len(slices)
        print(f"  CACTI video: {cacti_count} frames")

    # 5. Real MRI reconstructions
    mri_dir = datasets_root / "real_mri" / "multicoil_val"
    if mri_dir.exists():
        import glob as _glob
        mri_files = sorted(_glob.glob(str(mri_dir / "*.h5")))
        mri_count = 0
        for fp in mri_files:
            slices = _load_h5_mri_slices(Path(fp))
            sources.extend(slices)
            mri_count += len(slices)
        print(f"  Real MRI: {mri_count} slices")

    return sources


def _random_crop(img: np.ndarray, size: int,
                 rng: np.random.RandomState) -> np.ndarray:
    """Extract a random square crop from an image, resizing if needed."""
    h, w = img.shape[:2]
    if h < size or w < size:
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
        scale = max(size / h, size / w) * 1.1
        new_h, new_w = int(h * scale), int(w * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img = np.array(pil_img, dtype=np.float64) / 255.0
        h, w = img.shape

    y0 = rng.randint(0, h - size + 1)
    x0 = rng.randint(0, w - size + 1)
    return img[y0:y0 + size, x0:x0 + size]


def _norm01(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    mn, mx = img.min(), img.max()
    return (img - mn) / max(mx - mn, 1e-8)


# ── Phase scrambling ─────────────────────────────────────────────────────────

def _phase_scramble(img: np.ndarray, rng: np.random.RandomState,
                    scramble_frac: float = 0.7) -> np.ndarray:
    """Randomize Fourier phase while preserving magnitude spectrum."""
    fft = np.fft.fft2(img)
    mag = np.abs(fft)
    phase = np.angle(fft)

    # Random phase perturbation
    random_phase = rng.uniform(-np.pi, np.pi, img.shape)
    # Blend original and random phase
    new_phase = phase * (1 - scramble_frac) + random_phase * scramble_frac
    # Enforce conjugate symmetry for real output
    result = np.real(np.fft.ifft2(mag * np.exp(1j * new_phase)))
    return result


# ── Gradient field mixing ────────────────────────────────────────────────────

def _gradient_field_mix(imgs: list[np.ndarray],
                        rng: np.random.RandomState) -> np.ndarray:
    """Mix gradient fields from different images and integrate via Poisson."""
    n = len(imgs)
    h, w = imgs[0].shape

    # Compute gradients for each image
    gx_list = [np.diff(im, axis=1, append=im[:, -1:]) for im in imgs]
    gy_list = [np.diff(im, axis=0, append=im[-1:, :]) for im in imgs]

    # Random weights per gradient component (different for x and y!)
    wx = rng.dirichlet(np.ones(n) * 1.5)
    wy = rng.dirichlet(np.ones(n) * 1.5)

    gx_mixed = sum(wx[i] * gx_list[i] for i in range(n))
    gy_mixed = sum(wy[i] * gy_list[i] for i in range(n))

    # Simple Poisson integration via iterative averaging (50 iterations)
    result = np.mean(imgs, axis=0).copy()
    for _ in range(50):
        # Reconstruct from gradients: Gauss-Seidel
        result[1:, :] = 0.5 * (result[1:, :] + result[:-1, :] + gy_mixed[:-1, :])
        result[:, 1:] = 0.5 * (result[:, 1:] + result[:, :-1] + gx_mixed[:, :-1])

    return _norm01(result)


# ── Haar wavelet mixing ─────────────────────────────────────────────────────

def _haar_forward(img: np.ndarray) -> tuple:
    """Single-level 2D Haar wavelet transform."""
    h, w = img.shape
    h2, w2 = h // 2, w // 2

    # Row transform
    lo = (img[:, 0::2] + img[:, 1::2]) / 2
    hi = (img[:, 0::2] - img[:, 1::2]) / 2

    # Column transform
    ll = (lo[0::2, :] + lo[1::2, :]) / 2
    lh = (lo[0::2, :] - lo[1::2, :]) / 2
    hl = (hi[0::2, :] + hi[1::2, :]) / 2
    hh = (hi[0::2, :] - hi[1::2, :]) / 2

    return ll, lh, hl, hh


def _haar_inverse(ll, lh, hl, hh) -> np.ndarray:
    """Single-level 2D inverse Haar wavelet transform."""
    h2, w2 = ll.shape
    h, w = h2 * 2, w2 * 2

    lo = np.zeros((h, w2), dtype=np.float64)
    hi = np.zeros((h, w2), dtype=np.float64)

    lo[0::2, :] = ll + lh
    lo[1::2, :] = ll - lh
    hi[0::2, :] = hl + hh
    hi[1::2, :] = hl - hh

    result = np.zeros((h, w), dtype=np.float64)
    result[:, 0::2] = lo + hi
    result[:, 1::2] = lo - hi

    return result


def _wavelet_mix(imgs: list[np.ndarray],
                 rng: np.random.RandomState,
                 levels: int = 3) -> np.ndarray:
    """Mix wavelet coefficients from different images across multiple levels."""
    n = len(imgs)

    # Multi-level decomposition
    decomps = []
    for img in imgs:
        coeffs = []
        current = img.copy()
        for _ in range(levels):
            h, w = current.shape
            # Ensure even dimensions
            h2, w2 = h - h % 2, w - w % 2
            current = current[:h2, :w2]
            ll, lh, hl, hh = _haar_forward(current)
            coeffs.append((lh, hl, hh))
            current = ll
        coeffs.append(current)  # Final approximation
        decomps.append(coeffs)

    # Mix coefficients: each subband from a different random source
    mixed_coeffs = []
    for level in range(levels):
        mixed_detail = []
        for band in range(3):  # LH, HL, HH
            # Random weighted combination for each subband
            weights = rng.dirichlet(np.ones(n) * 0.8)
            mixed = sum(weights[i] * decomps[i][level][band] for i in range(n))
            mixed_detail.append(mixed)
        mixed_coeffs.append(tuple(mixed_detail))

    # Final approximation: mix
    approx_weights = rng.dirichlet(np.ones(n) * 2)
    mixed_approx = sum(approx_weights[i] * decomps[i][levels] for i in range(n))

    # Reconstruct
    current = mixed_approx
    for level in range(levels - 1, -1, -1):
        lh, hl, hh = mixed_coeffs[level]
        # Ensure matching sizes
        h2 = min(current.shape[0], lh.shape[0])
        w2 = min(current.shape[1], lh.shape[1])
        current = _haar_inverse(
            current[:h2, :w2], lh[:h2, :w2], hl[:h2, :w2], hh[:h2, :w2]
        )

    return _norm01(current)


# ── Multi-scale patch quilting ───────────────────────────────────────────────

def _patch_quilt(imgs: list[np.ndarray], size: int,
                 rng: np.random.RandomState) -> np.ndarray:
    """Stitch patches from different images at multiple scales."""
    result = np.zeros((size, size), dtype=np.float64)
    n = len(imgs)

    for patch_size in [64, 32, 16]:
        step = patch_size
        blend_margin = patch_size // 4

        for y0 in range(0, size, step):
            for x0 in range(0, size, step):
                y1 = min(y0 + patch_size, size)
                x1 = min(x0 + patch_size, size)
                ph, pw = y1 - y0, x1 - x0

                if ph < 4 or pw < 4:
                    continue

                # Pick random source and crop
                src_idx = rng.randint(0, n)
                crop = _random_crop(imgs[src_idx], max(ph, pw), rng)[:ph, :pw]

                # Create blending mask (feathered edges)
                mask = np.ones((ph, pw), dtype=np.float64)
                bm = min(blend_margin, ph // 2, pw // 2)
                if bm > 0:
                    for k in range(bm):
                        alpha = k / bm
                        mask[k, :] *= alpha
                        mask[-(k + 1), :] *= alpha
                        mask[:, k] *= alpha
                        mask[:, -(k + 1)] *= alpha

                # Alpha-blend with existing content
                weight = rng.uniform(0.2, 0.6)
                result[y0:y1, x0:x1] = (
                    result[y0:y1, x0:x1] * (1 - weight * mask) +
                    crop * weight * mask
                )

    return _norm01(result)


# ── Random convolutional filter bank ────────────────────────────────────────

def _random_conv_filters(img: np.ndarray, rng: np.random.RandomState,
                         n_filters: int = 6) -> np.ndarray:
    """Apply random convolutional filters and combine outputs."""
    h, w = img.shape
    result = np.zeros_like(img)

    for _ in range(n_filters):
        # Random kernel (3×3 to 7×7)
        ks = rng.choice([3, 5, 7])
        kernel = rng.randn(ks, ks)
        # Normalize
        kernel = kernel / (np.abs(kernel).sum() + 1e-8)

        filtered = fftconvolve(img, kernel, mode="same")
        weight = rng.uniform(0.1, 0.4)
        result += weight * filtered

    # Mix with original
    mix = rng.uniform(0.3, 0.7)
    return _norm01(mix * img + (1 - mix) * result)


# ── Perona-Malik nonlinear diffusion ────────────────────────────────────────

def _perona_malik_diffusion(img: np.ndarray, rng: np.random.RandomState,
                            n_iter: int = 20, kappa: float = 0.05,
                            dt: float = 0.15) -> np.ndarray:
    """Edge-preserving nonlinear diffusion."""
    u = img.copy()
    kappa = rng.uniform(0.03, 0.08)

    for _ in range(n_iter):
        # Compute gradients in 4 directions
        dn = np.roll(u, -1, axis=0) - u
        ds = np.roll(u, 1, axis=0) - u
        de = np.roll(u, -1, axis=1) - u
        dw = np.roll(u, 1, axis=1) - u

        # Perona-Malik conductance (Leclerc)
        cn = np.exp(-(dn / kappa) ** 2)
        cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2)
        cw = np.exp(-(dw / kappa) ** 2)

        u = u + dt * (cn * dn + cs * ds + ce * de + cw * dw)

    return np.clip(u, 0, 1)


# ── Frequency-domain mixing (enhanced) ──────────────────────────────────────

def _frequency_blend(imgs: list[np.ndarray], weights: np.ndarray,
                     rng: np.random.RandomState,
                     n_bands: int = 8) -> np.ndarray:
    """Blend images in frequency domain with many random band mixings."""
    h, w = imgs[0].shape
    result_fft = np.zeros((h, w), dtype=np.complex128)

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq_dist = np.sqrt(fy ** 2 + fx ** 2)

    boundaries = sorted(rng.uniform(0.005, 0.5, n_bands - 1))
    boundaries = [0.0] + list(boundaries) + [1.0]

    for band_idx in range(n_bands):
        lo, hi = boundaries[band_idx], boundaries[band_idx + 1]
        mask = ((freq_dist >= lo) & (freq_dist < hi)).astype(np.float64)
        mask = gaussian_filter(mask, sigma=1)

        band_weights = rng.dirichlet(weights * 3 + 0.1)
        band_fft = np.zeros((h, w), dtype=np.complex128)
        for i, img in enumerate(imgs):
            band_fft += band_weights[i] * np.fft.fft2(img)
        result_fft += mask * band_fft

    return np.real(np.fft.ifft2(result_fft))


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
    """CLAHE-like local contrast enhancement."""
    local_mean = uniform_filter(img, size=kernel_size)
    local_sq_mean = uniform_filter(img ** 2, size=kernel_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + 1e-6)

    contrast_gain = rng.uniform(0.5, 2.0)
    enhanced = local_mean + contrast_gain * (img - local_mean) / (local_std + 0.1)
    return np.clip(enhanced, 0, 1)


# ── Edge/structure injection ─────────────────────────────────────────────────

def _inject_edges(img: np.ndarray, rng: np.random.RandomState,
                  n_curves: int = 20) -> np.ndarray:
    """Add random Bezier curves and fine geometric structures."""
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

        width = rng.uniform(0.3, 1.5)
        intensity = rng.uniform(0.1, 0.4)
        sign = rng.choice([-1, 1])

        for yi, xi in curve.astype(int):
            if 0 <= yi < h and 0 <= xi < w:
                r = max(1, int(width))
                y_lo, y_hi = max(0, yi - r), min(h, yi + r + 1)
                x_lo, x_hi = max(0, xi - r), min(w, xi + r + 1)
                result[y_lo:y_hi, x_lo:x_hi] += sign * intensity * 0.05

    return np.clip(result, 0, 1)


# ── Micro-texture generation ─────────────────────────────────────────────────

def _micro_texture(shape: tuple[int, int],
                   rng: np.random.RandomState) -> np.ndarray:
    """Fine-grained natural-looking micro-texture."""
    h, w = shape
    texture = np.zeros((h, w), dtype=np.float64)

    for _ in range(12):  # More Gabor components
        freq = rng.uniform(0.03, 0.5)
        angle = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.03, 0.15)

        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        texture += amplitude * np.sin(2 * np.pi * freq * x_rot + phase)

    texture += 0.04 * rng.randn(h, w)
    return texture


# ── Cross-image nonlinear blending ───────────────────────────────────────────

def _nonlinear_superpose(imgs: list[np.ndarray],
                         rng: np.random.RandomState) -> np.ndarray:
    """Nonlinear pixel-wise combination of multiple images."""
    n = len(imgs)
    h, w = imgs[0].shape

    # Stack all images
    stack = np.stack(imgs, axis=0)  # (n, h, w)

    # Method: random per-pixel polynomial combination
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        # Random exponent per image
        exp = rng.uniform(0.5, 2.0)
        weight = rng.uniform(0.1, 1.0)
        result += weight * (stack[i] ** exp)

    # Add cross-terms (product of pairs)
    n_cross = min(n * (n - 1) // 2, 6)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    rng.shuffle(pairs)
    for i, j in pairs[:n_cross]:
        cross_weight = rng.uniform(0.05, 0.3)
        result += cross_weight * stack[i] * stack[j]

    return _norm01(result)


# ── Main synthesis pipeline ──────────────────────────────────────────────────

def generate_synthetic_scene(
    source_images: list[np.ndarray],
    target_size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Generate one complex synthetic scene through heavy multi-pass fusion.

    Uses 7-12 source images from diverse modalities, applies 3 iterative
    rounds of blending and transformation to make the result completely
    non-traceable to any single source.
    """
    rng = np.random.RandomState(seed)
    size = target_size
    n_pool = len(source_images)

    # ── Round 1: Initial multi-source fusion ─────────────────────────────

    # Select 7-12 diverse source crops
    n_sources = rng.randint(7, min(13, n_pool + 1))
    source_indices = rng.choice(n_pool, n_sources, replace=False)
    crops = [_random_crop(source_images[idx], size, rng) for idx in source_indices]

    # A) Frequency-domain blending with 8 bands
    weights = rng.dirichlet(np.ones(n_sources) * 2)
    freq_blended = _frequency_blend(crops, weights, rng, n_bands=8)
    freq_blended = _norm01(freq_blended)

    # B) Gradient field mixing
    grad_mixed = _gradient_field_mix(crops[:min(5, n_sources)], rng)

    # C) Wavelet coefficient shuffling
    wav_mixed = _wavelet_mix(crops[:min(6, n_sources)], rng, levels=3)

    # D) Multi-scale patch quilting
    quilted = _patch_quilt(crops, size, rng)

    # E) Nonlinear superposition
    nonlin = _nonlinear_superpose(crops[:min(5, n_sources)], rng)

    # Combine the 5 fusion results
    fusion_weights = rng.dirichlet(np.ones(5) * 1.5)
    img = (fusion_weights[0] * freq_blended +
           fusion_weights[1] * grad_mixed +
           fusion_weights[2] * wav_mixed +
           fusion_weights[3] * quilted +
           fusion_weights[4] * nonlin)
    img = _norm01(img)

    # ── Round 2: Geometric and spatial transformations ───────────────────

    # Phase scramble (mild — keep substantial structure)
    scramble_frac = rng.uniform(0.2, 0.45)
    img = _norm01(_phase_scramble(img, rng, scramble_frac))

    # Perlin noise injection
    noise_strength = rng.uniform(0.08, 0.20)
    perlin = multi_octave_noise((size, size), rng, octaves=7, persistence=0.5)
    perlin = _norm01(perlin)
    img = img * (1 - noise_strength) + perlin * noise_strength

    # Bezier curve network (moderate density)
    n_curves = rng.randint(8, 18)
    img = _inject_edges(img, rng, n_curves=n_curves)

    # Double elastic warp (two passes with different parameters)
    for _ in range(2):
        warp_alpha = rng.uniform(12, 35)
        warp_sigma = rng.uniform(3, 8)
        img = _elastic_warp(img, rng, alpha=warp_alpha, sigma=warp_sigma)

    # Multi-swirl (1-3 swirls at different locations)
    n_swirls = rng.randint(1, 4)
    for _ in range(n_swirls):
        swirl_strength = rng.uniform(0.2, 0.9)
        img = _swirl(img, rng, strength=swirl_strength,
                     radius=rng.uniform(40, 100))

    # ── Round 3: Enhancement and second-pass fusion ──────────────────────

    # Pick 3-5 MORE source images for second-pass fusion
    extra_indices = rng.choice(n_pool, rng.randint(3, 6), replace=False)
    extra_crops = [_random_crop(source_images[idx], size, rng) for idx in extra_indices]

    # Blend extras with current result in frequency domain
    combined = [img] + extra_crops
    blend_w = rng.dirichlet(np.array([3.0] + [0.8] * len(extra_crops)))
    img2 = _frequency_blend(combined, blend_w, rng, n_bands=6)
    img2 = _norm01(img2)

    # Mix with original (keep most of original)
    second_mix = rng.uniform(0.15, 0.35)
    img = img * (1 - second_mix) + img2 * second_mix
    img = _norm01(img)

    # Random convolutional filter bank
    img = _random_conv_filters(img, rng, n_filters=6)

    # Local contrast manipulation
    kernel = rng.choice([21, 31, 41, 51])
    img = _local_contrast(img, rng, kernel_size=kernel)

    # Perona-Malik nonlinear diffusion (edge-preserving smoothing)
    pm_iters = rng.randint(10, 25)
    img = _perona_malik_diffusion(img, rng, n_iter=pm_iters)

    # Micro-texture overlay
    micro_strength = rng.uniform(0.04, 0.12)
    micro = _micro_texture((size, size), rng)
    micro = _norm01(micro)
    img = img * (1 - micro_strength) + micro * micro_strength

    # ── Final intensity grading ──────────────────────────────────────────

    # Random gamma
    gamma = rng.uniform(0.6, 1.5)
    img = np.clip(img, 0, 1) ** gamma

    # Mild S-curve
    if rng.rand() > 0.4:
        midpoint = rng.uniform(0.42, 0.58)
        steepness = rng.uniform(3.5, 6)
        img = 1 / (1 + np.exp(-steepness * (img - midpoint)))

    # Random polynomial warp: a*x^3 + b*x^2 + c*x (monotonic)
    if rng.rand() > 0.5:
        a = rng.uniform(-0.3, 0.3)
        b = rng.uniform(-0.3, 0.3)
        c = 1.0 - a - b  # Ensure f(1)=1
        img_p = a * img ** 3 + b * img ** 2 + c * img
        # Only use if monotonic (derivative > 0 everywhere in [0,1])
        deriv_min = min(3 * a * 0 + 2 * b * 0 + c,
                        3 * a * 1 + 2 * b * 1 + c,
                        3 * a * 0.5 + 2 * b * 0.5 + c)
        if deriv_min > 0:
            img = img_p

    img = _norm01(img)

    # Histogram equalization fallback for extreme contrast
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    extreme_frac = (hist[:26].sum() + hist[230:].sum()) / hist.sum()
    if extreme_frac > 0.30:
        cdf = np.cumsum(hist).astype(np.float64)
        cdf = cdf / cdf[-1]
        img_flat = (img * 255).astype(int).clip(0, 255)
        img = cdf[img_flat]

    # Final unsharp masking
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

    # Load ALL source images from multiple modalities
    datasets_root = Path(__file__).resolve().parent.parent.parent / "datasets"
    print("Loading multi-domain source pool...")
    sources = load_all_sources(datasets_root)

    if len(sources) < 10:
        print(f"ERROR: Need at least 10 source images, got {len(sources)}")
        sys.exit(1)

    print(f"Total source pool: {len(sources)} images from 5 modalities")

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
            scene_seed = base_seed + i * 137
            img = generate_synthetic_scene(sources, args.size, scene_seed)
            np.save(tier_dir / f"scene_{i:02d}.npy", img)

            if args.preview:
                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
                pil_img.save(tier_dir / f"scene_{i:02d}.png")

            std = img.std()
            hist_data = np.histogram(img, bins=256, range=(0, 1), density=True)[0]
            entropy = -np.sum(hist_data * np.log(hist_data + 1e-10))
            print(f"  scene_{i:02d}: std={std:.3f}, entropy={entropy:.1f}, "
                  f"range=[{img.min():.3f}, {img.max():.3f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
