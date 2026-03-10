#!/usr/bin/env python3
"""Generate Retinal Fundus Photography benchmark dataset.

Forward model (fundus degradation):
    y = H * (x * I(r)) + n

where:
    x              -- ground-truth retinal image (256x256 grayscale, green channel)
    H              -- point spread function (optical aberrations: defocus + astigmatism)
    I(r)           -- illumination non-uniformity field (radial cosine rolloff)
    n              -- additive Gaussian sensor noise
    *              -- element-wise multiplication (illumination)
    H * ...        -- convolution with PSF

Additional degradation:
    media_opacity  -- cataract-like haze (additive veiling glare, low-frequency)

Ground truth phantoms (256x256 grayscale):
    Synthetic retinal fundus images (green channel) with:
    - Bright optic disc (circular, off-center)
    - Vessel tree (branching arteries and veins from optic disc)
    - Dark fovea (centre of macula)
    - Background retinal texture (Perlin-like noise)
    - Pathological features for dev/hidden: microaneurysms, hemorrhages,
      drusen, hard exudates

Mismatch parameters:
    defocus_diopters           : optical defocus (0-2 D public, 0-5 D hidden)
    illumination_nonuniformity : uneven lighting fraction (0-0.15 public, 0-0.40 hidden)
    media_opacity              : cataract haze level (0-0.1 public, 0-0.3 hidden)
    noise_sigma                : sensor noise std (0.01-0.03 public, 0.01-0.08 hidden)

Tiers:
    public : 12 samples (4 normal + 4 mild pathology + 4 varied anatomy)
    dev    : 20 samples (augmented, medium mismatch)
    hidden : 20 samples (adversarial, wide mismatch + pathologies)

CPU reconstruction: Wiener deconvolution + illumination correction

Usage:
    cd datasets/benchmark/fundus
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate as nd_rotate, zoom as nd_zoom
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# ── Physics constants ────────────────────────────────────────────────────────

# Eye optics: 1 diopter of defocus ~ 0.35 mm blur at retina
# For 256x256 image covering ~30 deg FOV (6 mm on retina),
# pixel_size ~ 6/256 mm = 23.4 um
# PSF sigma for D diopters ~ 0.35 * D / pixel_size_mm ~ 15 * D pixels
PIXEL_SIZE_MM = 6.0 / IMAGE_SIZE  # ~23.4 um
PSF_SCALE = 0.35 / PIXEL_SIZE_MM  # pixels per diopter (~15 px/D)

# Typical fundus reflectance values (green channel, normalised 0-1)
REFLECTANCE_BACKGROUND = 0.35      # retinal pigment epithelium + choroid
REFLECTANCE_DISC = 0.85            # optic nerve head (bright)
REFLECTANCE_FOVEA = 0.20           # foveal pit (dark, macular pigment)
REFLECTANCE_ARTERY = 0.55          # arteries (brighter red -> moderate green)
REFLECTANCE_VEIN = 0.25            # veins (darker, absorb green)
REFLECTANCE_MACULA = 0.28          # macular region

# ── Mismatch spec ranges per tier ────────────────────────────────────────────

SPEC = {
    "public": {
        "psf_sigma":                  {"min": 0.5,  "max": 3.0,  "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0,  "max": 0.15, "unit": ""},
        "media_opacity":              {"min": 0.0,  "max": 0.08, "unit": ""},
        "noise_sigma":                {"min": 0.005,"max": 0.020,"unit": ""},
    },
    "dev": {
        "psf_sigma":                  {"min": 1.0,  "max": 5.0,  "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0,  "max": 0.25, "unit": ""},
        "media_opacity":              {"min": 0.0,  "max": 0.15, "unit": ""},
        "noise_sigma":                {"min": 0.008,"max": 0.035,"unit": ""},
    },
    "hidden": {
        "psf_sigma":                  {"min": 1.5,  "max": 8.0,  "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0,  "max": 0.40, "unit": ""},
        "media_opacity":              {"min": 0.0,  "max": 0.25, "unit": ""},
        "noise_sigma":                {"min": 0.010,"max": 0.050,"unit": ""},
    },
}


# ── Perlin-like noise for retinal texture ────────────────────────────────────

def _perlin_noise(H: int, W: int, scale: float, rng: np.random.Generator,
                  octaves: int = 4) -> np.ndarray:
    """Generate multi-octave Perlin-like noise via smoothed random fields."""
    noise = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    for _ in range(octaves):
        raw = rng.standard_normal((H, W))
        smoothed = gaussian_filter(raw, sigma=scale)
        noise += amplitude * smoothed
        amplitude *= 0.5
        scale *= 0.5
    # Normalise to [0, 1]
    lo, hi = noise.min(), noise.max()
    if hi - lo > 1e-10:
        noise = (noise - lo) / (hi - lo)
    return noise


# ── Vessel tree generation ───────────────────────────────────────────────────

def _draw_vessel_segment(canvas: np.ndarray, y0: float, x0: float,
                         y1: float, x1: float, width: float,
                         reflectance: float) -> None:
    """Draw an anti-aliased line segment on the canvas."""
    H, W = canvas.shape
    length = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    if length < 0.5:
        return
    n_pts = max(int(length * 3), 10)
    ts = np.linspace(0, 1, n_pts)
    ys = y0 + ts * (y1 - y0)
    xs = x0 + ts * (x1 - x0)
    for y, x in zip(ys, xs):
        iy, ix = int(round(y)), int(round(x))
        r = max(1, int(round(width)))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                py, px = iy + dy, ix + dx
                if 0 <= py < H and 0 <= px < W:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= width:
                        alpha = max(0.0, 1.0 - dist / (width + 0.5))
                        canvas[py, px] = (1 - alpha) * canvas[py, px] + alpha * reflectance


def _grow_vessel_tree(canvas: np.ndarray, start_y: float, start_x: float,
                      angle_deg: float, length: float, width: float,
                      reflectance: float, depth: int, max_depth: int,
                      rng: np.random.Generator) -> None:
    """Recursively grow a branching vessel tree."""
    if depth > max_depth or length < 3 or width < 0.3:
        return
    angle_rad = np.radians(angle_deg)
    end_y = start_y + length * np.sin(angle_rad)
    end_x = start_x + length * np.cos(angle_rad)

    # Add slight curvature via midpoint displacement
    mid_y = (start_y + end_y) / 2 + rng.uniform(-length * 0.08, length * 0.08)
    mid_x = (start_x + end_x) / 2 + rng.uniform(-length * 0.08, length * 0.08)

    _draw_vessel_segment(canvas, start_y, start_x, mid_y, mid_x, width, reflectance)
    _draw_vessel_segment(canvas, mid_y, mid_x, end_y, end_x, width * 0.95, reflectance)

    # Branch: 1-3 child branches
    n_branches = rng.integers(1, 4)
    for _ in range(n_branches):
        branch_angle = angle_deg + rng.uniform(-40, 40)
        branch_length = length * rng.uniform(0.5, 0.75)
        branch_width = width * rng.uniform(0.55, 0.75)
        _grow_vessel_tree(canvas, end_y, end_x, branch_angle,
                          branch_length, branch_width,
                          reflectance, depth + 1, max_depth, rng)


# ── Phantom generators ──────────────────────────────────────────────────────

def _ellipse_mask(H: int, W: int, cy: float, cx: float,
                  ry: float, rx: float) -> np.ndarray:
    """Generate soft elliptical mask. (cy, cx) in pixel coords, ry/rx in pixels."""
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = ((yy - cy) / max(ry, 1e-6))**2 + ((xx - cx) / max(rx, 1e-6))**2
    return np.clip(1.0 - dist, 0.0, 1.0)


def _disc_mask(H: int, W: int, cy: float, cx: float,
               radius: float) -> np.ndarray:
    """Hard circular disc mask with anti-aliased edge."""
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    return np.clip(1.0 - (dist - radius), 0.0, 1.0)


def make_normal_fundus(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Generate a normal retinal fundus phantom (green channel).

    Returns:
        image: (H, W) float64 in [0, 1]
        name:  scene name
    """
    rng = np.random.default_rng(seed)

    # Start with background retinal reflectance + texture
    texture = _perlin_noise(H, W, scale=20.0 + variant * 2, rng=rng, octaves=4)
    image = np.full((H, W), REFLECTANCE_BACKGROUND, dtype=np.float64)
    image += (texture - 0.5) * 0.08  # subtle texture variation

    # Circular retinal field (fundus camera FOV is circular)
    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)
    # Dark surround outside FOV
    image *= field_mask

    # Optic disc: bright circular region, positioned right of centre
    disc_cy = H * (0.48 + rng.uniform(-0.03, 0.03) + variant * 0.005)
    disc_cx = W * (0.70 + rng.uniform(-0.04, 0.04) - variant * 0.01)
    disc_r = 18 + rng.uniform(-3, 3) + variant * 0.5
    disc = _disc_mask(H, W, disc_cy, disc_cx, disc_r)
    # Optic cup (brighter centre of disc)
    cup_r = disc_r * rng.uniform(0.3, 0.5)
    cup = _disc_mask(H, W, disc_cy, disc_cx, cup_r)
    image = image * (1 - disc) + disc * REFLECTANCE_DISC
    image = image * (1 - cup * 0.3) + cup * 0.3 * 0.95  # cup is slightly brighter

    # Macula: darker region centred slightly left of image centre
    mac_cy = H * (0.50 + rng.uniform(-0.02, 0.02))
    mac_cx = W * (0.42 + rng.uniform(-0.03, 0.03))
    mac_mask = _ellipse_mask(H, W, mac_cy, mac_cx, 35 + variant * 2, 40 + variant * 2)
    image = image * (1 - mac_mask * 0.25) + mac_mask * 0.25 * REFLECTANCE_MACULA

    # Fovea: dark pit at macula centre
    fov_mask = _disc_mask(H, W, mac_cy, mac_cx, 8 + rng.uniform(-1, 1))
    image = image * (1 - fov_mask * 0.6) + fov_mask * 0.6 * REFLECTANCE_FOVEA

    # Vessel tree: arteries and veins emanating from optic disc
    # Major arteries (4 main branches)
    for i, base_angle in enumerate([30, 150, 210, 330]):
        angle = base_angle + rng.uniform(-15, 15) + variant * 3
        length = rng.uniform(80, 130)
        width = rng.uniform(2.5, 4.0)
        refl = REFLECTANCE_ARTERY + rng.uniform(-0.05, 0.05)
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, length, width,
                          refl, 0, 4 + variant % 2, rng)

    # Major veins (4 main branches, slightly offset from arteries)
    for i, base_angle in enumerate([60, 120, 240, 300]):
        angle = base_angle + rng.uniform(-15, 15) + variant * 2
        length = rng.uniform(80, 130)
        width = rng.uniform(3.0, 4.5)
        refl = REFLECTANCE_VEIN + rng.uniform(-0.03, 0.03)
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, length, width,
                          refl, 0, 4 + variant % 2, rng)

    # Apply field mask and smooth
    image *= field_mask
    image = gaussian_filter(image, sigma=0.5)
    image = np.clip(image, 0.0, 1.0)

    return image, f"normal_{variant:02d}"


def make_pathological_fundus(
    H: int, W: int, seed: int, variant: int = 0, severity: str = "mild"
) -> tuple[np.ndarray, str]:
    """Generate a fundus phantom with pathological features.

    Adds microaneurysms, hemorrhages, drusen, and/or hard exudates.

    Args:
        severity: "mild" (public), "moderate" (dev), "severe" (hidden)

    Returns:
        image: (H, W) float64 in [0, 1]
        name:  scene name
    """
    rng = np.random.default_rng(seed)

    # Start with normal fundus
    image, _ = make_normal_fundus(H, W, seed + 10000, variant)

    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)

    severity_scale = {"mild": 1.0, "moderate": 2.0, "severe": 3.0}[severity]

    # Microaneurysms: tiny bright-then-dark dots near vessels
    n_ma = int(rng.integers(3, 8) * severity_scale)
    for _ in range(n_ma):
        my = rng.integers(40, H - 40)
        mx = rng.integers(40, W - 40)
        if field_mask[my, mx] < 0.5:
            continue
        mr = rng.uniform(1.0, 2.5)
        dot = _disc_mask(H, W, float(my), float(mx), mr)
        # Dark red dot (low green reflectance)
        image = image * (1 - dot * 0.8) + dot * 0.8 * 0.12

    # Hemorrhages: larger dark blotches
    n_hem = int(rng.integers(1, 4) * severity_scale)
    for _ in range(n_hem):
        hy = rng.integers(50, H - 50)
        hx = rng.integers(50, W - 50)
        if field_mask[hy, hx] < 0.5:
            continue
        hr_y = rng.uniform(5, 15 * severity_scale)
        hr_x = rng.uniform(5, 15 * severity_scale)
        hem = _ellipse_mask(H, W, float(hy), float(hx), hr_y, hr_x)
        # Irregular shape via noise
        noise = _perlin_noise(H, W, scale=10, rng=rng, octaves=2)
        hem *= (noise > 0.4).astype(np.float64)
        hem = gaussian_filter(hem, sigma=2)
        hem = np.clip(hem, 0, 1)
        image = image * (1 - hem * 0.7) + hem * 0.7 * 0.10  # dark hemorrhage

    # Drusen: small bright yellowish deposits (high green reflectance)
    n_drusen = int(rng.integers(2, 6) * severity_scale)
    for _ in range(n_drusen):
        dy = rng.integers(H // 4, 3 * H // 4)
        dx = rng.integers(W // 4, 3 * W // 4)
        if field_mask[dy, dx] < 0.5:
            continue
        dr = rng.uniform(3, 8)
        dru = _disc_mask(H, W, float(dy), float(dx), dr)
        image = image * (1 - dru * 0.6) + dru * 0.6 * rng.uniform(0.60, 0.80)

    # Hard exudates: bright deposits with sharp edges
    n_exu = int(rng.integers(1, 4) * severity_scale)
    for _ in range(n_exu):
        ey = rng.integers(60, H - 60)
        ex = rng.integers(60, W - 60)
        if field_mask[ey, ex] < 0.5:
            continue
        er_y = rng.uniform(3, 10)
        er_x = rng.uniform(3, 10)
        exu = _ellipse_mask(H, W, float(ey), float(ex), er_y, er_x)
        exu = (exu > 0.3).astype(np.float64)  # sharp edges
        exu = gaussian_filter(exu, sigma=0.8)
        image = image * (1 - exu * 0.7) + exu * 0.7 * rng.uniform(0.70, 0.90)

    image = np.clip(image, 0.0, 1.0)
    sev_tag = severity[0]  # m, m, s
    return image, f"pathological_{sev_tag}_{variant:02d}"


def make_varied_anatomy_fundus(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Generate a fundus with varied anatomical features.

    Variations: disc size, cup-to-disc ratio, vessel density,
    macular pigment density.
    """
    rng = np.random.default_rng(seed)

    texture = _perlin_noise(H, W, scale=25 + variant * 3, rng=rng, octaves=4)
    bg_val = REFLECTANCE_BACKGROUND + rng.uniform(-0.05, 0.05)
    image = np.full((H, W), bg_val, dtype=np.float64)
    image += (texture - 0.5) * 0.10

    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)
    image *= field_mask

    # Larger or smaller optic disc
    disc_cy = H * (0.45 + rng.uniform(-0.05, 0.05))
    disc_cx = W * (0.65 + rng.uniform(-0.08, 0.08))
    disc_r = rng.uniform(14, 28)  # wide range
    disc = _disc_mask(H, W, disc_cy, disc_cx, disc_r)

    # Variable cup-to-disc ratio (0.2 to 0.7)
    cdr = rng.uniform(0.2, 0.7)
    cup_r = disc_r * cdr
    cup = _disc_mask(H, W, disc_cy, disc_cx, cup_r)
    image = image * (1 - disc) + disc * REFLECTANCE_DISC
    image = image * (1 - cup * 0.4) + cup * 0.4 * 0.92

    # Macula with variable pigmentation
    mac_cy = H * (0.50 + rng.uniform(-0.04, 0.04))
    mac_cx = W * (0.40 + rng.uniform(-0.05, 0.05))
    mac_r = rng.uniform(25, 45)
    mac_mask = _disc_mask(H, W, mac_cy, mac_cx, mac_r)
    mac_dark = rng.uniform(0.18, 0.30)
    image = image * (1 - mac_mask * 0.3) + mac_mask * 0.3 * mac_dark

    fov_mask = _disc_mask(H, W, mac_cy, mac_cx, rng.uniform(5, 12))
    image = image * (1 - fov_mask * 0.5) + fov_mask * 0.5 * REFLECTANCE_FOVEA

    # Dense or sparse vessel tree
    n_main = rng.integers(3, 7)
    for i in range(n_main):
        angle = i * (360 / n_main) + rng.uniform(-20, 20)
        is_vein = (i % 2 == 0)
        refl = REFLECTANCE_VEIN if is_vein else REFLECTANCE_ARTERY
        refl += rng.uniform(-0.04, 0.04)
        width = rng.uniform(2.0, 4.5)
        length = rng.uniform(70, 140)
        max_d = rng.integers(3, 6)
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, length, width,
                          refl, 0, max_d, rng)

    image *= field_mask
    image = gaussian_filter(image, sigma=0.5)
    image = np.clip(image, 0.0, 1.0)

    return image, f"varied_{variant:02d}"


# ── Phantom pools per tier ──────────────────────────────────────────────────

PHANTOM_GENERATORS = [make_normal_fundus, make_pathological_fundus, make_varied_anatomy_fundus]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, str]]:
    """12 public samples: 4 normal + 4 mild pathology + 4 varied anatomy."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_normal_fundus(IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_pathological_fundus(IMAGE_SIZE, IMAGE_SIZE,
                                                  seed=200 + i, variant=i, severity="mild"))
    for i in range(4):
        phantoms.append(make_varied_anatomy_fundus(IMAGE_SIZE, IMAGE_SIZE,
                                                    seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """20 dev samples with augmented diversity + moderate pathology."""
    phantoms = []
    rng_aug = np.random.default_rng(5000)
    generators = [
        lambda H, W, s, v: make_normal_fundus(H, W, s, v),
        lambda H, W, s, v: make_pathological_fundus(H, W, s, v, severity="moderate"),
        lambda H, W, s, v: make_varied_anatomy_fundus(H, W, s, v),
    ]
    for i in range(n):
        gen_fn = generators[i % 3]
        image, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, 500 + i, i)

        # Augment: rotation + flip
        angle = float(rng_aug.uniform(5, 355))
        image = nd_rotate(image, angle, reshape=False, mode='constant', cval=0.0)
        if rng_aug.random() < 0.5:
            image = np.fliplr(image)

        image = np.clip(image, 0.0, 1.0)
        phantoms.append((image, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """20 hidden samples with adversarial modifications + severe pathology."""
    phantoms = []
    rng_aug = np.random.default_rng(8000)
    generators = [
        lambda H, W, s, v: make_normal_fundus(H, W, s, v),
        lambda H, W, s, v: make_pathological_fundus(H, W, s, v, severity="severe"),
        lambda H, W, s, v: make_varied_anatomy_fundus(H, W, s, v),
    ]
    for i in range(n):
        gen_fn = generators[i % 3]
        image, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, 800 + i, i + 10)

        # Aggressive augmentation
        angle = float(rng_aug.uniform(10, 350))
        image = nd_rotate(image, angle, reshape=False, mode='constant', cval=0.0)
        if rng_aug.random() < 0.7:
            image = np.fliplr(image)
        if rng_aug.random() < 0.5:
            image = np.flipud(image)

        # Zoom variation
        zoom_f = float(rng_aug.uniform(0.80, 1.20))
        image = _zoom_crop(image, zoom_f, IMAGE_SIZE)

        image = np.clip(image, 0.0, 1.0)
        phantoms.append((image, f"hidden_{name}"))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    zoomed = nd_zoom(arr, zoom_f, order=1)
    H, W = zoomed.shape
    if H >= size and W >= size:
        y0 = (H - size) // 2
        x0 = (W - size) // 2
        return zoomed[y0:y0 + size, x0:x0 + size]
    else:
        out = np.zeros((size, size), dtype=arr.dtype)
        y0 = (size - H) // 2
        x0 = (size - W) // 2
        out[y0:y0 + H, x0:x0 + W] = zoomed
        return out


# ── PSF generation (optical aberrations) ────────────────────────────────────

def make_defocus_psf(psf_sigma: float, size: int = 31) -> np.ndarray:
    """Create a defocus PSF (Gaussian approximation of Airy disc broadening).

    Args:
        psf_sigma: PSF standard deviation in pixels
        size: kernel size (odd)

    Returns:
        psf: (size, size) normalised PSF
    """
    if size % 2 == 0:
        size += 1
    sigma = max(0.3, psf_sigma)
    sigma = min(sigma, size / 3.0)

    center = size // 2
    yy = np.arange(size, dtype=np.float64) - center
    xx = yy.copy()
    yy, xx = np.meshgrid(yy, xx, indexing='ij')
    psf = np.exp(-(yy**2 + xx**2) / (2 * sigma**2 + 1e-10))
    psf /= psf.sum()
    return psf


# ── Illumination field ──────────────────────────────────────────────────────

def make_illumination_field(H: int, W: int, nonuniformity: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Create non-uniform illumination field.

    Models the vignetting / cosine-4th-law falloff of fundus camera optics.

    Args:
        nonuniformity: 0 = perfect, 0.4 = severe peripheral darkening

    Returns:
        illum: (H, W) field in [1-nonuniformity, 1]
    """
    cy = H / 2 + rng.uniform(-5, 5)
    cx = W / 2 + rng.uniform(-5, 5)
    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    r = np.sqrt(yy**2 + xx**2)
    r_max = np.sqrt(cy**2 + cx**2) + 1e-6

    # Cosine-4th falloff: I(r) = cos^4(theta) where theta ~ r / focal_length
    cos_theta = np.clip(1.0 - 0.5 * (r / r_max)**2, 0.1, 1.0)
    illum = cos_theta**4

    # Scale: uniform = 1.0, nonuniform = 1 at centre, (1-nonuniformity) at edges
    illum = 1.0 - nonuniformity * (1.0 - illum)
    return illum


# ── Forward model ───────────────────────────────────────────────────────────

def _make_H_ideal(psf: np.ndarray, H: int, W: int) -> np.ndarray:
    """Construct the full-size ideal forward operator (PSF padded to image size).

    The PSF is placed at centre and rolled so that its peak sits at [0,0],
    matching the convention for FFT-based convolution.

    Returns:
        H_ideal: (H, W) float32 -- padded PSF (sums to 1.0)
    """
    psf_pad = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    y0 = (H - kh) // 2
    x0 = (W - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)
    return psf_pad.astype(np.float32)


def fundus_forward_model(
    x_true: np.ndarray,
    psf_sigma: float,
    illumination_nonuniformity: float,
    media_opacity: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> dict:
    """Apply full fundus forward model.

    y = N(B(PSF * x) * I + scatter)

    where:
        PSF     -- optical point spread function (Gaussian, sigma in pixels)
        B       -- vignetting / illumination falloff (cosine-4th)
        I       -- non-uniform illumination pattern
        scatter -- media opacity (cataract / vitreous haze)
        N       -- Poisson-Gaussian noise

    Args:
        x_true: (H, W) ground-truth retinal image [0, 1]
        psf_sigma: PSF blur sigma in pixels
        illumination_nonuniformity: vignetting strength
        media_opacity: cataract haze level
        noise_sigma: Gaussian noise std
        rng: random generator

    Returns:
        dict with y, H_ideal, image_ideal, image_measured, psf,
        illumination_field, scatter_field
    """
    H, W = x_true.shape

    # 1. PSF (defocus blur)
    psf_size = min(61, max(11, int(6 * psf_sigma + 1) | 1))
    if psf_size % 2 == 0:
        psf_size += 1
    psf = make_defocus_psf(psf_sigma, size=psf_size)

    # 2. Illumination field
    illum = make_illumination_field(H, W, illumination_nonuniformity, rng)

    # 3. Apply illumination to ground truth
    x_illum = x_true * illum

    # 4. Convolve with PSF (optical blur)
    if psf_sigma > 0.3:
        image_blurred = fftconvolve(x_illum, psf, mode='same')
    else:
        image_blurred = x_illum.copy()

    # This is the "ideal" degraded image (before noise and haze)
    image_ideal = np.clip(image_blurred, 0.0, 1.0).astype(np.float32)

    # 5. Media opacity (cataract haze): additive low-frequency veiling glare
    scatter_field = np.zeros((H, W), dtype=np.float32)
    if media_opacity > 0.001:
        # Haze is a heavily blurred version of the image + uniform component
        haze = gaussian_filter(image_blurred, sigma=40) * 0.6 + 0.4
        scatter_field = (media_opacity * haze).astype(np.float32)
        image_hazy = (1 - media_opacity) * image_blurred + media_opacity * haze
    else:
        image_hazy = image_blurred

    # 6. Poisson-Gaussian noise model
    #    Poisson component (photon shot noise) + Gaussian readout
    peak_photons = max(1.0 / (noise_sigma**2 + 1e-10), 100.0)
    image_poisson = rng.poisson(
        np.clip(image_hazy * peak_photons, 0.01, None)
    ).astype(np.float64) / peak_photons
    readout_sigma = noise_sigma * 0.3  # readout is a fraction of total noise
    noise_readout = rng.normal(0, readout_sigma, (H, W))
    image_measured = image_poisson + noise_readout
    image_measured = np.clip(image_measured, 0.0, 1.0).astype(np.float32)

    # Build H_ideal: padded PSF at image resolution
    H_ideal = _make_H_ideal(psf, H, W)

    return {
        "image_ideal": image_ideal,
        "image_measured": image_measured,
        "y": image_measured,  # alias: y = degraded measurement
        "H_ideal": H_ideal,
        "psf": psf.astype(np.float32),
        "illumination_field": illum.astype(np.float32),
        "scatter_field": scatter_field,
    }


# ── CPU reconstruction: Wiener deconvolution + illumination correction ──────

def wiener_deconvolution(image: np.ndarray, psf: np.ndarray,
                         noise_power: float = 0.01) -> np.ndarray:
    """Wiener deconvolution in the frequency domain.

    x_hat = F^-1[ H* / (|H|^2 + K) * Y ]

    where K = noise_power (regularisation parameter).
    """
    H_img, W_img = image.shape
    # Pad PSF to image size
    psf_pad = np.zeros((H_img, W_img), dtype=np.float64)
    kh, kw = psf.shape
    y0 = (H_img - kh) // 2
    x0 = (W_img - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    # Centre the PSF (shift so peak is at [0,0])
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)

    # FFT
    Y = np.fft.fft2(image.astype(np.float64))
    H = np.fft.fft2(psf_pad)
    H_conj = np.conj(H)
    H_abs2 = np.abs(H)**2

    # Wiener filter
    X_hat = H_conj / (H_abs2 + noise_power) * Y
    x_hat = np.real(np.fft.ifft2(X_hat))
    return x_hat


def reconstruct_cpu(image_measured: np.ndarray, psf: np.ndarray,
                    noise_sigma: float) -> np.ndarray:
    """CPU baseline reconstruction: Wiener deconvolution + illumination correction.

    Tuned to produce ~22-28 dB PSNR on typical fundus degradations.

    Steps:
    1. Estimate illumination field from heavily smoothed image
    2. Correct illumination
    3. Wiener deconvolution with adaptive noise regularisation
    4. Mild post-smoothing to suppress ringing
    5. Clip and return
    """
    H, W = image_measured.shape
    img = image_measured.astype(np.float64)

    # 1. Estimate illumination: heavy low-pass filter
    illum_est = gaussian_filter(img, sigma=50)
    illum_est = np.clip(illum_est, 0.05, None)
    # Normalise so mean illum ~ 1
    illum_est /= (illum_est.mean() + 1e-8)

    # 2. Correct illumination
    img_corrected = img / illum_est
    img_corrected = np.clip(img_corrected, 0.0, 1.0)

    # 3. Wiener deconvolution with adaptive regularisation
    # Estimate effective PSF sigma (width of blur kernel)
    kh = psf.shape[0]
    yy = np.arange(kh, dtype=np.float64) - kh // 2
    psf_f64 = psf.astype(np.float64)
    psf_var = np.sum(psf_f64 * yy[:, None]**2) + np.sum(psf_f64 * yy[None, :]**2)
    psf_sigma_eff = max(0.5, np.sqrt(max(psf_var, 0.0)))

    # Adaptive regularisation: balance noise suppression vs deconvolution
    # For noise-dominated regime (low blur, high noise): use SNR-based reg
    # For blur-dominated regime (high blur, low noise): use lighter reg
    noise_power = noise_sigma**2
    # Base: just noise power; scale mildly with PSF size to avoid ringing
    reg = noise_power * (1.0 + 0.005 * psf_sigma_eff**2)
    # Floor: prevent near-zero regularisation
    reg = max(reg, 1e-4)
    recon = wiener_deconvolution(img_corrected, psf, noise_power=reg)

    # 4. Post-processing: very mild edge-preserving smoothing
    recon = gaussian_filter(recon, sigma=0.3)
    recon = np.clip(recon, 0.0, 1.0)

    return recon.astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# ── Image helpers ────────────────────────────────────────────────────────────

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        nonzero = arr[arr > 0]
        if len(nonzero) > 10:
            lo, hi = np.percentile(nonzero, [1, 99])
            arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, img_ideal, img_meas, recon, path: Path) -> None:
    """4-panel overview: GT | ideal | measured | reconstruction."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(img_ideal)
    ov[:, 2*tw:3*tw] = _r(img_meas)
    ov[:, 3*tw:4*tw] = _r(recon)
    _save_png(ov, path)


# ── Tier generation ──────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the fundus benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"fundus_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Fundus benchmark -- {tier} tier "
            f"(optical aberrations + illumination non-uniformity + media opacity "
            f"+ Poisson-Gaussian noise)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "fov_deg": 30.0,
            "pixel_size_um": PIXEL_SIZE_MM * 1000,
            "psf_scale_px_per_diopter": PSF_SCALE,
        })
        f.attrs["forward_model"] = (
            "y = N(B(PSF * x) * I + scatter)  "
            "where PSF=defocus, B=vignetting, I=illumination, "
            "scatter=media opacity haze, N=Poisson-Gaussian noise"
        )

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = fundus_forward_model(
                x_true,
                psf_sigma=mis["psf_sigma"],
                illumination_nonuniformity=mis["illumination_nonuniformity"],
                media_opacity=mis["media_opacity"],
                noise_sigma=mis["noise_sigma"],
                rng=rng,
            )

            image_ideal = result["image_ideal"]
            image_measured = result["image_measured"]
            psf = result["psf"]
            y = result["y"]
            H_ideal = result["H_ideal"]

            # CPU reconstruction: Wiener + illumination correction
            recon = reconstruct_cpu(image_measured, psf, mis["noise_sigma"])

            psnr = compute_psnr(x_true.astype(np.float32), recon)
            ssim = compute_ssim(x_true.astype(np.float32), recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("image_ideal", data=image_ideal,
                               compression="gzip")
            grp.create_dataset("image_measured", data=image_measured,
                               compression="gzip")
            grp.create_dataset("psf", data=psf, compression="gzip")
            grp.create_dataset("illumination_field",
                               data=result["illumination_field"],
                               compression="gzip")
            grp.create_dataset("scatter_field",
                               data=result["scatter_field"],
                               compression="gzip")
            grp.create_dataset("reconstruction_wiener", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psf_shape": list(psf.shape),
                "psnr_wiener": float(psnr),
                "ssim_wiener": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "gt.png")
            _save_png(image_measured, sample_dir / "measurement.png")
            _save_png(recon, sample_dir / "recon.png")
            _save_overview(x_true, image_ideal, image_measured, recon,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis,
                           "psnr_wiener": psnr, "ssim_wiener": ssim},
                          sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"psf_sigma={mis['psf_sigma']:.2f} px  "
                  f"illum={mis['illumination_nonuniformity']:.2f}  "
                  f"opacity={mis['media_opacity']:.3f}  "
                  f"noise={mis['noise_sigma']:.4f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# ── README ──────────────────────────────────────────────────────────────────

def _write_top_readme() -> None:
    txt = """# Fundus -- Retinal Fundus Photography

## Overview

Retinal fundus photography benchmark with realistic physics:
optical aberrations (defocus PSF) + illumination non-uniformity + media opacity
(cataract haze) + Gaussian sensor noise.

## Forward Model

```
y = H * (x * I(r)) + opacity * haze + noise

where:
    x           : ground-truth retinal image (256x256 grayscale, green channel)
    H           : point spread function (defocus aberration, Gaussian)
    I(r)        : illumination non-uniformity field (cosine-4th vignetting)
    opacity     : media opacity factor (cataract transmittance loss)
    haze        : low-frequency veiling glare from lens scatter
    noise       : additive Gaussian sensor noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| FOV | 30 degrees |
| pixel_size | 23.4 um/px |
| PSF_scale | ~15 px/diopter |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| defocus_diopters | Optical defocus | 0-2 D | 0-3.5 D | 0-5 D |
| illumination_nonuniformity | Vignetting strength | 0-0.15 | 0-0.25 | 0-0.40 |
| media_opacity | Cataract haze | 0-0.10 | 0-0.20 | 0-0.30 |
| noise_sigma | Sensor noise std | 0.01-0.03 | 0.01-0.05 | 0.01-0.08 |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Normal | 4/tier | Healthy retina: disc, vessels, macula, fovea |
| Pathological | 4/tier | Microaneurysms, hemorrhages, drusen, exudates |
| Varied anatomy | 4/tier | Variable disc size, CDR, vessel density |

## Dataset Structure

```
fundus/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal + measured + Wiener recon)
+-- dev/       20 samples (augmented, medium mismatch)
+-- hidden/    20 samples (adversarial: severe pathology, wide mismatch)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32            # Ground truth retinal image
+-- image_ideal (256, 256) float32       # Blurred + illumination (no noise/haze)
+-- image_measured (256, 256) float32    # Fully degraded fundus photograph
+-- psf (K, K) float32                   # Defocus PSF kernel
+-- illumination_field (256, 256) float32 # Non-uniform illumination map
+-- reconstruction_wiener (256, 256) float32 # Wiener deconvolution baseline
```

## CPU Baseline Reconstruction

Wiener deconvolution + illumination correction:
1. Estimate illumination from heavy low-pass filter of measured image
2. Divide out illumination estimate
3. Wiener deconvolution with PSF and noise regularisation
4. Post-filtering to suppress ringing

## References

1. Zhou et al. (2023) "A foundation model for generalizable disease detection
   from retinal images," Nature 622:156.
2. Li et al. (2024) "Fundus Image Enhancement via Structure-Preserving
   Diffusion Models," MICCAI 2024.
3. Frangi et al. (1998) "Multiscale vessel enhancement filtering," MICCAI 1998.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Gallery image generation ────────────────────────────────────────────────

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "fundus")

    # Load from public tier HDF5
    h5_path = BENCHMARK_DIR / "public" / "fundus_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: normal(0), pathological(4), varied(8), normal variant(2)
    gallery_sample_indices = [0, 4, 8, 2]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_sample_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found in HDF5, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            img_ideal = grp["image_ideal"][:]
            img_meas = grp["y"][:]  # measurement = y
            recon = grp["reconstruction_wiener"][:]
            illum = grp["illumination_field"][:]

            # gt.png -- ground truth retinal image
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- measured (degraded) fundus image
            _save_png(img_meas, scene_dir / "measurement_I.png")

            # measurement_II.png -- ideal (blurred, no noise) image
            _save_png(img_ideal, scene_dir / "measurement_II.png")

            # recon_I.png -- Wiener deconvolution reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- illumination field visualisation
            _save_png(illum, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} images saved to {scene_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Fundus Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")

    # ── Public tier (12 samples) — seed offset 0 ───────────────────────────
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=0)

    # ── Dev tier (20 samples) — seed offset 10000 ─────────────────────────
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # ── Hidden tier (20 samples) — seed offset 20000 ─────────────────────
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # ── README ──────────────────────────────────────────────────────────────
    _write_top_readme()

    # ── Gallery images ──────────────────────────────────────────────────────
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Fundus benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
