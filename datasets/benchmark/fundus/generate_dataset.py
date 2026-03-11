#!/usr/bin/env python3
"""Generate Retinal Fundus Photography benchmark dataset.

Forward model (fundus degradation):
    y = N(B(PSF * x) * I + scatter)

where:
    x              -- ground-truth retinal image (256x256 grayscale, green channel)
    PSF            -- optical point spread function (Gaussian blur, sigma in pixels)
    B              -- vignetting / cosine-4th illumination falloff
    I              -- non-uniform illumination pattern
    scatter        -- media opacity (cataract / vitreous haze)
    N              -- Poisson-Gaussian noise

Ground truth phantoms (256x256 grayscale):
    Synthetic retinal fundus images (green channel) with:
    - Bright optic disc (circular, off-center)
    - Vessel tree (branching arteries and veins from optic disc)
    - Dark fovea (centre of macula)
    - Background retinal texture (Perlin-like noise)
    - Pathological features for dev/hidden: microaneurysms, hemorrhages,
      drusen, hard exudates

Mismatch parameters:
    psf_sigma                  : optical blur (0.5-2.5 px public, 1.5-6 px hidden)
    illumination_nonuniformity : uneven lighting fraction (0-0.15 public, 0-0.40 hidden)
    media_opacity              : cataract haze level (0-0.06 public, 0-0.20 hidden)
    noise_sigma                : sensor noise std (0.005-0.015 public, 0.01-0.04 hidden)

Tiers:
    public : 12 samples (4 normal + 4 mild pathology + 4 varied anatomy)
    dev    : 20 samples (augmented, medium mismatch)
    hidden : 20 samples (adversarial, wide mismatch + pathologies)

CPU reconstruction: Wiener deconvolution + TV denoising + illumination correction
Expected baseline PSNR: ~20-23 dB

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

# ---- Physics constants -------------------------------------------------------

PIXEL_SIZE_MM = 6.0 / IMAGE_SIZE  # ~23.4 um
PSF_SCALE = 0.35 / PIXEL_SIZE_MM  # pixels per diopter (~15 px/D)

REFLECTANCE_BACKGROUND = 0.35
REFLECTANCE_DISC = 0.85
REFLECTANCE_FOVEA = 0.20
REFLECTANCE_ARTERY = 0.55
REFLECTANCE_VEIN = 0.25
REFLECTANCE_MACULA = 0.28

# ---- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "psf_sigma": {"min": 0.5, "max": 2.5, "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0, "max": 0.15, "unit": ""},
        "media_opacity": {"min": 0.0, "max": 0.06, "unit": ""},
        "noise_sigma": {"min": 0.005, "max": 0.015, "unit": ""},
    },
    "dev": {
        "psf_sigma": {"min": 1.0, "max": 4.0, "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0, "max": 0.25, "unit": ""},
        "media_opacity": {"min": 0.0, "max": 0.12, "unit": ""},
        "noise_sigma": {"min": 0.008, "max": 0.025, "unit": ""},
    },
    "hidden": {
        "psf_sigma": {"min": 1.5, "max": 6.0, "unit": "pixels"},
        "illumination_nonuniformity": {"min": 0.0, "max": 0.40, "unit": ""},
        "media_opacity": {"min": 0.0, "max": 0.20, "unit": ""},
        "noise_sigma": {"min": 0.010, "max": 0.040, "unit": ""},
    },
}


# ---- Perlin-like noise for retinal texture -----------------------------------

def _perlin_noise(H, W, scale, rng, octaves=4):
    noise = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    for _ in range(octaves):
        raw = rng.standard_normal((H, W))
        smoothed = gaussian_filter(raw, sigma=scale)
        noise += amplitude * smoothed
        amplitude *= 0.5
        scale *= 0.5
    lo, hi = noise.min(), noise.max()
    if hi - lo > 1e-10:
        noise = (noise - lo) / (hi - lo)
    return noise


# ---- Vessel tree generation --------------------------------------------------

def _draw_vessel_segment(canvas, y0, x0, y1, x1, width, reflectance):
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


def _grow_vessel_tree(canvas, start_y, start_x, angle_deg, length, width,
                      reflectance, depth, max_depth, rng):
    if depth > max_depth or length < 3 or width < 0.3:
        return
    angle_rad = np.radians(angle_deg)
    end_y = start_y + length * np.sin(angle_rad)
    end_x = start_x + length * np.cos(angle_rad)
    mid_y = (start_y + end_y) / 2 + rng.uniform(-length * 0.08, length * 0.08)
    mid_x = (start_x + end_x) / 2 + rng.uniform(-length * 0.08, length * 0.08)
    _draw_vessel_segment(canvas, start_y, start_x, mid_y, mid_x, width, reflectance)
    _draw_vessel_segment(canvas, mid_y, mid_x, end_y, end_x, width * 0.95, reflectance)
    n_branches = rng.integers(1, 4)
    for _ in range(n_branches):
        ba = angle_deg + rng.uniform(-40, 40)
        bl = length * rng.uniform(0.5, 0.75)
        bw = width * rng.uniform(0.55, 0.75)
        _grow_vessel_tree(canvas, end_y, end_x, ba, bl, bw,
                          reflectance, depth + 1, max_depth, rng)


# ---- Phantom generators -----------------------------------------------------

def _ellipse_mask(H, W, cy, cx, ry, rx):
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = ((yy - cy) / max(ry, 1e-6))**2 + ((xx - cx) / max(rx, 1e-6))**2
    return np.clip(1.0 - dist, 0.0, 1.0)


def _disc_mask(H, W, cy, cx, radius):
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    return np.clip(1.0 - (dist - radius), 0.0, 1.0)


def make_normal_fundus(H, W, seed, variant=0):
    rng = np.random.default_rng(seed)
    texture = _perlin_noise(H, W, scale=20.0 + variant * 2, rng=rng, octaves=4)
    image = np.full((H, W), REFLECTANCE_BACKGROUND, dtype=np.float64)
    image += (texture - 0.5) * 0.08
    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)
    image *= field_mask
    disc_cy = H * (0.48 + rng.uniform(-0.03, 0.03) + variant * 0.005)
    disc_cx = W * (0.70 + rng.uniform(-0.04, 0.04) - variant * 0.01)
    disc_r = 18 + rng.uniform(-3, 3) + variant * 0.5
    disc = _disc_mask(H, W, disc_cy, disc_cx, disc_r)
    cup_r = disc_r * rng.uniform(0.3, 0.5)
    cup = _disc_mask(H, W, disc_cy, disc_cx, cup_r)
    image = image * (1 - disc) + disc * REFLECTANCE_DISC
    image = image * (1 - cup * 0.3) + cup * 0.3 * 0.95
    mac_cy = H * (0.50 + rng.uniform(-0.02, 0.02))
    mac_cx = W * (0.42 + rng.uniform(-0.03, 0.03))
    mac_mask = _ellipse_mask(H, W, mac_cy, mac_cx, 35 + variant * 2, 40 + variant * 2)
    image = image * (1 - mac_mask * 0.25) + mac_mask * 0.25 * REFLECTANCE_MACULA
    fov_mask = _disc_mask(H, W, mac_cy, mac_cx, 8 + rng.uniform(-1, 1))
    image = image * (1 - fov_mask * 0.6) + fov_mask * 0.6 * REFLECTANCE_FOVEA
    for base_angle in [30, 150, 210, 330]:
        angle = base_angle + rng.uniform(-15, 15) + variant * 3
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, rng.uniform(80, 130),
                          rng.uniform(2.5, 4.0),
                          REFLECTANCE_ARTERY + rng.uniform(-0.05, 0.05),
                          0, 4 + variant % 2, rng)
    for base_angle in [60, 120, 240, 300]:
        angle = base_angle + rng.uniform(-15, 15) + variant * 2
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, rng.uniform(80, 130),
                          rng.uniform(3.0, 4.5),
                          REFLECTANCE_VEIN + rng.uniform(-0.03, 0.03),
                          0, 4 + variant % 2, rng)
    image *= field_mask
    image = gaussian_filter(image, sigma=0.5)
    return np.clip(image, 0.0, 1.0), f"normal_{variant:02d}"


def make_pathological_fundus(H, W, seed, variant=0, severity="mild"):
    rng = np.random.default_rng(seed)
    image, _ = make_normal_fundus(H, W, seed + 10000, variant)
    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)
    sscale = {"mild": 1.0, "moderate": 2.0, "severe": 3.0}[severity]
    for _ in range(int(rng.integers(3, 8) * sscale)):
        my, mx = rng.integers(40, H - 40), rng.integers(40, W - 40)
        if field_mask[my, mx] < 0.5:
            continue
        dot = _disc_mask(H, W, float(my), float(mx), rng.uniform(1.0, 2.5))
        image = image * (1 - dot * 0.8) + dot * 0.8 * 0.12
    for _ in range(int(rng.integers(1, 4) * sscale)):
        hy, hx = rng.integers(50, H - 50), rng.integers(50, W - 50)
        if field_mask[hy, hx] < 0.5:
            continue
        hem = _ellipse_mask(H, W, float(hy), float(hx),
                            rng.uniform(5, 15 * sscale), rng.uniform(5, 15 * sscale))
        noise = _perlin_noise(H, W, scale=10, rng=rng, octaves=2)
        hem *= (noise > 0.4).astype(np.float64)
        hem = np.clip(gaussian_filter(hem, sigma=2), 0, 1)
        image = image * (1 - hem * 0.7) + hem * 0.7 * 0.10
    for _ in range(int(rng.integers(2, 6) * sscale)):
        dy, dx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
        if field_mask[dy, dx] < 0.5:
            continue
        dru = _disc_mask(H, W, float(dy), float(dx), rng.uniform(3, 8))
        image = image * (1 - dru * 0.6) + dru * 0.6 * rng.uniform(0.60, 0.80)
    for _ in range(int(rng.integers(1, 4) * sscale)):
        ey, ex = rng.integers(60, H - 60), rng.integers(60, W - 60)
        if field_mask[ey, ex] < 0.5:
            continue
        exu = _ellipse_mask(H, W, float(ey), float(ex),
                            rng.uniform(3, 10), rng.uniform(3, 10))
        exu = gaussian_filter((exu > 0.3).astype(np.float64), sigma=0.8)
        image = image * (1 - exu * 0.7) + exu * 0.7 * rng.uniform(0.70, 0.90)
    return np.clip(image, 0.0, 1.0), f"pathological_{severity[0]}_{variant:02d}"


def make_varied_anatomy_fundus(H, W, seed, variant=0):
    rng = np.random.default_rng(seed)
    texture = _perlin_noise(H, W, scale=25 + variant * 3, rng=rng, octaves=4)
    bg_val = REFLECTANCE_BACKGROUND + rng.uniform(-0.05, 0.05)
    image = np.full((H, W), bg_val, dtype=np.float64) + (texture - 0.5) * 0.10
    field_mask = _disc_mask(H, W, H / 2, W / 2, H * 0.45)
    image *= field_mask
    disc_cy = H * (0.45 + rng.uniform(-0.05, 0.05))
    disc_cx = W * (0.65 + rng.uniform(-0.08, 0.08))
    disc_r = rng.uniform(14, 28)
    disc = _disc_mask(H, W, disc_cy, disc_cx, disc_r)
    cup = _disc_mask(H, W, disc_cy, disc_cx, disc_r * rng.uniform(0.2, 0.7))
    image = image * (1 - disc) + disc * REFLECTANCE_DISC
    image = image * (1 - cup * 0.4) + cup * 0.4 * 0.92
    mac_cy = H * (0.50 + rng.uniform(-0.04, 0.04))
    mac_cx = W * (0.40 + rng.uniform(-0.05, 0.05))
    mac_mask = _disc_mask(H, W, mac_cy, mac_cx, rng.uniform(25, 45))
    image = image * (1 - mac_mask * 0.3) + mac_mask * 0.3 * rng.uniform(0.18, 0.30)
    fov = _disc_mask(H, W, mac_cy, mac_cx, rng.uniform(5, 12))
    image = image * (1 - fov * 0.5) + fov * 0.5 * REFLECTANCE_FOVEA
    n_main = rng.integers(3, 7)
    for i in range(n_main):
        angle = i * (360 / n_main) + rng.uniform(-20, 20)
        refl = (REFLECTANCE_VEIN if i % 2 == 0 else REFLECTANCE_ARTERY) + rng.uniform(-0.04, 0.04)
        _grow_vessel_tree(image, disc_cy, disc_cx, angle, rng.uniform(70, 140),
                          rng.uniform(2.0, 4.5), refl, 0, rng.integers(3, 6), rng)
    image *= field_mask
    image = gaussian_filter(image, sigma=0.5)
    return np.clip(image, 0.0, 1.0), f"varied_{variant:02d}"


# ---- Phantom pools per tier --------------------------------------------------

def generate_phantoms_public(n=12):
    phantoms = []
    for i in range(4):
        phantoms.append(make_normal_fundus(IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_pathological_fundus(IMAGE_SIZE, IMAGE_SIZE,
                                                 seed=200 + i, variant=i, severity="mild"))
    for i in range(4):
        phantoms.append(make_varied_anatomy_fundus(IMAGE_SIZE, IMAGE_SIZE, seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n=20):
    phantoms = []
    rng_aug = np.random.default_rng(5000)
    gens = [
        lambda H, W, s, v: make_normal_fundus(H, W, s, v),
        lambda H, W, s, v: make_pathological_fundus(H, W, s, v, severity="moderate"),
        lambda H, W, s, v: make_varied_anatomy_fundus(H, W, s, v),
    ]
    for i in range(n):
        image, name = gens[i % 3](IMAGE_SIZE, IMAGE_SIZE, 500 + i, i)
        image = nd_rotate(image, float(rng_aug.uniform(5, 355)),
                          reshape=False, mode='constant', cval=0.0)
        if rng_aug.random() < 0.5:
            image = np.fliplr(image)
        phantoms.append((np.clip(image, 0.0, 1.0), f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n=20):
    phantoms = []
    rng_aug = np.random.default_rng(8000)
    gens = [
        lambda H, W, s, v: make_normal_fundus(H, W, s, v),
        lambda H, W, s, v: make_pathological_fundus(H, W, s, v, severity="severe"),
        lambda H, W, s, v: make_varied_anatomy_fundus(H, W, s, v),
    ]
    for i in range(n):
        image, name = gens[i % 3](IMAGE_SIZE, IMAGE_SIZE, 800 + i, i + 10)
        image = nd_rotate(image, float(rng_aug.uniform(10, 350)),
                          reshape=False, mode='constant', cval=0.0)
        if rng_aug.random() < 0.7:
            image = np.fliplr(image)
        if rng_aug.random() < 0.5:
            image = np.flipud(image)
        zoom_f = float(rng_aug.uniform(0.80, 1.20))
        zoomed = nd_zoom(image, zoom_f, order=1)
        zh, zw = zoomed.shape
        if zh >= IMAGE_SIZE and zw >= IMAGE_SIZE:
            y0 = (zh - IMAGE_SIZE) // 2
            x0 = (zw - IMAGE_SIZE) // 2
            image = zoomed[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE]
        else:
            out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=zoomed.dtype)
            y0 = (IMAGE_SIZE - zh) // 2
            x0 = (IMAGE_SIZE - zw) // 2
            out[y0:y0 + zh, x0:x0 + zw] = zoomed
            image = out
        phantoms.append((np.clip(image, 0.0, 1.0), f"hidden_{name}"))
    return phantoms


# ---- PSF generation ----------------------------------------------------------

def make_defocus_psf(psf_sigma, size=31):
    if size % 2 == 0:
        size += 1
    sigma = min(max(0.3, psf_sigma), size / 3.0)
    center = size // 2
    yy = np.arange(size, dtype=np.float64) - center
    yy, xx = np.meshgrid(yy, yy, indexing='ij')
    psf = np.exp(-(yy**2 + xx**2) / (2 * sigma**2 + 1e-10))
    psf /= psf.sum()
    return psf


# ---- Illumination field ------------------------------------------------------

def make_illumination_field(H, W, nonuniformity, rng):
    cy = H / 2 + rng.uniform(-5, 5)
    cx = W / 2 + rng.uniform(-5, 5)
    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    r = np.sqrt(yy**2 + xx**2)
    r_max = np.sqrt(cy**2 + cx**2) + 1e-6
    cos_theta = np.clip(1.0 - 0.5 * (r / r_max)**2, 0.1, 1.0)
    return 1.0 - nonuniformity * (1.0 - cos_theta**4)


# ---- Forward model -----------------------------------------------------------

def _make_H_ideal(psf, H, W):
    psf_pad = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    y0, x0 = (H - kh) // 2, (W - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)
    return psf_pad.astype(np.float32)


def fundus_forward_model(x_true, psf_sigma, illumination_nonuniformity,
                         media_opacity, noise_sigma, rng):
    H, W = x_true.shape
    psf_size = min(61, max(11, int(6 * psf_sigma + 1) | 1))
    if psf_size % 2 == 0:
        psf_size += 1
    psf = make_defocus_psf(psf_sigma, size=psf_size)
    illum = make_illumination_field(H, W, illumination_nonuniformity, rng)
    x_illum = x_true * illum
    if psf_sigma > 0.3:
        blurred = fftconvolve(x_illum, psf, mode='same')
    else:
        blurred = x_illum.copy()
    image_ideal = np.clip(blurred, 0.0, 1.0).astype(np.float32)
    scatter = np.zeros((H, W), dtype=np.float32)
    if media_opacity > 0.001:
        haze = gaussian_filter(blurred, sigma=40) * 0.6 + 0.4
        scatter = (media_opacity * haze).astype(np.float32)
        hazy = (1 - media_opacity) * blurred + media_opacity * haze
    else:
        hazy = blurred
    peak = max(1.0 / (noise_sigma**2 + 1e-10), 100.0)
    poisson = rng.poisson(np.clip(hazy * peak, 0.01, None)).astype(np.float64) / peak
    readout = rng.normal(0, noise_sigma * 0.3, (H, W))
    measured = np.clip(poisson + readout, 0.0, 1.0).astype(np.float32)
    return {
        "image_ideal": image_ideal,
        "image_measured": measured,
        "y": measured,
        "H_ideal": _make_H_ideal(psf, H, W),
        "psf": psf.astype(np.float32),
        "illumination_field": illum.astype(np.float32),
        "scatter_field": scatter,
    }


# ---- CPU reconstruction -----------------------------------------------------

def wiener_deconvolution(image, psf, noise_power=0.01):
    H_img, W_img = image.shape
    psf_pad = np.zeros((H_img, W_img), dtype=np.float64)
    kh, kw = psf.shape
    y0, x0 = (H_img - kh) // 2, (W_img - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)
    Y = np.fft.fft2(image.astype(np.float64))
    Hf = np.fft.fft2(psf_pad)
    return np.real(np.fft.ifft2(np.conj(Hf) / (np.abs(Hf)**2 + noise_power) * Y))


def _tv_denoise(image, weight, n_iter=30):
    img = image.astype(np.float64)
    H, W = img.shape
    px = np.zeros((H, W), dtype=np.float64)
    py = np.zeros((H, W), dtype=np.float64)
    tau = 0.25
    for _ in range(n_iter):
        div_p = np.zeros_like(img)
        div_p[:-1, :] += px[:-1, :]
        div_p[1:, :] -= px[:-1, :]
        div_p[:, :-1] += py[:, :-1]
        div_p[:, 1:] -= py[:, :-1]
        u = div_p - img / weight
        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]
        denom = 1.0 + tau * np.sqrt(gx**2 + gy**2)
        px = (px + tau * gx) / denom
        py = (py + tau * gy) / denom
    div_p = np.zeros_like(img)
    div_p[:-1, :] += px[:-1, :]
    div_p[1:, :] -= px[:-1, :]
    div_p[:, :-1] += py[:, :-1]
    div_p[:, 1:] -= py[:, :-1]
    return img - weight * div_p


def reconstruct_cpu(image_measured, psf, noise_sigma):
    img = image_measured.astype(np.float64)
    illum_est = gaussian_filter(img, sigma=50)
    illum_est = np.clip(illum_est, 0.05, None)
    illum_est /= (illum_est.mean() + 1e-8)
    img_c = np.clip(img / illum_est, 0.0, 1.0)
    kh = psf.shape[0]
    yy = np.arange(kh, dtype=np.float64) - kh // 2
    psf64 = psf.astype(np.float64)
    psf_var = np.sum(psf64 * yy[:, None]**2) + np.sum(psf64 * yy[None, :]**2)
    psf_sig = max(0.5, np.sqrt(max(psf_var, 0.0)))
    reg = max(noise_sigma**2 * (1.0 + 0.003 * psf_sig**2), 5e-5)
    recon = np.clip(wiener_deconvolution(img_c, psf, noise_power=reg), 0.0, 1.0)
    tv_w = max(0.02, noise_sigma * 3.0)
    recon = np.clip(_tv_denoise(recon, weight=tv_w, n_iter=40), 0.0, 1.0)
    return recon.astype(np.float32)


# ---- Metrics -----------------------------------------------------------------

def compute_psnr(gt, recon):
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64))**2)
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    return float(10 * np.log10(dr**2 / mse)) if dr > 1e-12 else 0.0


def compute_ssim(gt, recon):
    g, r = gt.astype(np.float64), recon.astype(np.float64)
    dr = g.max() - g.min()
    if dr < 1e-12:
        return 0.0
    c1, c2 = (0.01 * dr)**2, (0.03 * dr)**2
    mx, my = g.mean(), r.mean()
    vx, vy = g.var(), r.var()
    cov = np.mean((g - mx) * (r - my))
    return float((2 * mx * my + c1) * (2 * cov + c2) / ((mx**2 + my**2 + c1) * (vx + vy + c2)))


# ---- Image helpers -----------------------------------------------------------

def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr, path):
    Image.fromarray(np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L").save(str(path))


def _save_overview(x_true, img_ideal, img_meas, recon, path):
    th, tw = 128, 128
    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0
    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(img_ideal)
    ov[:, 2 * tw:3 * tw] = _r(img_meas)
    ov[:, 3 * tw:4 * tw] = _r(recon)
    _save_png(ov, path)


# ---- Tier generation ---------------------------------------------------------

def sample_mismatch(rng, spec):
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier, phantoms, base_seed):
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    h5_path = tier_dir / f"fundus_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs, all_ssims = [], []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Fundus benchmark -- {tier} tier "
            f"(optical blur + illumination non-uniformity + media opacity "
            f"+ Poisson-Gaussian noise)")
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE, "fov_deg": 30.0,
            "pixel_size_um": PIXEL_SIZE_MM * 1000,
            "psf_scale_px_per_diopter": PSF_SCALE})
        f.attrs["forward_model"] = (
            "y = N(B(PSF * x) * I + scatter)")

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] {key} ({scene_name})...", end="", flush=True)
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis
            result = fundus_forward_model(
                x_true, psf_sigma=mis["psf_sigma"],
                illumination_nonuniformity=mis["illumination_nonuniformity"],
                media_opacity=mis["media_opacity"],
                noise_sigma=mis["noise_sigma"], rng=rng)
            recon = reconstruct_cpu(result["image_measured"], result["psf"],
                                    mis["noise_sigma"])
            psnr = compute_psnr(x_true.astype(np.float32), recon)
            ssim = compute_ssim(x_true.astype(np.float32), recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32), compression="gzip")
            grp.create_dataset("y", data=result["y"], compression="gzip")
            grp.create_dataset("H_ideal", data=result["H_ideal"], compression="gzip")
            grp.create_dataset("image_ideal", data=result["image_ideal"], compression="gzip")
            grp.create_dataset("image_measured", data=result["image_measured"], compression="gzip")
            grp.create_dataset("psf", data=result["psf"], compression="gzip")
            grp.create_dataset("illumination_field", data=result["illumination_field"], compression="gzip")
            grp.create_dataset("scatter_field", data=result["scatter_field"], compression="gzip")
            grp.create_dataset("reconstruction_wiener", data=recon, compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name, "shape": list(x_true.shape),
                "psf_shape": list(result["psf"].shape),
                "psnr_wiener": float(psnr), "ssim_wiener": float(ssim)})
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            sd = images_dir / f"sample_{idx:02d}_{scene_name}"
            sd.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sd / "gt.png")
            _save_png(result["image_measured"], sd / "measurement.png")
            _save_png(recon, sd / "recon.png")
            _save_overview(x_true, result["image_ideal"], result["image_measured"],
                           recon, sd / "overview.png")
            with open(sd / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "psnr_wiener": psnr,
                           "ssim_wiener": ssim}, sf, indent=2)
            print(f"  PSNR={psnr:.2f} SSIM={ssim:.3f}")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={np.mean(all_psnrs):.2f} dB | Mean SSIM={np.mean(all_ssims):.3f}")


# ---- Gallery -----------------------------------------------------------------

def generate_gallery_images():
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "fundus")
    h5_path = BENCHMARK_DIR / "public" / "fundus_challenge_public.h5"
    if not h5_path.exists():
        return
    for scene_idx, sample_idx in enumerate([0, 4, 8, 2]):
        scene_dir = gallery_base / f"scene_{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "r") as f:
            key = f"sample_{sample_idx:02d}"
            if key not in f:
                continue
            grp = f[key]
            _save_png(grp["x_true"][:], scene_dir / "gt.png")
            _save_png(grp["y"][:], scene_dir / "measurement_I.png")
            _save_png(grp["image_ideal"][:], scene_dir / "measurement_II.png")
            _save_png(grp["reconstruction_wiener"][:], scene_dir / "recon_I.png")
            _save_png(grp["illumination_field"][:], scene_dir / "recon_II.png")
        print(f"  [gallery] scene_{scene_idx:02d} saved")


# ---- Main --------------------------------------------------------------------

def main():
    print("Fundus Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
    print("Generating public tier (12 samples)...")
    generate_tier("public", generate_phantoms_public(12), base_seed=0)
    print("\nGenerating dev tier (20 samples)...")
    generate_tier("dev", generate_phantoms_dev(20), base_seed=10000)
    print("\nGenerating hidden tier (20 samples)...")
    generate_tier("hidden", generate_phantoms_hidden(20), base_seed=20000)
    print("\nGenerating gallery images...")
    generate_gallery_images()
    print(f"\n{'=' * 68}\nFundus benchmark complete!\n{'=' * 68}")


if __name__ == "__main__":
    main()
