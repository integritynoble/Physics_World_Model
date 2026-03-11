#!/usr/bin/env python3
"""Generate the CBCT (Cone-Beam CT) benchmark dataset.

Follows the exact same structure as the CT benchmark dataset.
Uses skimage.transform.radon for fast projection (fan-beam approximation).

Tiers: Public 12 samples, Dev 20 samples, Hidden 20 samples.
Ground truth: 256x256 dental/head phantoms.
Mismatch: scatter_fraction, truncation_fov_factor, ring_artifact_amplitude,
          rotation_offset_deg.

Usage:
    cd datasets/benchmark/cbct
    python3 generate_dataset_v2.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from skimage.transform import radon, iradon

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_VIEWS = 180
I0 = 5000.0
SIGMA_RO = 3.0
MU_SCALE = 0.045

# -- Attenuation coefficients -------------------------------------------------

MU_AIR = 0.00
MU_FAT = 0.18
MU_SOFT = 0.25
MU_MUSCLE = 0.28
MU_CARTILAGE = 0.30
MU_DENTIN = 0.55
MU_BONE = 0.65
MU_ENAMEL = 0.85
MU_METAL = 1.00

# -- Mismatch spec ranges per tier --------------------------------------------

SPEC = {
    "public": {
        "scatter_fraction": {"min": 0.20, "max": 0.35, "unit": ""},
        "truncation_fov_factor": {"min": 0.85, "max": 1.00, "unit": ""},
        "ring_artifact_amplitude": {"min": 0.00, "max": 0.02, "unit": ""},
        "rotation_offset_deg": {"min": 0.00, "max": 1.00, "unit": "degrees"},
    },
    "dev": {
        "scatter_fraction": {"min": 0.25, "max": 0.45, "unit": ""},
        "truncation_fov_factor": {"min": 0.78, "max": 1.00, "unit": ""},
        "ring_artifact_amplitude": {"min": 0.00, "max": 0.03, "unit": ""},
        "rotation_offset_deg": {"min": 0.00, "max": 2.00, "unit": "degrees"},
    },
    "hidden": {
        "scatter_fraction": {"min": 0.30, "max": 0.60, "unit": ""},
        "truncation_fov_factor": {"min": 0.70, "max": 1.00, "unit": ""},
        "ring_artifact_amplitude": {"min": 0.00, "max": 0.05, "unit": ""},
        "rotation_offset_deg": {"min": 0.00, "max": 3.00, "unit": "degrees"},
    },
}


# =============================================================================
# Fast Radon projection using skimage
# =============================================================================


def radon_project(image, angles_deg):
    """Radon transform using skimage (fast). Returns (n_views, n_det)."""
    sino = radon(image.astype(np.float64), theta=angles_deg, circle=False)
    # radon returns (n_det, n_angles), transpose to (n_angles, n_det)
    return sino.T.astype(np.float32)


# =============================================================================
# CBCT-specific mismatch effects
# =============================================================================


def apply_scatter(sino, scatter_fraction, rng):
    """Add smooth scatter background."""
    if scatter_fraction < 1e-6:
        return sino
    mean_signal = max(float(sino.mean()), 1e-8)
    scatter_base = gaussian_filter(sino.astype(np.float64), sigma=[3.0, 8.0])
    scatter_norm = scatter_base / max(float(scatter_base.max()), 1e-8)
    scatter = scatter_fraction * mean_signal * scatter_norm
    return (sino + scatter).astype(np.float32)


def apply_truncation(sino, fov_factor):
    """Simulate FOV truncation."""
    if fov_factor >= 0.999:
        return sino
    n_det = sino.shape[1]
    active_det = int(n_det * fov_factor)
    pad = (n_det - active_det) // 2
    mask = np.ones(n_det, dtype=np.float32)
    if pad > 0:
        ramp = np.linspace(0.0, 1.0, max(pad, 2))
        mask[:pad] = ramp
        mask[-pad:] = ramp[::-1]
    return sino * mask[np.newaxis, :]


def apply_ring_artifact(sino, amplitude, rng):
    """Simulate ring artifacts."""
    if amplitude < 1e-6:
        return sino
    n_det = sino.shape[1]
    gain_offset = rng.normal(0.0, amplitude, size=n_det).astype(np.float32)
    gain_offset = gaussian_filter(gain_offset, sigma=1.5).astype(np.float32)
    return sino + gain_offset[np.newaxis, :]


def apply_mismatch(x, angles_deg, scatter_fraction, truncation_fov_factor,
                   ring_artifact_amplitude, rotation_offset_deg, rng,
                   i0=I0, sigma_ro=SIGMA_RO):
    """Apply all four CBCT mismatch effects + noise."""
    # 1. Re-project with rotation offset
    shifted_angles = angles_deg + rotation_offset_deg
    p_geo = radon_project(x, shifted_angles)

    # 2. Scale to physical nepers
    p_phys = p_geo * MU_SCALE

    # 3. Add scatter
    p_scatter = apply_scatter(p_phys, scatter_fraction, rng)

    # 4. Truncation
    p_trunc = apply_truncation(p_scatter, truncation_fov_factor)

    # 5. Ring artifacts
    p_ring = apply_ring_artifact(p_trunc, ring_artifact_amplitude, rng)

    # 6. Noise (Beer-Lambert + Poisson + readout)
    p_clamped = np.clip(p_ring, 0.0, 20.0)
    i_expect = i0 * np.exp(-p_clamped)
    i_noisy = rng.poisson(np.maximum(i_expect, 1e-3)).astype(np.float64)
    i_noisy += rng.normal(0.0, sigma_ro, i_noisy.shape)
    i_noisy = np.maximum(i_noisy, 1.0)

    return (-np.log(i_noisy / i0)).astype(np.float32)


# =============================================================================
# Dental / head phantom generators (2D, 256x256)
# =============================================================================


def _grid(shape):
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    return np.meshgrid(yy, xx, indexing="ij")


def _ellipse(yy, xx, cy, cx, ry, rx, angle_deg=0.0):
    t = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    dy, dx = yy - cy, xx - cx
    yr = cos_t * dy + sin_t * dx
    xr = -sin_t * dy + cos_t * dx
    return ((yr / max(ry, 1e-6))**2 + (xr / max(rx, 1e-6))**2 <= 1.0).astype(np.float32)


def _soft_ellipse(yy, xx, cy, cx, ry, rx, angle_deg=0.0, sigma=2.0):
    return gaussian_filter(_ellipse(yy, xx, cy, cx, ry, rx, angle_deg), sigma=sigma)


def _ring_shape(yy, xx, cy, cx, r0, thickness):
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    return np.exp(-((r - r0)**2) / (2 * thickness**2)).astype(np.float32)


def _fbm(shape, rng, octaves=4, persistence=0.55, base_sigma=6.0):
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp, total, sigma = 1.0, 0.0, base_sigma
    for _ in range(octaves):
        n = rng.standard_normal((h, w)).astype(np.float32)
        out += amp * gaussian_filter(n, sigma=sigma)
        total += amp
        amp *= persistence
        sigma *= 2.0
    out /= max(total, 1e-6)
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def _dental_panoramic(rng, shape):
    """Dental panoramic: mandible, teeth, tongue, airway."""
    yy, xx = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    face_ry = rng.uniform(0.72, 0.85)
    face_rx = rng.uniform(0.60, 0.72)
    face = _soft_ellipse(yy, xx, 0.0, 0.0, face_ry, face_rx, sigma=3.0)
    x += MU_SOFT * (face > 0.3)

    fat_ry = face_ry - rng.uniform(0.04, 0.08)
    fat_rx = face_rx - rng.uniform(0.03, 0.06)
    inner = _soft_ellipse(yy, xx, 0.0, 0.0, fat_ry, fat_rx, sigma=2.5)
    fat_r = np.clip(face - inner, 0.0, 1.0)
    x += (MU_FAT - MU_SOFT) * (fat_r > 0.2)

    mand_r0 = rng.uniform(0.38, 0.48)
    mand_thick = rng.uniform(0.03, 0.06)
    ring = _ring_shape(yy, xx, rng.uniform(0.05, 0.15), 0.0, mand_r0, mand_thick)
    ant_mask = (yy < rng.uniform(0.25, 0.40)).astype(np.float32)
    x = np.where((ring * ant_mask) > 0.3, MU_BONE + rng.uniform(-0.05, 0.05), x)

    for sign in [+1, -1]:
        rc = sign * rng.uniform(0.35, 0.48)
        rm = _soft_ellipse(yy, xx, rng.uniform(0.10, 0.30), rc,
                           rng.uniform(0.15, 0.25), rng.uniform(0.04, 0.07),
                           angle_deg=sign * rng.uniform(-15, 15), sigma=2.0)
        x = np.where(rm > 0.4, MU_BONE + rng.uniform(-0.05, 0.05), x)

    n_teeth = rng.integers(8, 17)
    arch_angles = np.linspace(-2.2, 2.2, n_teeth)
    for ang in arch_angles:
        if abs(ang) > 1.5:
            t_cy = rng.uniform(0.12, 0.30)
            t_cx = np.sign(ang) * rng.uniform(0.28, 0.42)
        else:
            t_cy = mand_r0 * 0.85 * np.sin(ang) * 0.35 + rng.uniform(-0.02, 0.10)
            t_cx = mand_r0 * 0.85 * np.cos(ang) * 0.85 + rng.uniform(-0.02, 0.02)

        er = rng.uniform(0.018, 0.032)
        em = _soft_ellipse(yy, xx, t_cy, t_cx, er, er * rng.uniform(0.8, 1.2), sigma=1.0)
        x = np.where(em > 0.4, MU_ENAMEL + rng.uniform(-0.05, 0.05), x)
        dr = er * rng.uniform(0.5, 0.7)
        dm = _soft_ellipse(yy, xx, t_cy, t_cx, dr, dr * rng.uniform(0.8, 1.2), sigma=0.8)
        x = np.where(dm > 0.5, MU_DENTIN + rng.uniform(-0.03, 0.03), x)
        pr = dr * rng.uniform(0.3, 0.5)
        pm = _soft_ellipse(yy, xx, t_cy, t_cx, pr, pr, sigma=0.5)
        x = np.where(pm > 0.5, MU_SOFT * rng.uniform(0.8, 1.2), x)

    tm = _soft_ellipse(yy, xx, rng.uniform(-0.10, 0.05), rng.uniform(-0.03, 0.03),
                       rng.uniform(0.12, 0.20), rng.uniform(0.15, 0.25), sigma=4.0)
    x = np.where(tm > 0.4, MU_MUSCLE + rng.uniform(-0.02, 0.02), x)

    am = _soft_ellipse(yy, xx, rng.uniform(0.25, 0.45), rng.uniform(-0.03, 0.03),
                       rng.uniform(0.06, 0.12), rng.uniform(0.05, 0.10), sigma=2.0)
    x = np.where(am > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    tx = _fbm(shape, rng, octaves=4, base_sigma=10.0)
    face_m = (face > 0.3).astype(np.float32)
    x = np.clip(x + 0.015 * (tx - 0.5) * face_m, 0.0, 1.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _head_axial(rng, shape):
    """Head axial: skull, brain, ventricles, sinuses."""
    yy, xx = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    scalp_ry = rng.uniform(0.72, 0.85)
    scalp_rx = rng.uniform(0.62, 0.75)
    scalp = _soft_ellipse(yy, xx, rng.uniform(-0.03, 0.03), 0.0, scalp_ry, scalp_rx, sigma=3.0)
    x += MU_SOFT * (scalp > 0.3)

    skull_thick = rng.uniform(0.03, 0.06)
    skull_outer = _soft_ellipse(yy, xx, 0.0, 0.0, scalp_ry - 0.02, scalp_rx - 0.02, sigma=2.0)
    skull_inner = _soft_ellipse(yy, xx, 0.0, 0.0,
                                scalp_ry - 0.02 - skull_thick,
                                scalp_rx - 0.02 - skull_thick, sigma=2.0)
    skull_ring = np.clip(skull_outer - skull_inner, 0.0, 1.0)
    x = np.where(skull_ring > 0.3, MU_BONE + rng.uniform(-0.05, 0.08), x)

    brain_ry = scalp_ry - 0.02 - skull_thick - 0.02
    brain_rx = scalp_rx - 0.02 - skull_thick - 0.02
    brain = _soft_ellipse(yy, xx, 0.0, 0.0, brain_ry, brain_rx, sigma=3.0)
    gm_val = MU_SOFT + rng.uniform(0.00, 0.03)
    x = np.where((brain > 0.3) & (x < MU_BONE * 0.5), gm_val, x)

    wm_ry = brain_ry * rng.uniform(0.65, 0.80)
    wm_rx = brain_rx * rng.uniform(0.65, 0.80)
    wm = _soft_ellipse(yy, xx, 0.0, 0.0, wm_ry, wm_rx, sigma=4.0)
    x = np.where((wm > 0.4) & (x < MU_BONE * 0.5), MU_SOFT - rng.uniform(0.01, 0.03), x)

    for sign in [+1, -1]:
        vm = _soft_ellipse(yy, xx, rng.uniform(-0.05, 0.05), sign * rng.uniform(0.06, 0.14),
                           rng.uniform(0.04, 0.08), rng.uniform(0.06, 0.12),
                           angle_deg=sign * rng.uniform(-10, 10), sigma=2.0)
        x = np.where(vm > 0.45, MU_AIR + rng.uniform(0.04, 0.08), x)

    fm = _soft_ellipse(yy, xx, 0.0, 0.0, brain_ry * 0.95, rng.uniform(0.005, 0.012), sigma=1.0)
    x = np.where(fm > 0.5, MU_CARTILAGE + rng.uniform(-0.02, 0.02), x)

    if rng.random() < 0.6:
        for sign in [+1, -1]:
            sm = _soft_ellipse(yy, xx, rng.uniform(-0.50, -0.35), sign * rng.uniform(0.04, 0.14),
                               rng.uniform(0.04, 0.08), rng.uniform(0.03, 0.06), sigma=2.0)
            x = np.where(sm > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    for sign in [+1, -1]:
        if rng.random() < 0.5:
            mm = _soft_ellipse(yy, xx, rng.uniform(0.30, 0.50), sign * rng.uniform(0.30, 0.50),
                               rng.uniform(0.03, 0.07), rng.uniform(0.02, 0.05), sigma=1.5)
            x = np.where(mm > 0.45, MU_AIR + rng.uniform(0.02, 0.05), x)

    tx = _fbm(shape, rng, octaves=5, base_sigma=8.0)
    brain_m = (brain > 0.3).astype(np.float32)
    x = np.clip(x + 0.012 * (tx - 0.5) * brain_m, 0.0, 1.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _dental_mixed(rng, shape):
    """Mixed dental: maxilla + mandible + sinuses."""
    yy, xx = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    face_ry = rng.uniform(0.74, 0.88)
    face_rx = rng.uniform(0.62, 0.78)
    face = _soft_ellipse(yy, xx, 0.0, 0.0, face_ry, face_rx, sigma=3.0)
    x += MU_SOFT * (face > 0.3)
    fat_ry = face_ry - rng.uniform(0.04, 0.07)
    fat_rx = face_rx - rng.uniform(0.03, 0.05)
    inner = _soft_ellipse(yy, xx, 0.0, 0.0, fat_ry, fat_rx, sigma=2.5)
    x += (MU_FAT - MU_SOFT) * (np.clip(face - inner, 0, 1) > 0.2)

    max_r0 = rng.uniform(0.30, 0.40)
    ring = _ring_shape(yy, xx, rng.uniform(-0.15, -0.05), 0.0, max_r0, rng.uniform(0.03, 0.05))
    x = np.where((ring * (yy < rng.uniform(0.10, 0.25))) > 0.3, MU_BONE + rng.uniform(-0.04, 0.04), x)

    mand_r0 = rng.uniform(0.35, 0.45)
    ring2 = _ring_shape(yy, xx, rng.uniform(0.10, 0.20), 0.0, mand_r0, rng.uniform(0.03, 0.06))
    x = np.where((ring2 * (yy < rng.uniform(0.35, 0.50))) > 0.3, MU_BONE + rng.uniform(-0.04, 0.05), x)

    for i in range(rng.integers(6, 11)):
        ang = -1.8 + i * 3.6 / max(rng.integers(6, 11) - 1, 1)
        t_cy = -0.08 + max_r0 * 0.7 * np.sin(ang) * 0.3 + rng.uniform(-0.02, 0.02)
        t_cx = max_r0 * 0.7 * np.cos(ang) * 0.7 + rng.uniform(-0.02, 0.02)
        er = rng.uniform(0.015, 0.028)
        em = _soft_ellipse(yy, xx, t_cy, t_cx, er, er * rng.uniform(0.8, 1.2), sigma=0.8)
        x = np.where(em > 0.4, MU_ENAMEL + rng.uniform(-0.05, 0.05), x)

    for i in range(rng.integers(6, 11)):
        ang = -1.8 + i * 3.6 / max(rng.integers(6, 11) - 1, 1)
        t_cy = 0.12 + mand_r0 * 0.65 * np.sin(ang) * 0.3 + rng.uniform(-0.02, 0.02)
        t_cx = mand_r0 * 0.65 * np.cos(ang) * 0.7 + rng.uniform(-0.02, 0.02)
        er = rng.uniform(0.015, 0.028)
        em = _soft_ellipse(yy, xx, t_cy, t_cx, er, er * rng.uniform(0.8, 1.2), sigma=0.8)
        x = np.where(em > 0.4, MU_ENAMEL + rng.uniform(-0.05, 0.05), x)

    for sign in [+1, -1]:
        sm = _soft_ellipse(yy, xx, rng.uniform(-0.30, -0.15), sign * rng.uniform(0.12, 0.28),
                           rng.uniform(0.06, 0.12), rng.uniform(0.05, 0.10), sigma=2.5)
        x = np.where(sm > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    nm = _soft_ellipse(yy, xx, rng.uniform(-0.25, -0.10), 0.0,
                       rng.uniform(0.04, 0.08), rng.uniform(0.08, 0.15), sigma=2.0)
    x = np.where(nm > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    tm = _soft_ellipse(yy, xx, rng.uniform(0.0, 0.10), 0.0,
                       rng.uniform(0.10, 0.16), rng.uniform(0.12, 0.20), sigma=3.5)
    x = np.where(tm > 0.4, MU_MUSCLE + rng.uniform(-0.02, 0.02), x)

    am = _soft_ellipse(yy, xx, rng.uniform(0.30, 0.50), 0.0,
                       rng.uniform(0.05, 0.10), rng.uniform(0.04, 0.08), sigma=2.0)
    x = np.where(am > 0.45, MU_AIR + rng.uniform(0.01, 0.02), x)

    tx = _fbm(shape, rng, octaves=4, base_sigma=10.0)
    x = np.clip(x + 0.012 * (tx - 0.5) * (face > 0.3).astype(np.float32), 0.0, 1.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _head_lower(rng, shape):
    """Head lower: skull base, petrous bones, TMJ, cervical spine."""
    yy, xx = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    head_ry = rng.uniform(0.70, 0.84)
    head_rx = rng.uniform(0.58, 0.72)
    head = _soft_ellipse(yy, xx, 0.0, 0.0, head_ry, head_rx, sigma=3.0)
    x += MU_SOFT * (head > 0.3)

    skull_thick = rng.uniform(0.04, 0.07)
    skull_outer = _soft_ellipse(yy, xx, 0.0, 0.0, head_ry - 0.02, head_rx - 0.02, sigma=2.0)
    skull_inner = _soft_ellipse(yy, xx, 0.0, 0.0,
                                head_ry - 0.02 - skull_thick,
                                head_rx - 0.02 - skull_thick, sigma=2.0)
    x = np.where(np.clip(skull_outer - skull_inner, 0, 1) > 0.3, MU_BONE + rng.uniform(-0.03, 0.08), x)

    for sign in [+1, -1]:
        pm = _soft_ellipse(yy, xx, rng.uniform(0.15, 0.35), sign * rng.uniform(0.25, 0.40),
                           rng.uniform(0.06, 0.10), rng.uniform(0.04, 0.07),
                           angle_deg=sign * rng.uniform(20, 45), sigma=2.0)
        x = np.where(pm > 0.4, MU_BONE + rng.uniform(0.05, 0.15), x)

    for sign in [+1, -1]:
        tm = _soft_ellipse(yy, xx, rng.uniform(-0.05, 0.15), sign * rng.uniform(0.38, 0.52),
                           rng.uniform(0.03, 0.06), rng.uniform(0.03, 0.06), sigma=1.5)
        x = np.where(tm > 0.45, MU_BONE + rng.uniform(-0.05, 0.05), x)

    bm = _soft_ellipse(yy, xx, rng.uniform(0.10, 0.25), 0.0,
                       rng.uniform(0.15, 0.25), rng.uniform(0.12, 0.20), sigma=3.0)
    x = np.where((bm > 0.4) & (x < MU_BONE * 0.5), MU_SOFT + rng.uniform(-0.02, 0.02), x)

    sm = _soft_ellipse(yy, xx, rng.uniform(0.05, 0.20), rng.uniform(-0.03, 0.03),
                       rng.uniform(0.04, 0.08), rng.uniform(0.05, 0.10), sigma=2.0)
    x = np.where(sm > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    sp_cy = rng.uniform(0.40, 0.55)
    spine = _soft_ellipse(yy, xx, sp_cy, 0.0, rng.uniform(0.04, 0.06), rng.uniform(0.03, 0.05), sigma=1.5)
    x = np.where(spine > 0.45, MU_BONE + rng.uniform(-0.03, 0.06), x)

    am = _soft_ellipse(yy, xx, rng.uniform(0.20, 0.38), 0.0,
                       rng.uniform(0.04, 0.08), rng.uniform(0.04, 0.08), sigma=2.0)
    x = np.where(am > 0.45, MU_AIR + rng.uniform(0.01, 0.03), x)

    tx = _fbm(shape, rng, octaves=4, base_sigma=10.0)
    x = np.clip(x + 0.012 * (tx - 0.5) * (head > 0.3).astype(np.float32), 0.0, 1.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


_SCENE_BUILDERS = [_dental_panoramic, _head_axial, _dental_mixed, _head_lower]
_SCENE_NAMES = ["dental_panoramic", "head_axial", "dental_mixed", "head_lower"]


# -- Adversarial modifications ------------------------------------------------

def _dental_with_metal(rng, shape):
    x = _dental_panoramic(rng, shape)
    yy, xx = _grid(shape)
    for _ in range(rng.integers(1, 5)):
        mm = _soft_ellipse(yy, xx, rng.uniform(-0.15, 0.25), rng.uniform(-0.40, 0.40),
                           rng.uniform(0.015, 0.035), rng.uniform(0.015, 0.035),
                           angle_deg=rng.uniform(0, 180), sigma=0.8)
        x = np.where(mm > 0.4, MU_METAL * rng.uniform(0.85, 1.0), x)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _head_with_implant(rng, shape):
    x = _head_axial(rng, shape)
    yy, xx = _grid(shape)
    for _ in range(rng.integers(1, 4)):
        ang = rng.uniform(0, 360)
        dist = rng.uniform(0.50, 0.68)
        im = _soft_ellipse(yy, xx, dist * np.sin(np.deg2rad(ang)), dist * np.cos(np.deg2rad(ang)),
                           rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.04),
                           angle_deg=rng.uniform(0, 180), sigma=0.7)
        x = np.where(im > 0.4, MU_METAL * rng.uniform(0.80, 1.0), x)
    for _ in range(rng.integers(3, 10)):
        sigma_c = rng.uniform(0.003, 0.010)
        r2 = (yy - rng.uniform(-0.40, 0.40))**2 + (xx - rng.uniform(-0.40, 0.40))**2
        x = np.clip(x + rng.uniform(0.30, 0.60) * np.exp(-r2 / (2 * sigma_c**2)), 0.0, 1.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _apply_low_contrast_lesion(x, rng):
    yy, xx = _grid(x.shape)
    for _ in range(rng.integers(3, 8)):
        lr = rng.uniform(0.02, 0.05)
        lm = _soft_ellipse(yy, xx, rng.uniform(-0.40, 0.40), rng.uniform(-0.40, 0.40),
                           lr, lr * rng.uniform(0.8, 1.3), sigma=2.0)
        x = np.clip(x + rng.choice([-1, +1]) * rng.uniform(0.02, 0.05) * (lm > 0.45), 0.0, 1.0)
    return x


def _apply_calcification(x, rng):
    yy, xx = _grid(x.shape)
    for _ in range(rng.integers(5, 15)):
        sigma_c = rng.uniform(0.003, 0.008)
        r2 = (yy - rng.uniform(-0.45, 0.45))**2 + (xx - rng.uniform(-0.45, 0.45))**2
        x = np.clip(x + rng.uniform(0.30, 0.65) * np.exp(-r2 / (2 * sigma_c**2)), 0.0, 1.0)
    return x


_ADVERSARIAL_FNS = [
    (0.30, lambda x, rng: _dental_with_metal(rng, x.shape)),
    (0.25, lambda x, rng: _head_with_implant(rng, x.shape)),
    (0.25, _apply_low_contrast_lesion),
    (0.20, _apply_calcification),
]


def generate_cbct_phantom(seed, mode="public", shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Generate a 2D dental/head CBCT phantom."""
    rng = np.random.default_rng(seed)
    scene_idx = seed % len(_SCENE_BUILDERS)
    x = _SCENE_BUILDERS[scene_idx](rng, shape)
    scene_name = _SCENE_NAMES[scene_idx]

    if mode == "hidden":
        probs = [p for p, _ in _ADVERSARIAL_FNS]
        adv_fn = _ADVERSARIAL_FNS[rng.choice(len(_ADVERSARIAL_FNS), p=probs)][1]
        x = adv_fn(x.copy(), rng)
        scene_name = f"{scene_name}_adversarial"

    return np.clip(x, 0.0, 1.0).astype(np.float32), scene_name


def _augment_diversity(x, rng, mode="dev"):
    from scipy.ndimage import rotate as nd_rotate
    angle = float(rng.uniform(15.0, 345.0)) if mode == "dev" else float(rng.uniform(10.0, 350.0))
    x = nd_rotate(x, angle, reshape=False, mode="constant", cval=0.0)
    if rng.random() < 0.85:
        x = np.fliplr(x)
    if rng.random() < 0.70:
        x = np.flipud(x)
    lo, hi = (0.60, 1.40) if mode == "dev" else (0.55, 1.45)
    zoom_f = float(rng.uniform(lo, hi))
    x_z = zoom(x, zoom_f, order=1)
    sz = IMAGE_SIZE
    if zoom_f >= 1.0:
        zh, zw = x_z.shape
        y0, x0 = (zh - sz) // 2, (zw - sz) // 2
        x = np.ascontiguousarray(x_z[y0:y0 + sz, x0:x0 + sz])
    else:
        zh, zw = x_z.shape
        pad = np.zeros((sz, sz), dtype=np.float32)
        pad[(sz - zh) // 2:(sz - zh) // 2 + zh, (sz - zw) // 2:(sz - zw) // 2 + zw] = x_z
        x = pad
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Image helpers
# =============================================================================


def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr, path):
    Image.fromarray(np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L").save(str(path))


def _save_overview(x_true, sino_ideal, sino_meas, path):
    th, tw = 128, 128
    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0
    ov = np.zeros((th, 3 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(sino_ideal)
    ov[:, 2*tw:3*tw] = _r(sino_meas)
    _save_png(ov, path)


# =============================================================================
# Dataset tier generator
# =============================================================================


def sample_mismatch(rng, spec):
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier, phantoms, base_seed, n_views_range, source_label="synthetic"):
    """Generate one tier matching CT benchmark structure exactly."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"cbct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    rows, true_specs = [], {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = f"PWM CBCT benchmark -- {tier} tier (cone-beam, dental/head phantoms)"
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE, "n_views_default": N_VIEWS,
            "I0": I0, "sigma_readout": SIGMA_RO, "mu_scale": MU_SCALE,
        })
        f.attrs["source"] = source_label

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            n_views = int(rng.integers(n_views_range[0], n_views_range[1] + 1))
            # Use degrees for skimage radon
            angles_deg = np.linspace(0, 180, n_views, endpoint=False).astype(np.float64)
            angles_rad = np.deg2rad(angles_deg).astype(np.float32)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_views": n_views}

            # Ideal sinogram
            sino_ideal = radon_project(x_true, angles_deg) * MU_SCALE

            # Measured sinogram (mismatch + noise)
            sino_meas = apply_mismatch(
                x_true, angles_deg,
                scatter_fraction=mis["scatter_fraction"],
                truncation_fov_factor=mis["truncation_fov_factor"],
                ring_artifact_amplitude=mis["ring_artifact_amplitude"],
                rotation_offset_deg=mis["rotation_offset_deg"],
                rng=rng,
            )

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("sinogram_ideal", data=sino_ideal, compression="gzip")
            grp.create_dataset("sinogram_measured", data=sino_meas, compression="gzip")
            grp.create_dataset("angles_nominal", data=angles_rad)
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name, "shape": list(x_true.shape),
                "n_views": n_views, "source": source_label,
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps({**mis, "n_views": n_views})

            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(sino_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(sino_meas, sample_dir / "sinogram_measured.png")
            _save_overview(x_true, sino_ideal, sino_meas, sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "n_views": n_views}, sf, indent=2)

            rows.append((key, scene_name, x_true.shape, n_views, mis))
            print(f"  [{tier}] {key} {scene_name}  views={n_views}  "
                  f"scatter={mis['scatter_fraction']:.3f} "
                  f"trunc={mis['truncation_fov_factor']:.3f} "
                  f"ring={mis['ring_artifact_amplitude']:.4f} "
                  f"rot_off={mis['rotation_offset_deg']:.3f}")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows)
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


def _write_tier_readme(tier, tier_dir, rows):
    spec = SPEC[tier]
    param_desc = {
        "scatter_fraction": "Scatter-to-primary ratio",
        "truncation_fov_factor": "FOV truncation factor",
        "ring_artifact_amplitude": "Ring artifact amplitude",
        "rotation_offset_deg": "Rotation centre offset",
    }
    if tier == "public":
        source = "Synthetic dental/head CBCT phantoms (12 samples)"
        access = "Full (GT + true spec + ideal sinogram)"
    elif tier == "dev":
        source = "Synthetic dental/head CBCT phantoms with augmentation (20 samples)"
        access = "Blind (measured sinogram + spec ranges only)"
    else:
        source = "Adversarial dental/head CBCT phantoms (20 samples)"
        access = "Server-only"

    lines = [f"# CBCT {tier.capitalize()} Tier\n\n",
             f"**Source:** {source}\n\n**Access:** {access}\n\n",
             "## Mismatch Parameters\n\n| Parameter | Description | Range |\n|-----------|-------------|-------|\n"]
    for k, v in spec.items():
        lines.append(f"| `{k}` | {param_desc[k]} | [{v['min']}, {v['max']}] {v.get('unit', '')} |\n")
    lines += ["\n## Samples\n\n| Sample | Scene | Views | Scatter | Truncation | Ring | Rot Offset |\n",
              "|--------|-------|-------|---------|------------|------|------------|\n"]
    for key, scene, shape, n_views, mis in rows:
        lines.append(f"| {key} | {scene} | {n_views} | {mis['scatter_fraction']:.3f}"
                     f" | {mis['truncation_fov_factor']:.3f} | {mis['ring_artifact_amplitude']:.4f}"
                     f" | {mis['rotation_offset_deg']:.3f} |\n")
    lines += [f"\n## HDF5 Keys\n\n| Key | Shape | Description |\n|-----|-------|-------------|\n",
              f"| `x_true` | ({IMAGE_SIZE}, {IMAGE_SIZE}) | Ground-truth attenuation |\n",
              "| `sinogram_ideal` | (n_views, n_det) | Ideal sinogram (nepers) |\n",
              "| `sinogram_measured` | (n_views, n_det) | Measured sinogram (mismatch + noise) |\n",
              "| `angles_nominal` | (n_views,) | Projection angles [rad] |\n"]
    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme():
    txt = """# CBCT -- Cone-Beam Computed Tomography

## Overview

Cone-Beam CT benchmark with dental/head phantoms and CBCT-specific mismatch.

## Forward Model

Radon projection (parallel-beam approximation of cone-beam in 2D) + Beer-Lambert
noise model: Poisson(I0=5000) + readout N(0, 3.0^2).

## Mismatch Parameters

| Parameter | Description |
|-----------|-------------|
| `scatter_fraction` | Scatter-to-primary ratio (0.2-0.6) |
| `truncation_fov_factor` | FOV truncation (0.7-1.0) |
| `ring_artifact_amplitude` | Detector non-uniformity (0-0.05) |
| `rotation_offset_deg` | Rotation centre misalignment (0-3 deg) |

## Phantom Types

| Type | Anatomy |
|------|---------|
| dental_panoramic | Mandible, teeth, tongue, airway |
| head_axial | Skull, brain, ventricles, sinuses |
| dental_mixed | Maxilla + mandible, teeth, sinuses |
| head_lower | Skull base, petrous bones, TMJ |

## Tiers

| Tier | Samples | Views | Mismatch |
|------|---------|-------|----------|
| Public | 12 | 180 | Mild |
| Dev | 20 | 180 | Medium |
| Hidden | 20 | 120-240 | Severe + adversarial |

## References

- Feldkamp, Davis & Kress (1984) JOSA A 1:612-619.
- PWM Benchmark: https://pwm.platformai.org/benchmark/cbct
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


def main():
    import time
    print("CBCT Benchmark Dataset Generator (dental/head, skimage radon)")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}\n")
    t0 = time.time()

    shape = (IMAGE_SIZE, IMAGE_SIZE)

    print("Generating public tier (12 samples, 180 views)...")
    public_phantoms = []
    for i in range(12):
        x, scene = generate_cbct_phantom(seed=2000 + i, mode="public", shape=shape)
        public_phantoms.append((f"cbct_pub_{i:02d}_{scene}", x))
    generate_tier("public", public_phantoms, base_seed=2000, n_views_range=(N_VIEWS, N_VIEWS),
                  source_label="synthetic_dental_head_cbct")

    print("\nGenerating dev tier (20 samples, 180 views, diversity aug)...")
    dev_phantoms = []
    rng_dev = np.random.default_rng(6666)
    for i in range(20):
        x, scene = generate_cbct_phantom(seed=8000 + i, mode="dev", shape=shape)
        x_aug = _augment_diversity(x, rng_dev, mode="dev")
        dev_phantoms.append((f"cbct_dev_{i:02d}_{scene}", x_aug))
    generate_tier("dev", dev_phantoms, base_seed=8000, n_views_range=(N_VIEWS, N_VIEWS),
                  source_label="synthetic_dental_head_cbct_augmented")

    print("\nGenerating hidden tier (20 samples, 120-240 views, adversarial)...")
    hidden_phantoms = []
    rng_hid = np.random.default_rng(9999)
    for i in range(20):
        x, scene = generate_cbct_phantom(seed=9000 + i, mode="hidden", shape=shape)
        x_aug = _augment_diversity(x, rng_hid, mode="hidden")
        hidden_phantoms.append((f"cbct_hid_{i:02d}_{scene}", x_aug))
    generate_tier("hidden", hidden_phantoms, base_seed=9000, n_views_range=(120, 240),
                  source_label="synthetic_dental_head_cbct_adversarial")

    _write_top_readme()
    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f"Done -- CBCT benchmark ready at {BENCHMARK_DIR}")
    print(f"Total time: {elapsed:.0f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
