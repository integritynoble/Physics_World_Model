"""Procedural CT phantom generator — fan-beam sparse-view benchmark.

Generates 2D cross-sections (attenuation maps) purely procedurally.
No external dataset is required; every sample is fully synthetic.

Five anatomical background types, each with a distinct tissue layout:

  chest_upper  — carina level: large bilateral lungs, trachea, upper mediastinum
  chest_mid    — heart level:  cardiac shadow, full lungs, mediastinum
  chest_lower  — diaphragm:   small lung bases, liver onset, stomach
  abdomen_upper— liver level: liver, spleen, stomach, kidneys, no lungs
  abdomen_mid  — kidney level: bowel loops, psoas, retroperitoneal fat

Dev tier uses these 5 types in round-robin (4 instances each = 20 samples).
Hidden tier applies adversarial modifications (metal, low-contrast lesions,
calcifications, high-contrast bone) on top of the same background diversity.

Value scale matches LoDoPaB-CT normalisation (x = (HU + 1000) / 4071):
    0.00  →  air / vacuum   (−1000 HU)
    0.05  →  lung parenchyma (−800 HU)
    0.22  →  subcutaneous fat (−100 HU)
    0.25  →  soft tissue / water  (  0 HU)
    0.26  →  blood              ( +40 HU)
    0.30  →  dense soft tissue  (+200 HU)
    0.42  →  cortical bone      (+700 HU)
    0.55  →  dense bone cap for dev (≈ 1240 HU)
    0.85  →  metal/implant cap for hidden

Usage:
    from simulate_scenes import generate_ct_gt
    x, scene_type = generate_ct_gt(seed=7000, mode="dev", shape=(362, 362))
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# ── Physical value constants (LoDoPaB-CT normalisation) ─────────────────────
MU_AIR        = 0.00   # vacuum / air  (−1000 HU)
MU_LUNG       = 0.07   # lung parenchyma (−700 HU)
MU_FAT        = 0.22   # subcutaneous fat (−100 HU)
MU_SOFT       = 0.25   # muscle / water  (0 HU)
MU_BLOOD      = 0.26   # blood / heart   (+40 HU)
MU_DENSE_SOFT = 0.30   # dense soft tissue (+200 HU)
MU_BONE_CANC  = 0.37   # cancellous bone (+500 HU)
MU_BONE       = 0.42   # cortical bone   (+700 HU)
MU_BONE_DENSE = 0.52   # dense cortical  (+1200 HU)


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (YY, XX) coordinate grids normalised to [−1, +1]."""
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    return np.meshgrid(yy, xx, indexing="ij")


def _ellipse(
    YY: np.ndarray, XX: np.ndarray,
    cy: float, cx: float, ry: float, rx: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Hard binary ellipse mask in normalised coordinates."""
    t = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    dy, dx = YY - cy, XX - cx
    yr =  cos_t * dy + sin_t * dx
    xr = -sin_t * dy + cos_t * dx
    return ((yr / ry) ** 2 + (xr / rx) ** 2 <= 1.0).astype(np.float32)


def _soft_ellipse(
    YY: np.ndarray, XX: np.ndarray,
    cy: float, cx: float, ry: float, rx: float,
    angle_deg: float = 0.0, sigma: float = 2.0,
) -> np.ndarray:
    """Smoothed-edge filled ellipse (Gaussian-blurred binary mask)."""
    return gaussian_filter(_ellipse(YY, XX, cy, cx, ry, rx, angle_deg), sigma=sigma)


def _ring(
    YY: np.ndarray, XX: np.ndarray,
    cy: float, cx: float, r0: float, thickness: float,
) -> np.ndarray:
    """Thin Gaussian ring (cortical bone shell) in normalised coordinates."""
    r = np.sqrt((YY - cy) ** 2 + (XX - cx) ** 2)
    return np.exp(-((r - r0) ** 2) / (2 * thickness ** 2)).astype(np.float32)


def _fbm(shape: tuple[int, int], rng: np.random.Generator,
         octaves: int = 4, persistence: float = 0.55,
         base_sigma: float = 6.0) -> np.ndarray:
    """Fractal Brownian motion texture, result in [0, 1]."""
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


# ── Body boundary ──────────────────────────────────────────────────────────────

def _body(
    YY: np.ndarray, XX: np.ndarray, rng: np.random.Generator,
    ry: float, rx: float, cy: float = 0.0,
    fat_thick: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body_mask, fat_mask, core_mask) for a patient outline.

    body_mask: outermost body boundary (1 inside, 0 outside)
    fat_mask:  subcutaneous fat ring between body and core
    core_mask: soft-tissue core inside fat
    """
    body = _soft_ellipse(YY, XX, cy, 0.0, ry, rx, sigma=3.0)
    core = _soft_ellipse(YY, XX, cy, 0.0, ry - fat_thick, rx - fat_thick, sigma=2.5)
    fat = np.clip(body - core, 0.0, 1.0)
    return body, fat, core


# ─────────────────────────────────────────────────────────────────────────────
# 5 background builders — each creates a fundamentally different scene layout
# ─────────────────────────────────────────────────────────────────────────────

def _chest_upper(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Carina level: large bilateral lungs, trachea, upper mediastinum.

    Distinctive features vs. other levels:
    - Trachea: central oval air column
    - Very large lung fields (lungs dominate the cross-section)
    - Small mediastinum (vessels only, no heart)
    - Thin thoracic spine (upper thoracic)
    """
    YY, XX = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    # Body outline (narrower thorax, slightly elliptical)
    ry = rng.uniform(0.70, 0.82)
    rx = rng.uniform(0.82, 0.94)
    fat_th = rng.uniform(0.04, 0.10)
    body, fat, core = _body(YY, XX, rng, ry, rx, cy=rng.uniform(-0.04, 0.04), fat_thick=fat_th)
    x += MU_SOFT  * (body  > 0.3)
    x += (MU_FAT - MU_SOFT) * (fat > 0.3)

    # Texture on soft tissue
    tx = _fbm(shape, rng, octaves=4, base_sigma=10.0)
    body_m = (body > 0.3).astype(np.float32)
    x = np.clip(x + 0.015 * (tx - 0.5) * body_m, 0.0, 1.0)

    # Two large lung cavities
    lung_r = rng.uniform(0.24, 0.32)
    lung_cx = rng.uniform(0.16, 0.26)
    for sign in [+1, -1]:
        lry = lung_r + rng.uniform(-0.03, 0.04)
        lrx = lung_r * rng.uniform(0.90, 1.10)
        lcx = sign * (lung_cx + rng.uniform(-0.02, 0.03))
        lcy = rng.uniform(-0.10, 0.10)
        lm = _soft_ellipse(YY, XX, lcy, lcx, lry, lrx,
                            angle_deg=rng.uniform(-10, 10), sigma=4.0)
        lung_val = MU_LUNG + rng.uniform(-0.02, 0.02)
        x = np.where(lm > 0.45, lung_val, x)

    # Trachea (central midline, oval air column)
    trachea_w = rng.uniform(0.025, 0.040)
    trachea_h = rng.uniform(0.035, 0.055)
    trachea_cy = rng.uniform(-0.05, 0.10)
    tm = _soft_ellipse(YY, XX, trachea_cy, rng.uniform(-0.02, 0.02),
                        trachea_h, trachea_w, sigma=1.5)
    x = np.where(tm > 0.5, MU_AIR + rng.uniform(0.01, 0.03), x)

    # Upper mediastinal vessels (aortic arch, SVC — two small circles)
    for sign, size_frac in [(+1, 1.0), (-1, 0.7)]:
        vcy = rng.uniform(-0.05, 0.10)
        vcx = sign * rng.uniform(0.02, 0.06)
        vr = rng.uniform(0.025, 0.040) * size_frac
        vm = _soft_ellipse(YY, XX, vcy, vcx, vr, vr * rng.uniform(0.9, 1.1), sigma=1.5)
        x = np.where(vm > 0.5, MU_BLOOD + rng.uniform(-0.01, 0.02), x)

    # Upper thoracic spine (smaller, more circular than lower)
    s_ry = rng.uniform(0.040, 0.060)
    s_rx = rng.uniform(0.030, 0.048)
    s_cy = rng.uniform(0.52, 0.68)
    sm = _soft_ellipse(YY, XX, s_cy, rng.uniform(-0.02, 0.02), s_ry, s_rx, sigma=1.5)
    x = np.where(sm > 0.5, MU_BONE + rng.uniform(-0.04, 0.06), x)
    # Spinal cord
    cord_m = _soft_ellipse(YY, XX, s_cy, 0.0, s_ry * 0.40, s_rx * 0.40, sigma=1.0)
    x = np.where(cord_m > 0.6, MU_SOFT, x)

    # 2-4 rib pairs (upper ribs, more anteriorly placed)
    n_ribs = rng.integers(2, 5)
    for i in range(n_ribs):
        level = -0.30 + i * 0.18 + rng.uniform(-0.04, 0.04)
        r0 = rng.uniform(0.22, 0.34)
        thick = rng.uniform(0.010, 0.018)
        for sign in [+1, -1]:
            ring = _ring(YY, XX, level + rng.uniform(-0.02, 0.02),
                          sign * rng.uniform(0.02, 0.08), r0, thick)
            half = (np.sign(XX) == sign).astype(np.float32)
            x = np.clip(x + MU_BONE * ring * half * body_m, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _chest_mid(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Heart level: cardiac shadow dominant, full bilateral lungs, ribs.

    Distinctive features:
    - Large central cardiac shadow (occupies mediastinum)
    - Descending aorta (small circle left of spine)
    - Full lung fields but smaller than upper chest
    - 4-6 rib pairs
    - Mid-thoracic spine (larger cross-section than upper)
    """
    YY, XX = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    # Body outline (typical mid-chest, slightly wider)
    ry = rng.uniform(0.72, 0.85)
    rx = rng.uniform(0.84, 0.96)
    fat_th = rng.uniform(0.05, 0.12)
    body, fat, core = _body(YY, XX, rng, ry, rx, cy=rng.uniform(-0.04, 0.04), fat_thick=fat_th)
    x += MU_SOFT * (body > 0.3)
    x += (MU_FAT - MU_SOFT) * (fat > 0.3)
    body_m = (body > 0.3).astype(np.float32)
    tx = _fbm(shape, rng, octaves=4, base_sigma=12.0)
    x = np.clip(x + 0.015 * (tx - 0.5) * body_m, 0.0, 1.0)

    # Bilateral lungs (mid-chest — somewhat smaller than upper to make room for heart)
    lung_r = rng.uniform(0.20, 0.27)
    lung_cx = rng.uniform(0.18, 0.28)
    asym = rng.uniform(-0.04, 0.04)
    for sign in [+1, -1]:
        lry = lung_r + rng.uniform(-0.02, 0.03) + (asym if sign == +1 else -asym)
        lrx = lry * rng.uniform(0.85, 1.15)
        lcx = sign * (lung_cx + rng.uniform(-0.02, 0.03))
        lcy = rng.uniform(-0.08, 0.08)
        lm = _soft_ellipse(YY, XX, lcy, lcx, lry, lrx,
                            angle_deg=rng.uniform(-15, 15), sigma=4.0)
        x = np.where(lm > 0.45, MU_LUNG + rng.uniform(-0.02, 0.02), x)

    # Heart (large mediastinal shadow — the defining feature of this level)
    h_ry = rng.uniform(0.16, 0.24)
    h_rx = rng.uniform(0.14, 0.22)
    h_cy = rng.uniform(-0.10, 0.08)
    h_cx = rng.uniform(-0.08, 0.06)
    h_angle = rng.uniform(-20, 20)
    hm = _soft_ellipse(YY, XX, h_cy, h_cx, h_ry, h_rx, angle_deg=h_angle, sigma=4.0)
    heart_val = MU_BLOOD + rng.uniform(-0.01, 0.02)
    x = np.where(hm > 0.45, heart_val, x)

    # Pericardium (thin bright ring around heart)
    peri_r = max(h_ry, h_rx) + rng.uniform(0.005, 0.012)
    peri_ring = _ring(YY, XX, h_cy, h_cx, peri_r, rng.uniform(0.006, 0.012))
    x = np.clip(x + MU_DENSE_SOFT * peri_ring * 0.6, 0.0, 1.0)

    # Descending aorta (small circle, left posterior)
    ao_cy = rng.uniform(0.20, 0.45)
    ao_cx = rng.uniform(-0.12, -0.04)
    ao_r = rng.uniform(0.018, 0.030)
    am = _soft_ellipse(YY, XX, ao_cy, ao_cx, ao_r, ao_r, sigma=1.5)
    x = np.where(am > 0.5, MU_BLOOD, x)

    # Mid-thoracic spine (larger, more square cross-section)
    s_ry = rng.uniform(0.048, 0.072)
    s_rx = rng.uniform(0.038, 0.060)
    s_cy = rng.uniform(0.50, 0.66)
    sm = _soft_ellipse(YY, XX, s_cy, rng.uniform(-0.02, 0.02), s_ry, s_rx, sigma=1.5)
    x = np.where(sm > 0.5, MU_BONE + rng.uniform(-0.04, 0.07), x)
    cord_m = _soft_ellipse(YY, XX, s_cy, 0.0, s_ry * 0.38, s_rx * 0.38, sigma=1.0)
    x = np.where(cord_m > 0.6, MU_SOFT, x)

    # 4-6 rib pairs
    n_ribs = rng.integers(4, 7)
    for i in range(n_ribs):
        level = -0.40 + i * 0.16 + rng.uniform(-0.03, 0.03)
        r0 = rng.uniform(0.22, 0.36)
        thick = rng.uniform(0.010, 0.020)
        for sign in [+1, -1]:
            ring = _ring(YY, XX, level + rng.uniform(-0.02, 0.02),
                          sign * rng.uniform(0.01, 0.07), r0, thick)
            half = (np.sign(XX) == sign).astype(np.float32)
            x = np.clip(x + MU_BONE * ring * half * body_m, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _chest_lower(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Lower chest / diaphragm level: small lung bases, liver onset, stomach.

    Distinctive features:
    - Very small lung 'bases' (just the lower tips of lungs, posterior)
    - Liver parenchyma begins on right side
    - Stomach on left (may be gas-filled)
    - Heart partially visible (lower cardiac border)
    - Diaphragm visible as curved density
    - Lower thoracic/upper lumbar spine (largest spine cross-section)
    - 6-8 rib pairs
    """
    YY, XX = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    # Body (bigger than pure chest — upper abdomen beginning)
    ry = rng.uniform(0.74, 0.88)
    rx = rng.uniform(0.86, 0.98)
    fat_th = rng.uniform(0.06, 0.14)
    body, fat, core = _body(YY, XX, rng, ry, rx, cy=rng.uniform(-0.04, 0.04), fat_thick=fat_th)
    x += MU_SOFT * (body > 0.3)
    x += (MU_FAT - MU_SOFT) * (fat > 0.3)
    body_m = (body > 0.3).astype(np.float32)
    tx = _fbm(shape, rng, octaves=4, base_sigma=14.0)
    x = np.clip(x + 0.018 * (tx - 0.5) * body_m, 0.0, 1.0)

    # Small lung bases (posterior, bilateral) — just the tips
    for sign in [+1, -1]:
        lcy = rng.uniform(0.10, 0.30)   # posterior
        lcx = sign * rng.uniform(0.14, 0.26)
        lry = rng.uniform(0.10, 0.18)
        lrx = lry * rng.uniform(0.85, 1.20)
        lm = _soft_ellipse(YY, XX, lcy, lcx, lry, lrx,
                            angle_deg=rng.uniform(-20, 20), sigma=3.0)
        x = np.where(lm > 0.45, MU_LUNG + rng.uniform(-0.02, 0.02), x)

    # Lower heart border (partial, anterior)
    hry = rng.uniform(0.10, 0.16)
    hrx = rng.uniform(0.10, 0.16)
    hcy = rng.uniform(-0.25, -0.12)
    hcx = rng.uniform(-0.08, 0.06)
    hm = _soft_ellipse(YY, XX, hcy, hcx, hry, hrx, sigma=3.5)
    x = np.where(hm > 0.45, MU_BLOOD + rng.uniform(-0.01, 0.02), x)

    # Liver (right side, large homogeneous parenchyma)
    liver_ry = rng.uniform(0.18, 0.28)
    liver_rx = rng.uniform(0.22, 0.34)
    liver_cy = rng.uniform(-0.10, 0.12)
    liver_cx = rng.uniform(0.18, 0.34)
    liver_angle = rng.uniform(-20, 10)
    lvm = _soft_ellipse(YY, XX, liver_cy, liver_cx, liver_ry, liver_rx,
                         angle_deg=liver_angle, sigma=5.0)
    liver_val = MU_DENSE_SOFT + rng.uniform(-0.02, 0.03)
    x = np.where(lvm > 0.40, liver_val, x)

    # Stomach (left side, can be fluid or gas-filled)
    st_ry = rng.uniform(0.08, 0.15)
    st_rx = rng.uniform(0.06, 0.12)
    st_cy = rng.uniform(-0.12, 0.10)
    st_cx = rng.uniform(-0.28, -0.14)
    stm = _soft_ellipse(YY, XX, st_cy, st_cx, st_ry, st_rx,
                         angle_deg=rng.uniform(-30, 30), sigma=3.0)
    stomach_val = rng.choice([MU_AIR + 0.02, MU_SOFT])  # gas or fluid
    x = np.where(stm > 0.45, stomach_val, x)

    # Lower thoracic spine (largest cross-section)
    s_ry = rng.uniform(0.055, 0.080)
    s_rx = rng.uniform(0.042, 0.065)
    s_cy = rng.uniform(0.44, 0.62)
    sm = _soft_ellipse(YY, XX, s_cy, rng.uniform(-0.02, 0.02), s_ry, s_rx, sigma=2.0)
    x = np.where(sm > 0.5, MU_BONE + rng.uniform(-0.03, 0.08), x)
    cord_m = _soft_ellipse(YY, XX, s_cy, 0.0, s_ry * 0.35, s_rx * 0.35, sigma=1.0)
    x = np.where(cord_m > 0.6, MU_SOFT, x)

    # 6-8 rib pairs
    n_ribs = rng.integers(6, 9)
    for i in range(n_ribs):
        level = -0.48 + i * 0.14 + rng.uniform(-0.03, 0.03)
        r0 = rng.uniform(0.24, 0.38)
        thick = rng.uniform(0.011, 0.022)
        for sign in [+1, -1]:
            ring = _ring(YY, XX, level + rng.uniform(-0.02, 0.02),
                          sign * rng.uniform(0.01, 0.06), r0, thick)
            half = (np.sign(XX) == sign).astype(np.float32)
            x = np.clip(x + MU_BONE * ring * half * body_m, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _abdomen_upper(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Liver / upper abdomen level: liver, spleen, stomach, kidneys — no lungs.

    Distinctive features:
    - NO lung cavities at all
    - Liver dominates the right side (largest solid organ)
    - Spleen on posterior-left
    - Stomach (variable: gas-filled or fluid)
    - Bilateral kidneys (posterior, bean-shaped)
    - Aorta + IVC (small midline circles)
    - Prominent retroperitoneal fat
    - Visceral fat between organs
    - Lumbar spine (large, square cross-section)
    """
    YY, XX = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    # Body (wider, rounder than thorax)
    ry = rng.uniform(0.76, 0.90)
    rx = rng.uniform(0.86, 0.98)
    fat_th = rng.uniform(0.08, 0.18)
    body, fat, core = _body(YY, XX, rng, ry, rx, cy=rng.uniform(-0.04, 0.04), fat_thick=fat_th)
    x += MU_SOFT * (body > 0.3)
    x += (MU_FAT - MU_SOFT) * (fat > 0.3)
    body_m = (body > 0.3).astype(np.float32)
    tx = _fbm(shape, rng, octaves=5, base_sigma=14.0)
    x = np.clip(x + 0.018 * (tx - 0.5) * body_m, 0.0, 1.0)

    # Visceral fat layer (interior fat texture — retroperitoneal / omental)
    vis_fat = _fbm(shape, rng, octaves=3, base_sigma=16.0) * 0.20
    core_m = (core > 0.3).astype(np.float32)
    x = np.clip(x - vis_fat * core_m * rng.uniform(0.0, 0.04), 0.0, 1.0)

    # Liver (right side dominant — fills right 40-60% of abdomen)
    lv_ry = rng.uniform(0.26, 0.38)
    lv_rx = rng.uniform(0.28, 0.42)
    lv_cy = rng.uniform(-0.08, 0.14)
    lv_cx = rng.uniform(0.16, 0.32)
    lv_angle = rng.uniform(-15, 15)
    lvm = _soft_ellipse(YY, XX, lv_cy, lv_cx, lv_ry, lv_rx, angle_deg=lv_angle, sigma=6.0)
    liver_val = MU_DENSE_SOFT + rng.uniform(-0.02, 0.04)
    x = np.where(lvm > 0.38, liver_val, x)

    # Gallbladder (small oval within liver)
    gb_ry = rng.uniform(0.03, 0.06)
    gb_rx = rng.uniform(0.02, 0.04)
    gb_cy = lv_cy + rng.uniform(0.05, 0.14)
    gb_cx = lv_cx + rng.uniform(-0.08, 0.08)
    gbm = _soft_ellipse(YY, XX, gb_cy, gb_cx, gb_ry, gb_rx, sigma=2.0)
    x = np.where(gbm > 0.5, MU_BLOOD + rng.uniform(-0.02, 0.02), x)

    # Spleen (posterior-left)
    sp_ry = rng.uniform(0.08, 0.14)
    sp_rx = rng.uniform(0.10, 0.16)
    sp_cy = rng.uniform(0.10, 0.30)
    sp_cx = rng.uniform(-0.36, -0.22)
    spm = _soft_ellipse(YY, XX, sp_cy, sp_cx, sp_ry, sp_rx,
                         angle_deg=rng.uniform(-20, 20), sigma=3.5)
    spleen_val = MU_DENSE_SOFT + rng.uniform(-0.03, 0.03)
    x = np.where(spm > 0.45, spleen_val, x)

    # Stomach (left side, variable content)
    st_ry = rng.uniform(0.08, 0.14)
    st_rx = rng.uniform(0.06, 0.12)
    st_cy = rng.uniform(-0.15, 0.10)
    st_cx = rng.uniform(-0.28, -0.14)
    stm = _soft_ellipse(YY, XX, st_cy, st_cx, st_ry, st_rx,
                         angle_deg=rng.uniform(-30, 30), sigma=3.0)
    # Stomach content: gas, fluid, or food (random)
    st_val = rng.choice([MU_AIR + 0.02, MU_SOFT + 0.01, MU_DENSE_SOFT - 0.03])
    x = np.where(stm > 0.45, st_val, x)

    # Bilateral kidneys (posterior lateral, bean-shaped)
    for sign in [+1, -1]:
        k_ry = rng.uniform(0.09, 0.13)
        k_rx = rng.uniform(0.05, 0.08)
        k_cy = rng.uniform(0.14, 0.34)
        k_cx = sign * rng.uniform(0.20, 0.34)
        k_angle = sign * rng.uniform(10, 25)
        km = _soft_ellipse(YY, XX, k_cy, k_cx, k_ry, k_rx,
                            angle_deg=k_angle, sigma=2.5)
        kidney_val = MU_DENSE_SOFT + rng.uniform(-0.02, 0.03)
        x = np.where(km > 0.45, kidney_val, x)
        # Renal pelvis (inner darker region)
        pelvis_m = _soft_ellipse(YY, XX, k_cy, k_cx, k_ry * 0.35, k_rx * 0.35,
                                  angle_deg=k_angle, sigma=1.5)
        x = np.where(pelvis_m > 0.5, MU_BLOOD + rng.uniform(-0.01, 0.01), x)

    # Aorta (midline, slightly left)
    ao_r = rng.uniform(0.015, 0.025)
    ao_cy = rng.uniform(0.30, 0.48)
    ao_cx = rng.uniform(-0.05, 0.02)
    am = _soft_ellipse(YY, XX, ao_cy, ao_cx, ao_r, ao_r, sigma=1.5)
    x = np.where(am > 0.5, MU_BLOOD, x)

    # IVC (inferior vena cava, slightly right of aorta, larger)
    ivc_r = rng.uniform(0.018, 0.030)
    ivcm = _soft_ellipse(YY, XX, ao_cy + rng.uniform(-0.02, 0.02),
                          ao_cx + rng.uniform(0.03, 0.06),
                          ivc_r, ivc_r * rng.uniform(0.9, 1.3), sigma=1.5)
    x = np.where(ivcm > 0.5, MU_BLOOD, x)

    # Lumbar spine (large square-ish cross-section — largest of all levels)
    s_ry = rng.uniform(0.062, 0.090)
    s_rx = rng.uniform(0.050, 0.075)
    s_cy = rng.uniform(0.40, 0.58)
    sm = _soft_ellipse(YY, XX, s_cy, rng.uniform(-0.02, 0.02), s_ry, s_rx, sigma=2.0)
    x = np.where(sm > 0.5, MU_BONE + rng.uniform(-0.03, 0.09), x)
    cord_m = _soft_ellipse(YY, XX, s_cy, 0.0, s_ry * 0.32, s_rx * 0.32, sigma=1.0)
    x = np.where(cord_m > 0.6, MU_SOFT, x)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _abdomen_mid(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Mid-abdomen / kidney level: bowel, psoas, retroperitoneal fat — no lungs.

    Distinctive features:
    - NO lungs
    - Prominent bowel loops (multiple small circles, variable density)
    - Large psoas muscles (bilateral thick ovals beside spine)
    - Prominent retroperitoneal fat
    - Kidneys may be partially visible (depends on level)
    - Mid-lumbar spine (prominent)
    - Abdominal wall layers (skin, fat, muscle)
    - Mesentery (fat between bowel)
    """
    YY, XX = _grid(shape)
    x = np.zeros(shape, dtype=np.float32)

    # Body (widest of all levels — upper abdomen girth)
    ry = rng.uniform(0.76, 0.92)
    rx = rng.uniform(0.88, 0.99)
    fat_th = rng.uniform(0.08, 0.22)
    body, fat, core = _body(YY, XX, rng, ry, rx, cy=rng.uniform(-0.04, 0.04), fat_thick=fat_th)
    x += MU_SOFT * (body > 0.3)
    x += (MU_FAT - MU_SOFT) * (fat > 0.3)
    body_m = (body > 0.3).astype(np.float32)

    # Retroperitoneal / mesenteric fat (interior light texture)
    ret_fat = _fbm(shape, rng, octaves=3, base_sigma=18.0)
    core_m = (core > 0.3).astype(np.float32)
    fat_val = MU_FAT + rng.uniform(-0.02, 0.02)
    x = np.where((core_m > 0.5) & (ret_fat > 0.6), fat_val, x)

    # Abdominal wall muscles (bilateral — rectus abdominis, obliques)
    for sign in [+1, -1]:
        m_ry = rng.uniform(0.07, 0.12)
        m_rx = rng.uniform(0.05, 0.09)
        m_cx = sign * rng.uniform(0.30, 0.50)
        m_cy = rng.uniform(-0.25, 0.10)
        mm = _soft_ellipse(YY, XX, m_cy, m_cx, m_ry, m_rx,
                            angle_deg=rng.uniform(-20, 20), sigma=2.5)
        x = np.where(mm > 0.5, MU_DENSE_SOFT + rng.uniform(-0.02, 0.02), x)

    # Psoas muscles (bilateral, large ovals beside spine)
    for sign in [+1, -1]:
        p_ry = rng.uniform(0.09, 0.14)
        p_rx = rng.uniform(0.06, 0.10)
        p_cx = sign * rng.uniform(0.08, 0.16)
        p_cy = rng.uniform(0.18, 0.40)
        pm = _soft_ellipse(YY, XX, p_cy, p_cx, p_ry, p_rx, sigma=2.5)
        x = np.where(pm > 0.5, MU_DENSE_SOFT + rng.uniform(-0.02, 0.03), x)

    # Bowel loops (multiple small circles, variable content)
    n_loops = rng.integers(4, 10)
    for _ in range(n_loops):
        bcy = rng.uniform(-0.45, 0.40)
        bcx = rng.uniform(-0.50, 0.50)
        br = rng.uniform(0.02, 0.07)
        bm = _soft_ellipse(YY, XX, bcy, bcx, br, br * rng.uniform(0.8, 1.3), sigma=1.5)
        # Content: air (30%), fluid (50%), soft tissue (20%)
        b_content = rng.choice([MU_AIR + 0.02, MU_SOFT, MU_BLOOD],
                                p=[0.30, 0.50, 0.20])
        x = np.where(bm > 0.5, b_content, x)
        # Bowel wall (bright ring)
        b_wall = _ring(YY, XX, bcy, bcx, br + 0.01, 0.008)
        x = np.clip(x + MU_DENSE_SOFT * b_wall * body_m * 0.4, 0.0, 1.0)

    # Kidneys (may be partial or absent)
    show_kidneys = rng.random() < 0.65
    if show_kidneys:
        for sign in [+1, -1]:
            k_ry = rng.uniform(0.07, 0.11)
            k_rx = rng.uniform(0.04, 0.07)
            k_cy = rng.uniform(0.20, 0.38)
            k_cx = sign * rng.uniform(0.18, 0.32)
            km = _soft_ellipse(YY, XX, k_cy, k_cx, k_ry, k_rx,
                                angle_deg=sign * rng.uniform(10, 25), sigma=2.0)
            x = np.where(km > 0.45, MU_DENSE_SOFT + rng.uniform(-0.02, 0.03), x)

    # Aorta / IVC (midline, small)
    ao_r = rng.uniform(0.012, 0.022)
    ao_cy = rng.uniform(0.30, 0.52)
    ao_cx = rng.uniform(-0.04, 0.02)
    am = _soft_ellipse(YY, XX, ao_cy, ao_cx, ao_r, ao_r, sigma=1.5)
    x = np.where(am > 0.5, MU_BLOOD, x)

    # Mid-lumbar spine (large, square — no spinal cord at L-level, just cauda equina)
    s_ry = rng.uniform(0.065, 0.090)
    s_rx = rng.uniform(0.055, 0.078)
    s_cy = rng.uniform(0.38, 0.58)
    sm = _soft_ellipse(YY, XX, s_cy, rng.uniform(-0.02, 0.02), s_ry, s_rx, sigma=2.0)
    x = np.where(sm > 0.5, MU_BONE + rng.uniform(-0.02, 0.10), x)
    # Spinal canal (small, rounded — cauda equina)
    canal_m = _soft_ellipse(YY, XX, s_cy, 0.0, s_ry * 0.28, s_rx * 0.30, sigma=1.0)
    x = np.where(canal_m > 0.6, MU_FAT, x)

    # Fine texture
    tx = _fbm(shape, rng, octaves=5, base_sigma=10.0)
    x = np.clip(x + 0.012 * (tx - 0.5) * body_m, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ── Adversarial modifications for hidden tier ──────────────────────────────────

def _apply_metal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dense metal inserts (surgical implants, clips, prosthetics)."""
    YY, XX = _grid(x.shape)
    count = rng.integers(1, 4)
    for _ in range(count):
        mcy = rng.uniform(-0.50, 0.50)
        mcx = rng.uniform(-0.50, 0.50)
        mry = rng.uniform(0.008, 0.030)
        mrx = rng.uniform(0.008, 0.030)
        metal_val = rng.uniform(0.75, 1.00)   # very dense
        mm = _soft_ellipse(YY, XX, mcy, mcx, mry, mrx,
                            angle_deg=rng.uniform(0, 180), sigma=0.8)
        x = np.where(mm > 0.5, metal_val, x)
    return x


def _apply_low_contrast_lesion(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Subtle lesions (hepatic cysts, lymph nodes, micro-nodules)."""
    YY, XX = _grid(x.shape)
    count = rng.integers(3, 8)
    for _ in range(count):
        lcy = rng.uniform(-0.45, 0.45)
        lcx = rng.uniform(-0.45, 0.45)
        lry = rng.uniform(0.020, 0.055)
        lrx = rng.uniform(0.020, 0.055)
        delta = rng.uniform(0.020, 0.055)   # subtle contrast
        lm = _soft_ellipse(YY, XX, lcy, lcx, lry, lrx, sigma=2.5)
        # Can be hypo- or hyper-dense relative to background
        sign = rng.choice([-1, +1])
        x = np.clip(x + sign * delta * (lm > 0.45), 0.0, 1.0)
    return x


def _apply_calcification(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Punctate vascular or glandular calcifications."""
    YY, XX = _grid(x.shape)
    count = rng.integers(8, 20)
    for _ in range(count):
        ccy = rng.uniform(-0.50, 0.50)
        ccx = rng.uniform(-0.50, 0.50)
        sigma_c = rng.uniform(0.003, 0.010)
        r2 = (YY - ccy) ** 2 + (XX - ccx) ** 2
        spot = np.exp(-r2 / (2 * sigma_c ** 2)).astype(np.float32)
        x = np.clip(x + rng.uniform(0.40, 0.85) * spot, 0.0, 1.0)
    return x


def _apply_high_contrast(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boost bone contrast, add cortical thickening or fracture line."""
    # Increase existing bright structures (bone)
    high_mask = (x > MU_BONE_CANC).astype(np.float32)
    x = np.clip(x + rng.uniform(0.05, 0.15) * high_mask, 0.0, 1.0)
    # Add one extra dense bone shell
    YY, XX = _grid(x.shape)
    r0 = rng.uniform(0.10, 0.38)
    thick = rng.uniform(0.008, 0.018)
    cy = rng.uniform(-0.30, 0.30)
    cx = rng.uniform(-0.30, 0.30)
    ring = _ring(YY, XX, cy, cx, r0, thick)
    x = np.clip(x + MU_BONE_DENSE * ring * 0.8, 0.0, 1.0)
    return x


_ADVERSARIAL_FNS = [
    (0.35, _apply_metal),
    (0.30, _apply_low_contrast_lesion),
    (0.20, _apply_calcification),
    (0.15, _apply_high_contrast),
]

# ── Scene type dispatcher ──────────────────────────────────────────────────────

_SCENE_BUILDERS = [
    _chest_upper,
    _chest_mid,
    _chest_lower,
    _abdomen_upper,
    _abdomen_mid,
]
_SCENE_NAMES = [
    "chest_upper",
    "chest_mid",
    "chest_lower",
    "abdomen_upper",
    "abdomen_mid",
]
_N_TYPES = len(_SCENE_BUILDERS)


def generate_ct_gt(
    seed: int,
    mode: str = "dev",
    shape: tuple[int, int] = (362, 362),
) -> tuple[np.ndarray, str]:
    """Generate a unique procedural CT phantom per seed.

    Each seed produces a DIFFERENT background type AND a different
    parameterisation within that type.  No external data is used.

    Background types cycle deterministically across seeds so that 20 dev
    samples contain exactly 4 instances of each of the 5 backgrounds:
        chest_upper · chest_mid · chest_lower · abdomen_upper · abdomen_mid

    Value scale matches LoDoPaB-CT normalisation:
        x = (HU + 1000) / 4071  →  x_true ∈ [0, 0.55] (dev) / [0, 0.85] (hidden)

    Returns:
        (x_true, scene_name)  where x_true is float32 in [0, 1].
    """
    if mode not in ("dev", "hidden"):
        raise ValueError(f"mode must be 'dev' or 'hidden', got {mode!r}")

    # Round-robin scene type: each consecutive 5 seeds get one of each type.
    # E.g. seeds 7000–7004 → chest_upper, chest_mid, chest_lower, abdomen_upper, abdomen_mid
    #       seeds 7005–7009 → same rotation, different parameters
    scene_idx = seed % _N_TYPES
    builder   = _SCENE_BUILDERS[scene_idx]
    scene_name = _SCENE_NAMES[scene_idx]

    # Use the full seed for internal RNG → unique parameterisation each time
    rng = np.random.default_rng(seed)
    x = builder(rng, shape)

    if mode == "hidden":
        # Pick one adversarial modification deterministically from seed
        adv_probs = [p for p, _ in _ADVERSARIAL_FNS]
        adv_fn = _ADVERSARIAL_FNS[rng.choice(len(_ADVERSARIAL_FNS), p=adv_probs)][1]
        x = adv_fn(x, rng)
        x_max = 0.85
        scene_name = f"{scene_name}_adversarial"
    else:
        x_max = 0.55   # match LoDoPaB-CT max (≈ 1240 HU dense bone)

    return np.clip(x, 0.0, x_max).astype(np.float32), scene_name
