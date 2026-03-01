"""Procedural knee MRI phantom generator for PWM MRI benchmark.

Produces synthetic 2D T2-weighted TSE knee MRI ground-truth images that
mimic fastMRI multi-coil knee data appearance (320×320, float32 in [0,1]).

T2w TSE signal intensity reference (normalised):
  Joint fluid          ~0.90  (very bright — long T2)
  Subcutaneous fat     ~0.83  (bright — short T1, moderate T2)
  Bone marrow fat      ~0.78
  Articular cartilage  ~0.52  (intermediate — thin layer)
  Muscle               ~0.30  (intermediate-low)
  Fibrocartilage       ~0.08  (dark — short T2, menisci)
  Cortical bone        ~0.04  (dark — very short T2)
  Background / air      0.00

Recipes
-------
Dev (mild):
  knee_coronal_normal   55%  Standard coronal TSE knee, mild effusion
  knee_coronal_effusion 30%  Prominent joint fluid (synovial effusion)
  knee_axial_patella    15%  Axial slice through patello-femoral joint

Hidden (adversarial):
  knee_osteophyte       35%  Bony spurs on condyle margins (stress test)
  knee_multicompartment 35%  Posterior Baker's cyst + extra structure
  knee_high_contrast    20%  Extreme fluid/bone contrast ratio
  knee_thin_cartilage   10%  Very thin/absent cartilage (edge stress)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (320, 320)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_mri_gt(
    seed: int,
    mode: str = "dev",
    shape: tuple = SHAPE,
) -> tuple:
    """Generate a synthetic knee MRI phantom image.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.
    mode : {'dev', 'hidden'}
        Controls which recipe distribution is sampled.
    shape : (H, W)
        Output image size.  Default matches fastMRI knee (320x320).

    Returns
    -------
    x : np.ndarray  float32  shape (H, W)  values in [0, 1]
    recipe_name : str
    """
    rng = np.random.default_rng(seed)

    if mode == "dev":
        recipes = [
            ("knee_coronal_normal",   0.55),
            ("knee_coronal_effusion", 0.30),
            ("knee_axial_patella",    0.15),
        ]
    else:  # hidden — adversarial
        recipes = [
            ("knee_osteophyte",       0.35),
            ("knee_multicompartment", 0.35),
            ("knee_high_contrast",    0.20),
            ("knee_thin_cartilage",   0.10),
        ]

    names, probs = zip(*recipes)
    recipe = str(rng.choice(names, p=list(probs)))
    x = _build_scene(recipe, shape, rng)
    return x.astype(np.float32), recipe


# ── Scene dispatcher ──────────────────────────────────────────────────────────

def _build_scene(recipe, shape, rng):
    if recipe == "knee_coronal_normal":
        return _knee_coronal(shape, rng, effusion=rng.uniform(0.10, 0.30))
    if recipe == "knee_coronal_effusion":
        return _knee_coronal(shape, rng, effusion=rng.uniform(0.55, 0.88))
    if recipe == "knee_axial_patella":
        return _knee_axial_patella(shape, rng)
    if recipe == "knee_osteophyte":
        return _knee_coronal(shape, rng, effusion=rng.uniform(0.20, 0.50),
                             osteophytes=True)
    if recipe == "knee_multicompartment":
        return _knee_multicompartment(shape, rng)
    if recipe == "knee_high_contrast":
        return _knee_coronal(shape, rng, effusion=rng.uniform(0.72, 0.95),
                             high_contrast=True)
    if recipe == "knee_thin_cartilage":
        return _knee_coronal(shape, rng, effusion=rng.uniform(0.10, 0.40),
                             thin_cartilage=True)
    return _knee_coronal(shape, rng)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _ellipse(H, W, cy, cx, ry, rx, angle_deg=0.0):
    """Boolean ellipse mask, shape (H, W)."""
    yy = np.arange(H, dtype=np.float32)[:, None] - cy
    xx = np.arange(W, dtype=np.float32)[None, :] - cx
    if angle_deg:
        ang = np.radians(angle_deg)
        yr = yy * np.cos(ang) + xx * np.sin(ang)
        xr = -yy * np.sin(ang) + xx * np.cos(ang)
    else:
        yr, xr = yy, xx
    return ((yr / ry) ** 2 + (xr / rx) ** 2) <= 1.0


def _soft(mask, sigma=2.5):
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


# ── Coronal knee phantom ──────────────────────────────────────────────────────

def _knee_coronal(shape, rng, *, effusion=0.20, osteophytes=False,
                  high_contrast=False, thin_cartilage=False):
    """Coronal cross-section: femoral condyles + tibial plateau + joint space."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    jy = int(rng.integers(-12, 13))
    jx = int(rng.integers(-10, 11))
    cy, cx = H // 2 + jy, W // 2 + jx

    # ── Limb + muscle background ───────────────────────────────────────────
    limb = _ellipse(H, W, cy, cx, H * 0.44, W * 0.41).astype(np.float32)
    muscle_sig = rng.uniform(0.18, 0.22) if high_contrast else rng.uniform(0.27, 0.38)
    img += limb * muscle_sig

    # Subcutaneous fat ring
    inner_limb = _ellipse(H, W, cy, cx, H * 0.395, W * 0.365).astype(np.float32)
    fat_ring = _soft((limb - inner_limb).clip(0), sigma=2.0)
    img += fat_ring * rng.uniform(0.74, 0.87)

    # ── Femoral condyles ───────────────────────────────────────────────────
    cond_y   = cy - int(H * 0.12)
    cond_sep = int(W * 0.14)
    cond_ry  = H * 0.135
    cond_rx  = W * 0.10

    for side, cx_c in [(-1, cx - cond_sep), (1, cx + cond_sep)]:
        # Marrow
        img += _soft(_ellipse(H, W, cond_y, cx_c, cond_ry * 0.73, cond_rx * 0.73),
                     sigma=2.5) * rng.uniform(0.72, 0.82)

        # Cortical rim
        cort = _soft(
            (_ellipse(H, W, cond_y, cx_c, cond_ry, cond_rx).astype(np.float32)
             - _ellipse(H, W, cond_y, cx_c, cond_ry * 0.83, cond_rx * 0.83).astype(np.float32)
             ).clip(0), sigma=1.5)
        img -= cort * rng.uniform(0.40, 0.56)

        # Articular cartilage (inferior condyle surface)
        ct = 0.055 if thin_cartilage else rng.uniform(0.08, 0.13)
        cart = _soft(_ellipse(H, W, cond_y + int(cond_ry * 0.86), cx_c,
                               cond_ry * ct, cond_rx * 0.86), sigma=1.8)
        img += cart * rng.uniform(0.46, 0.60)

        # Osteophytes (hidden-tier bony spurs)
        if osteophytes:
            for _ in range(rng.integers(1, 4)):
                ang = rng.uniform(-30, 30)
                oy = cond_y + int(cond_ry * rng.uniform(0.72, 1.05))
                ox = cx_c + int(cond_rx * rng.uniform(0.60, 0.95)) * side
                spur = _soft(_ellipse(H, W, oy, ox,
                                      H * rng.uniform(0.014, 0.030),
                                      W * rng.uniform(0.013, 0.026),
                                      angle_deg=ang), sigma=1.2)
                img -= spur * rng.uniform(0.30, 0.50)

    # ── Tibial plateau ─────────────────────────────────────────────────────
    tib_y  = cy + int(H * 0.14)
    tib_ry = H * 0.10
    tib_rx = W * 0.285

    img += _soft(_ellipse(H, W, tib_y, cx, tib_ry * 0.72, tib_rx * 0.88),
                 sigma=2.5) * rng.uniform(0.68, 0.80)

    tib_cort = _soft(
        (_ellipse(H, W, tib_y, cx, tib_ry, tib_rx).astype(np.float32)
         - _ellipse(H, W, tib_y, cx, tib_ry * 0.81, tib_rx * 0.91).astype(np.float32)
         ).clip(0), sigma=1.5)
    img -= tib_cort * rng.uniform(0.35, 0.50)

    tib_cart_th = 0.055 if thin_cartilage else rng.uniform(0.09, 0.14)
    img += _soft(_ellipse(H, W, tib_y - int(tib_ry * 0.86), cx,
                           tib_ry * tib_cart_th, tib_rx * 0.84),
                 sigma=1.8) * rng.uniform(0.44, 0.58)

    # ── Joint space ────────────────────────────────────────────────────────
    jt_y  = cy + int(H * 0.01)
    jt_ry = H * 0.065
    jt_rx = W * 0.28
    joint = _soft(_ellipse(H, W, jt_y, cx, jt_ry, jt_rx), sigma=2.0)
    fluid_sig = 0.91 if high_contrast else rng.uniform(0.82, 0.94)
    img += joint * effusion * fluid_sig

    # ── Menisci ────────────────────────────────────────────────────────────
    for side in [-1, 1]:
        men_cx = cx + side * int(W * 0.10)
        men = _soft(_ellipse(H, W, jt_y, men_cx,
                              jt_ry * 0.58, cond_rx * 0.52), sigma=2.0)
        img -= men * rng.uniform(0.12, 0.22)

    # ── Texture noise ──────────────────────────────────────────────────────
    noise = rng.standard_normal((H, W)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=rng.uniform(1.2, 2.5))
    img += noise * rng.uniform(0.018, 0.032) * limb

    return img.clip(0.0, 1.0)


# ── Axial patella phantom ─────────────────────────────────────────────────────

def _knee_axial_patella(shape, rng):
    """Axial cross-section: patella + trochlear groove + Hoffa fat pad."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    jy = int(rng.integers(-14, 15))
    jx = int(rng.integers(-12, 13))
    cy, cx = H // 2 + jy, W // 2 + jx

    limb = _ellipse(H, W, cy, cx, H * 0.43, W * 0.42).astype(np.float32)
    img += limb * rng.uniform(0.26, 0.35)
    inner = _ellipse(H, W, cy, cx, H * 0.385, W * 0.375).astype(np.float32)
    img += _soft((limb - inner).clip(0), sigma=2.5) * rng.uniform(0.73, 0.86)

    # Femoral trochlea (posterior)
    fem_y  = cy + int(H * 0.05)
    fem_ry = H * 0.17
    fem_rx = W * 0.22
    img += _soft(_ellipse(H, W, fem_y, cx, fem_ry * 0.76, fem_rx * 0.76),
                 sigma=2.5) * rng.uniform(0.72, 0.82)
    img -= _soft(
        (_ellipse(H, W, fem_y, cx, fem_ry, fem_rx).astype(np.float32)
         - _ellipse(H, W, fem_y, cx, fem_ry * 0.82, fem_rx * 0.82).astype(np.float32)
         ).clip(0), sigma=1.5) * rng.uniform(0.38, 0.52)

    # Patella (anterior)
    pat_y  = cy - int(H * 0.17)
    pat_ry = H * 0.08
    pat_rx = W * 0.075
    img += _soft(_ellipse(H, W, pat_y, cx, pat_ry * 0.70, pat_rx * 0.70),
                 sigma=2.0) * rng.uniform(0.68, 0.80)
    img -= _soft(
        (_ellipse(H, W, pat_y, cx, pat_ry, pat_rx).astype(np.float32)
         - _ellipse(H, W, pat_y, cx, pat_ry * 0.80, pat_rx * 0.80).astype(np.float32)
         ).clip(0), sigma=1.5) * rng.uniform(0.35, 0.50)

    # Patellar cartilage (posterior face)
    img += _soft(_ellipse(H, W, pat_y + int(pat_ry * 0.82), cx,
                           pat_ry * 0.17, pat_rx * 0.90), sigma=1.8) \
           * rng.uniform(0.48, 0.60)

    # Hoffa fat pad
    img += _soft(_ellipse(H, W, cy - int(H * 0.04), cx, H * 0.10, W * 0.10),
                 sigma=3.0) * rng.uniform(0.77, 0.87)

    # Patellar joint fluid crescent
    mid_y    = (pat_y + int(pat_ry) + fem_y - int(fem_ry)) // 2
    fluid_ry = max(4, abs(pat_y + int(pat_ry) - fem_y + int(fem_ry)) // 2 + 3)
    img += _soft(_ellipse(H, W, mid_y, cx, fluid_ry, W * 0.13), sigma=2.0) \
           * rng.uniform(0.62, 0.90)

    noise = rng.standard_normal((H, W)).astype(np.float32)
    img += gaussian_filter(noise, sigma=rng.uniform(1.2, 2.5)) \
           * rng.uniform(0.015, 0.030) * limb

    return img.clip(0.0, 1.0)


# ── Multi-compartment phantom (hidden tier) ───────────────────────────────────

def _knee_multicompartment(shape, rng):
    """Coronal + posterior Baker's cyst (popliteal fluid collection)."""
    img = _knee_coronal(shape, rng, effusion=rng.uniform(0.35, 0.72))
    H, W = shape
    cy, cx = H // 2, W // 2

    cyst_y  = cy + int(H * rng.uniform(0.30, 0.40))
    cyst_cx = cx + int(W * rng.uniform(-0.12, 0.12))
    cyst = _soft(_ellipse(H, W, cyst_y, cyst_cx,
                           H * rng.uniform(0.05, 0.10),
                           W * rng.uniform(0.06, 0.13)), sigma=2.5)
    img += cyst * rng.uniform(0.76, 0.92)

    return img.clip(0.0, 1.0)
