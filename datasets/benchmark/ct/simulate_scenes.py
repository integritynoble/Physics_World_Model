"""Procedural CT phantom generator for fan-beam sparse-view benchmark.

Generates 2D cross-sections (attenuation maps) matching LoDoPaB-CT anatomy:
  - Chest / thorax cross-sections (lung parenchyma, ribs, spine, mediastinum)
  - Value scale matches LoDoPaB-CT normalisation:
      x_true in [0, 1]  where
        0.00  ≈ air (HU ≈ −1000)
        0.25  ≈ soft tissue / water (HU ≈ 0)
        0.42  ≈ cortical bone (HU ≈ 700)
        1.00  ≈ maximum density (HU ≈ 3071)
  Formula: x = (HU + 1000) / 4071

Dev tier     — Tissue-like procedural chest phantoms  (20 samples)
Hidden tier  — Adversarial: metal inserts, low-contrast lesions (20 samples)

Usage:
    from simulate_scenes import generate_ct_gt
    x, recipe = generate_ct_gt(seed=42, mode="dev", shape=(362, 362))
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ── Value-range constants (LoDoPaB-CT normalisation) ─────────────────────────
# x = (HU + 1000) / 4071
HU_AIR        = 0.00   # lung parenchyma / air cavities
HU_FAT        = 0.22   # subcutaneous fat  (≈ −100 HU)
HU_SOFT       = 0.25   # muscle / water    (≈ 0 HU)
HU_BLOOD      = 0.26   # blood / heart     (≈ 45 HU)
HU_BONE_SOFT  = 0.32   # cancellous bone   (≈ 300 HU)
HU_BONE_DENSE = 0.55   # cortical bone     (≈ 1240 HU)
HU_MAX        = 1.00   # maximum (3071 HU)


# ── Primitives ────────────────────────────────────────────────────────────────

def fbm_noise(
    shape: tuple[int, int],
    octaves: int = 5,
    persistence: float = 0.55,
    base_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp, total_amp, sigma = 1.0, 0.0, base_sigma
    for _ in range(octaves):
        n = rng.standard_normal((h, w)).astype(np.float32)
        out += amp * gaussian_filter(n, sigma=sigma)
        total_amp += amp
        amp *= persistence
        sigma *= 2.0
    out /= max(total_amp, 1e-6)
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def ellipse_mask(
    shape: tuple[int, int],
    cy: float, cx: float,
    ry: float, rx: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Soft ellipse mask in normalised coords [-1, 1]."""
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    t = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    yr = cos_t * (YY - cy) + sin_t * (XX - cx)
    xr = -sin_t * (YY - cy) + cos_t * (XX - cx)
    return ((yr / ry) ** 2 + (xr / rx) ** 2 <= 1.0).astype(np.float32)


def smooth_ellipse(
    shape: tuple[int, int],
    cy: float, cx: float,
    ry: float, rx: float,
    angle_deg: float = 0.0,
    sigma: float = 1.5,
) -> np.ndarray:
    """Smoothed-edge filled ellipse."""
    m = ellipse_mask(shape, cy, cx, ry, rx, angle_deg)
    return gaussian_filter(m, sigma=sigma)


def add_rib_arc(
    canvas: np.ndarray,
    rng: np.random.Generator,
    side: int,         # +1 right, -1 left
    level: float,      # y offset in normalised coords
    mu: float = HU_BONE_DENSE,
) -> np.ndarray:
    """Add a single curved rib arc on one side."""
    h, w = canvas.shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)

    # Rib is a thin annular arc centred near the spine on opposite side
    r0 = rng.uniform(0.25, 0.42)          # radius of rib arc
    cx_rib = -side * rng.uniform(0.02, 0.10)  # rib centre x (near midline)
    cy_rib = level + rng.uniform(-0.04, 0.04) # rib level y
    thick = rng.uniform(0.010, 0.020)         # rib thickness (normalised)

    r = np.sqrt((YY - cy_rib) ** 2 + (XX - cx_rib) ** 2)
    arc = np.exp(-((r - r0) ** 2) / (2 * thick ** 2)).astype(np.float32)

    # Mask to the correct lateral half and within body
    half_mask = (np.sign(XX - cx_rib) == side).astype(np.float32)
    return np.clip(canvas + mu * arc * half_mask, 0.0, 1.0)


def add_bone_shell(
    canvas: np.ndarray,
    rng: np.random.Generator,
    count: int = 1,
    mu: float = HU_BONE_DENSE,
) -> np.ndarray:
    """Add thin cortical bone shells (rings)."""
    h, w = canvas.shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    out = canvas.copy()
    for _ in range(count):
        cy = rng.uniform(-0.3, 0.3)
        cx = rng.uniform(-0.3, 0.3)
        r0 = rng.uniform(0.08, 0.35)
        t  = rng.uniform(0.008, 0.025)  # thickness in normalised units
        r  = np.sqrt((YY - cy) ** 2 + (XX - cx) ** 2)
        shell = np.exp(-((r - r0) ** 2) / (2 * t ** 2)).astype(np.float32)
        out = np.clip(out + mu * shell, 0.0, 1.0)
    return out


def add_calcifications(
    canvas: np.ndarray,
    rng: np.random.Generator,
    count: int = 6,
) -> np.ndarray:
    """Tiny punctate bright calcification spots."""
    h, w = canvas.shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    out = canvas.copy()
    for _ in range(count):
        cy = rng.uniform(-0.4, 0.4)
        cx = rng.uniform(-0.4, 0.4)
        sigma = rng.uniform(0.004, 0.012)
        r2 = (YY - cy) ** 2 + (XX - cx) ** 2
        spot = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
        out = np.clip(out + rng.uniform(0.5, 1.0) * spot, 0.0, 1.0)
    return out


def add_metal_insert(
    canvas: np.ndarray,
    rng: np.random.Generator,
    count: int = 2,
) -> np.ndarray:
    """Dense metal inserts — beam-hardening stress test."""
    h, w = canvas.shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    out = canvas.copy()
    for _ in range(count):
        cy = rng.uniform(-0.35, 0.35)
        cx = rng.uniform(-0.35, 0.35)
        ry = rng.uniform(0.010, 0.035)
        rx = rng.uniform(0.010, 0.035)
        r2 = ((YY - cy) / ry) ** 2 + ((XX - cx) / rx) ** 2
        metal = (r2 <= 1.0).astype(np.float32)
        out = np.clip(out + 1.0 * metal, 0.0, 1.0)
    return out


# ── Chest CT cross-section builder ────────────────────────────────────────────

def chest_phantom(
    rng: np.random.Generator,
    shape: tuple[int, int],
    n_ribs: int | None = None,
    lung_asymmetry: float = 0.0,   # 0=symmetric, >0 right larger, <0 left larger
    add_spine: bool = True,
    add_mediastinum: bool = True,
    fat_thickness: float | None = None,
) -> np.ndarray:
    """Procedural 2-D thorax cross-section matching LoDoPaB-CT value ranges.

    Layout (posterior = top in array):
        - Oval body boundary filled with soft tissue / fat
        - Two lung cavities (air)
        - Posterior spine (dense bone)
        - Bilateral ribs (curved bone arcs)
        - Mediastinal heart shadow (blood-density ellipse)
    """
    h, w = shape

    # ── 1. Body outline (oval, slight variation) ───────────────────────────
    body_ry = rng.uniform(0.78, 0.90)
    body_rx = rng.uniform(0.86, 0.98)
    body_cy = rng.uniform(-0.06, 0.06)
    body = smooth_ellipse(shape, body_cy, 0.0, body_ry, body_rx, sigma=2.5)
    x = np.zeros(shape, dtype=np.float32)
    # Interior: fat/muscle mix
    fat_val = HU_FAT + rng.uniform(-0.02, 0.02)
    x = np.where(body > 0.3, fat_val, x)

    # ── 2. Sub-fat soft-tissue core ─────────────────────────────────────────
    core_ry = body_ry - 0.08
    core_rx = body_rx - 0.08
    core = smooth_ellipse(shape, body_cy, 0.0, core_ry, core_rx, sigma=2.0)
    soft_val = HU_SOFT + rng.uniform(-0.02, 0.02)
    x = np.where(core > 0.3, soft_val, x)

    # ── 3. Two lung cavities ─────────────────────────────────────────────────
    lung_r_base = rng.uniform(0.20, 0.30)
    lung_cy = rng.uniform(-0.10, 0.15)   # slightly anterior (upper in image)
    lung_cx = rng.uniform(0.18, 0.28)    # lateral offset from midline
    asym = lung_asymmetry * 0.05

    for side, sign in [("right", +1), ("left", -1)]:
        ry = lung_r_base + rng.uniform(-0.03, 0.03) + (asym if sign == +1 else -asym)
        rx = ry * rng.uniform(0.85, 1.15)
        cx = sign * (lung_cx + rng.uniform(-0.02, 0.02))
        cy_l = lung_cy + rng.uniform(-0.03, 0.03)
        lung_mask = smooth_ellipse(shape, cy_l, cx, ry, rx,
                                   angle_deg=rng.uniform(-15, 15), sigma=3.0)
        lung_val = HU_AIR + rng.uniform(0.0, 0.08)   # lung parenchyma (not pure air)
        x = np.where(lung_mask > 0.4, lung_val, x)

    # ── 4. Spine (posterior midline) ─────────────────────────────────────────
    if add_spine:
        spine_ry  = rng.uniform(0.045, 0.075)
        spine_rx  = rng.uniform(0.035, 0.055)
        spine_cy  = rng.uniform(0.50, 0.70)   # posterior
        spine_val = HU_BONE_DENSE + rng.uniform(-0.05, 0.10)
        spine_mask = smooth_ellipse(shape, spine_cy, rng.uniform(-0.02, 0.02),
                                    spine_ry, spine_rx, sigma=1.5)
        x = np.where(spine_mask > 0.4, np.clip(spine_val, 0, 1), x)
        # Spinal cord inside spine
        cord_mask = smooth_ellipse(shape, spine_cy, 0.0,
                                   spine_ry * 0.45, spine_rx * 0.45, sigma=1.0)
        x = np.where(cord_mask > 0.5, HU_SOFT, x)

    # ── 5. Ribs ───────────────────────────────────────────────────────────────
    if n_ribs is None:
        n_ribs = rng.integers(3, 7)
    for i in range(n_ribs):
        level = -0.4 + i * (0.8 / max(n_ribs - 1, 1)) + rng.uniform(-0.05, 0.05)
        for side in [+1, -1]:
            x = add_rib_arc(x, rng, side=side, level=level)

    # ── 6. Mediastinum / heart shadow ─────────────────────────────────────────
    if add_mediastinum:
        med_ry  = rng.uniform(0.12, 0.20)
        med_rx  = rng.uniform(0.10, 0.18)
        med_cy  = rng.uniform(-0.05, 0.15)
        med_cx  = rng.uniform(-0.08, 0.08)
        med_val = HU_BLOOD + rng.uniform(-0.02, 0.03)
        med_mask = smooth_ellipse(shape, med_cy, med_cx, med_ry, med_rx,
                                   angle_deg=rng.uniform(-20, 20), sigma=3.0)
        x = np.where(med_mask > 0.4, med_val, x)

    # ── 7. Fine texture (simulate noise/tissue heterogeneity) ─────────────────
    noise = fbm_noise(shape, octaves=4, persistence=0.55, base_sigma=4.0, rng=rng)
    body_mask = (body > 0.3).astype(np.float32)
    x = np.clip(x + 0.02 * (noise - 0.5) * body_mask, 0.0, 1.0)

    return x.astype(np.float32)


# ── Dev / Hidden procedural phantoms ──────────────────────────────────────────

def _recipe_typical_chest(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Standard chest cross-section — soft tissue + lungs + ribs."""
    return chest_phantom(rng, shape, n_ribs=rng.integers(3, 7))


def _recipe_large_patient(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Large patient — thicker fat layer, compressed lung cavities."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(4, 6),
                      fat_thickness=0.12)
    # Extra fat ring
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    r = np.sqrt(YY ** 2 + XX ** 2)
    fat_ring = np.clip(0.25 * np.exp(-(np.clip(r - 0.82, 0, None) / 0.06) ** 2), 0, 1)
    x = np.clip(x + HU_FAT * fat_ring, 0.0, 1.0)
    return x.astype(np.float32)


def _recipe_bone_heavy(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Dense spine + thick ribs, bone-dominant cross-section."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(5, 8), add_spine=True,
                      add_mediastinum=rng.random() < 0.5)
    x = add_bone_shell(x, rng, count=rng.integers(1, 3))
    return x.astype(np.float32)


def _recipe_asymmetric(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Asymmetric lungs — pleural effusion / mass on one side."""
    asym = rng.uniform(-0.8, 0.8)
    x = chest_phantom(rng, shape, lung_asymmetry=asym,
                      n_ribs=rng.integers(3, 6))
    # Add lesion on the compressed side
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    side = np.sign(asym) if asym != 0 else 1.0
    cy_l = rng.uniform(-0.1, 0.2)
    cx_l = side * rng.uniform(0.1, 0.35)
    ry_l = rng.uniform(0.06, 0.14)
    rx_l = rng.uniform(0.06, 0.14)
    yr = YY - cy_l; xr = XX - cx_l
    lesion = (yr ** 2 / ry_l ** 2 + xr ** 2 / rx_l ** 2 <= 1.0).astype(np.float32)
    lesion = gaussian_filter(lesion, sigma=3.0)
    x = np.clip(x + HU_BLOOD * lesion, 0.0, 1.0)
    return x.astype(np.float32)


# ── Hidden-tier adversarial recipes ───────────────────────────────────────────

def _recipe_metal(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Metal inserts (implants / leads) — beam-hardening stress test."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(3, 6))
    x = add_metal_insert(x, rng, count=rng.integers(1, 4))
    return x.astype(np.float32)


def _recipe_low_contrast(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Low-contrast lesions — lesion-detection-limit stress test."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(3, 5))
    # Add subtle lesions inside lung/tissue boundary
    h, w = shape
    yy = (np.arange(h) - h / 2.0) / (h / 2.0)
    xx = (np.arange(w) - w / 2.0) / (w / 2.0)
    XX, YY = np.meshgrid(xx, yy)
    for _ in range(rng.integers(3, 7)):
        cy = rng.uniform(-0.35, 0.35)
        cx = rng.uniform(-0.35, 0.35)
        ry = rng.uniform(0.025, 0.06)
        rx = rng.uniform(0.025, 0.06)
        delta = rng.uniform(0.02, 0.06)   # subtle contrast
        yr = YY - cy; xr = XX - cx
        les = (yr ** 2 / ry ** 2 + xr ** 2 / rx ** 2 <= 1.0).astype(np.float32)
        les = gaussian_filter(les, sigma=2.0)
        x = np.clip(x + delta * les, 0.0, 1.0)
    return x.astype(np.float32)


def _recipe_calcification_heavy(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Many small calcifications — thin-structure stress test."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(3, 6))
    x = add_calcifications(x, rng, count=rng.integers(12, 22))
    return x.astype(np.float32)


def _recipe_high_contrast_hidden(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """High dynamic range — bone/air/metal all present."""
    x = chest_phantom(rng, shape, n_ribs=rng.integers(5, 8), add_spine=True)
    x = add_metal_insert(x, rng, count=rng.integers(1, 3))
    x = add_bone_shell(x, rng, count=rng.integers(1, 2), mu=HU_BONE_DENSE * 1.2)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


_DEV_RECIPES = [
    (0.45, _recipe_typical_chest),
    (0.25, _recipe_large_patient),
    (0.20, _recipe_bone_heavy),
    (0.10, _recipe_asymmetric),
]
_HIDDEN_RECIPES = [
    (0.35, _recipe_metal),
    (0.35, _recipe_low_contrast),
    (0.20, _recipe_calcification_heavy),
    (0.10, _recipe_high_contrast_hidden),
]


def generate_ct_gt(
    seed: int,
    mode: str = "dev",
    shape: tuple[int, int] = (362, 362),
) -> tuple[np.ndarray, str]:
    """Generate a procedural CT phantom matching LoDoPaB-CT anatomy.

    Value scale matches LoDoPaB-CT statistics:
      - dev:    x_true clipped at 0.55  (≈ 1240 HU, dense cortical bone)
      - hidden: x_true clipped at 0.85  (allows metal implants > 2000 HU)

    Returns (x, recipe_name) where x is float32 in [0, 1].
    """
    if mode not in ("dev", "hidden"):
        raise ValueError(f"mode must be 'dev' or 'hidden', got {mode!r}")
    recipes = _DEV_RECIPES if mode == "dev" else _HIDDEN_RECIPES
    rng = np.random.default_rng(seed)
    probs = [p for p, _ in recipes]
    idx = rng.choice(len(recipes), p=probs)
    func = recipes[idx][1]
    x = func(rng, shape)
    recipe_name = func.__name__.replace("_recipe_", "")
    x_max = 0.55 if mode == "dev" else 0.85
    return np.clip(x, 0.0, x_max).astype(np.float32), recipe_name
