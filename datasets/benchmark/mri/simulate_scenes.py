"""Procedural MRI phantom generator for benchmark dev/hidden tiers.

Generates fully synthetic 2D MRI cross-sections in [0, 1] representing
the signal magnitude of a T2-weighted brain MRI. All phantoms are
procedurally generated — no external datasets required.

Layer-based generation:
  1) Brain mask  — elliptical FOV mask (randomised shape)
  2) White matter — smooth interior at medium T2 signal
  3) Gray matter  — slightly brighter rim (T2-weighted contrast)
  4) Ventricles   — bright CSF cavities (high T2)
  5) Vessels/structures — small bright features
  6) Stress patterns   — lesions, fine structure, HDR (hidden only)
  7) Postprocess       — blur, contrast, clipping

Recipe mix:
  Dev  (brain-like, easier):   60% gray_white_matter, 25% with_vessels, 15% fat_saturated
  Hidden (adversarial):        35% lesion_pathological, 35% fine_structure,
                               20% high_contrast, 10% edge_heavy

Usage:
    from simulate_scenes import generate_mri_gt
    x, recipe = generate_mri_gt(seed=42, mode="dev", shape=(256, 256))
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


# ── Layer primitives ──────────────────────────────────────────────────────────

def fbm_noise(
    shape: tuple[int, int],
    octaves: int = 5,
    persistence: float = 0.55,
    base_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fractional Brownian motion-like noise: sum of blurred white noises."""
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


def brain_mask(
    shape: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Elliptical brain mask with smooth boundary and random shape jitter."""
    rng = np.random.default_rng() if rng is None else rng
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = h / 2 + rng.uniform(-0.02, 0.02) * h
    cx = w / 2 + rng.uniform(-0.02, 0.02) * w
    ry = rng.uniform(0.38, 0.44) * h
    rx = rng.uniform(0.36, 0.42) * w
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    mask = (r <= 1.0).astype(np.float32)
    mask = gaussian_filter(mask, sigma=rng.uniform(1.5, 3.0))
    return np.clip(mask, 0.0, 1.0)


def add_ventricles(
    x: np.ndarray,
    rng: np.random.Generator,
    count: int = 2,
    value: float = 0.90,
) -> np.ndarray:
    """Bright CSF ventricle regions (high T2 signal ~0.9)."""
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = x.copy()
    for _ in range(count):
        cy = h / 2 + rng.uniform(-0.10, 0.10) * h
        cx = w / 2 + rng.uniform(-0.10, 0.10) * w
        ry = rng.uniform(0.04, 0.10) * h
        rx = rng.uniform(0.04, 0.12) * w
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        vtx = gaussian_filter(inside.astype(np.float32), sigma=rng.uniform(1.5, 4.0))
        vtx = (vtx / max(vtx.max(), 1e-6)) * rng.uniform(value - 0.05, value + 0.05)
        out = np.where(inside, vtx, out)
    return np.clip(out, 0.0, 1.0)


def add_vessels(
    x: np.ndarray,
    rng: np.random.Generator,
    count: int = 8,
) -> np.ndarray:
    """Small bright blood-vessel cross-sections (hyperintense point-like)."""
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = x.copy()
    for _ in range(count):
        cy = rng.uniform(0.2 * h, 0.8 * h)
        cx = rng.uniform(0.2 * w, 0.8 * w)
        r = rng.uniform(1.0, 4.0)
        intensity = rng.uniform(0.70, 1.0)
        vessel = intensity * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        out = np.maximum(out, vessel.astype(np.float32))
    return np.clip(out, 0.0, 1.0)


def add_lesions(
    x: np.ndarray,
    rng: np.random.Generator,
    count: int = 3,
) -> np.ndarray:
    """Bright focal lesions — hyperintense on T2 (e.g. MS plaques, tumours)."""
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = x.copy()
    for _ in range(count):
        cy = rng.uniform(0.25 * h, 0.75 * h)
        cx = rng.uniform(0.25 * w, 0.75 * w)
        ry = rng.uniform(0.03, 0.09) * h
        rx = rng.uniform(0.03, 0.09) * w
        intensity = rng.uniform(0.85, 1.0)
        lesion = intensity * np.exp(
            -(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) / 2.0
        )
        out = np.maximum(out, lesion.astype(np.float32))
    return np.clip(out, 0.0, 1.0)


def add_scalp_ring(
    x: np.ndarray,
    bmask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bright scalp / skull outer ring."""
    inner = bmask > 0.5
    outer = binary_dilation(inner, iterations=int(rng.integers(3, 9)))
    scalp = (outer & ~inner).astype(np.float32)
    scalp = gaussian_filter(scalp, sigma=rng.uniform(1.0, 2.5))
    intensity = rng.uniform(0.30, 0.60)
    return np.clip(x + intensity * scalp, 0.0, 1.0)


def add_thin_edges(
    x: np.ndarray,
    rng: np.random.Generator,
    count: int = 6,
) -> np.ndarray:
    """Sharp tissue-boundary bright/dark rims (edge stress-test)."""
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = x.copy()
    for _ in range(count):
        cy = rng.uniform(0.25 * h, 0.75 * h)
        cx = rng.uniform(0.25 * w, 0.75 * w)
        ry = rng.uniform(0.04, 0.14) * h
        rx = rng.uniform(0.04, 0.14) * w
        val = rng.uniform(0.2, 0.9)
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        inside = (dist <= 1.0).astype(np.float32)
        rim = np.abs(
            gaussian_filter(inside, sigma=0.5) - gaussian_filter(inside, sigma=2.0)
        )
        rim_max = rim.max()
        if rim_max > 1e-6:
            rim /= rim_max
        out = np.clip(out + val * rim, 0.0, 1.0)
    return out


# ── Recipe types ──────────────────────────────────────────────────────────────

def _recipe_gray_white_matter(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """60% of dev: classic T2 brain with GM/WM/CSF contrast."""
    h, w = shape
    bmask = brain_mask(shape, rng)
    # White matter: medium T2 signal (~0.50) with fBm heterogeneity
    wm = fbm_noise(shape, octaves=5, persistence=0.50, base_sigma=4.0, rng=rng) * 0.15 + 0.50
    # Gray matter: slightly brighter (~0.65) at the rim
    gm = fbm_noise(shape, octaves=4, persistence=0.55, base_sigma=2.0, rng=rng) * 0.12 + 0.65
    # Spatial weight: WM in interior, GM at rim
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r_norm = np.sqrt(((yy - cy) / (h * 0.35)) ** 2 + ((xx - cx) / (w * 0.33)) ** 2)
    wm_weight = np.clip(1.0 - r_norm, 0.0, 1.0)
    x = wm_weight * wm + (1.0 - wm_weight) * gm
    x = add_ventricles(x, rng, count=int(rng.integers(2, 5)))
    x *= bmask
    x = add_scalp_ring(x, bmask, rng)
    x = gaussian_filter(x, sigma=rng.uniform(0.5, 1.2))
    return np.clip(x, 0.0, 1.0)


def _recipe_with_vessels(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """25% of dev: brain with small visible vessel cross-sections."""
    x = _recipe_gray_white_matter(rng, shape)
    x = add_vessels(x, rng, count=int(rng.integers(5, 16)))
    return np.clip(x, 0.0, 1.0)


def _recipe_fat_saturated(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """15% of dev: fat-saturated T2 — no bright scalp rim."""
    shape_ = shape
    bmask = brain_mask(shape_, rng)
    base = fbm_noise(shape_, octaves=5, persistence=0.55, base_sigma=3.0, rng=rng)
    x = np.clip(base * 0.30 + 0.45, 0.0, 1.0)
    x = add_ventricles(x, rng, count=int(rng.integers(2, 4)), value=0.85)
    x *= bmask  # fat suppressed — no scalp ring
    x = gaussian_filter(x, sigma=rng.uniform(0.8, 1.5))
    return np.clip(x, 0.0, 1.0)


def _recipe_lesion_pathological(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """35% of hidden: T2 brain with hyperintense focal lesions."""
    x = _recipe_gray_white_matter(rng, shape)
    x = add_lesions(x, rng, count=int(rng.integers(2, 7)))
    return np.clip(x, 0.0, 1.0)


def _recipe_fine_structure(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """35% of hidden: many small vessels + fine texture (high spatial freq)."""
    x = _recipe_gray_white_matter(rng, shape)
    x = add_vessels(x, rng, count=int(rng.integers(15, 31)))
    fine = fbm_noise(shape, octaves=7, persistence=0.50, base_sigma=0.8, rng=rng) * 0.08
    x = np.clip(x + fine, 0.0, 1.0)
    return x


def _recipe_high_contrast(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """20% of hidden: extreme tissue contrast — stress-tests dynamic range."""
    h, w = shape
    bmask = brain_mask(shape, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r_norm = np.sqrt(((yy - cy) / (h * 0.35)) ** 2 + ((xx - cx) / (w * 0.33)) ** 2)
    wm_core = gaussian_filter((r_norm < 0.55).astype(np.float32), sigma=1.5)
    x = wm_core * 0.45 + (1.0 - wm_core) * 0.90  # WM=0.45, GM=0.90
    x = add_ventricles(x, rng, count=3, value=0.98)
    x *= bmask
    x = add_scalp_ring(x, bmask, rng)
    x = np.clip(x * rng.uniform(1.05, 1.30), 0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def _recipe_edge_heavy(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """10% of hidden: many sharp tissue-boundary rims (edge stress-test)."""
    x = _recipe_gray_white_matter(rng, shape)
    x = add_thin_edges(x, rng, count=int(rng.integers(6, 13)))
    return np.clip(x, 0.0, 1.0)


# ── Dev/Hidden recipe selection ───────────────────────────────────────────────

_DEV_RECIPES = [
    (0.60, _recipe_gray_white_matter),
    (0.25, _recipe_with_vessels),
    (0.15, _recipe_fat_saturated),
]

_HIDDEN_RECIPES = [
    (0.35, _recipe_lesion_pathological),
    (0.35, _recipe_fine_structure),
    (0.20, _recipe_high_contrast),
    (0.10, _recipe_edge_heavy),
]


def _pick_recipe(mode: str, rng: np.random.Generator):
    recipes = _DEV_RECIPES if mode == "dev" else _HIDDEN_RECIPES
    probs = [p for p, _ in recipes]
    funcs = [f for _, f in recipes]
    idx = rng.choice(len(funcs), p=probs)
    return funcs[idx]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_mri_gt(
    seed: int,
    mode: str = "dev",
    shape: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, str]:
    """Generate a procedural MRI ground-truth phantom.

    Args:
        seed: Random seed for reproducibility.
        mode: "dev" (brain-like, easier) or "hidden" (adversarial, harder).
        shape: Output image shape (H, W).

    Returns:
        (x, recipe_name) where x is float32 in [0, 1].
    """
    if mode not in ("dev", "hidden"):
        raise ValueError(f"mode must be 'dev' or 'hidden', got {mode!r}")
    rng = np.random.default_rng(seed)
    func = _pick_recipe(mode, rng)
    recipe_name = func.__name__.replace("_recipe_", "")
    x = func(rng, shape)
    return np.clip(x, 0.0, 1.0).astype(np.float32), recipe_name


def generate_batch(
    n: int,
    mode: str = "dev",
    base_seed: int = 5000,
    shape: tuple[int, int] = (256, 256),
) -> list[tuple[str, np.ndarray, str]]:
    """Generate a batch of procedural MRI phantoms.

    Returns list of (name, image, recipe_name) tuples.
    """
    scenes = []
    for i in range(n):
        x, recipe = generate_mri_gt(base_seed + i, mode=mode, shape=shape)
        scenes.append((f"proc_{mode}_{i:02d}", x, recipe))
    return scenes


def shepp_logan_phantom(
    shape: tuple[int, int] = (256, 256),
    variant: int = 0,
) -> np.ndarray:
    """Modified Shepp-Logan phantom with contrast variants for 11 public samples.

    Returns float32 in [0, 1].
    """
    H, W = shape
    yy = (np.arange(H) - H / 2.0) / (H / 2.0)
    xx = (np.arange(W) - W / 2.0) / (W / 2.0)
    XX, YY = np.meshgrid(xx, yy)

    # Standard Shepp-Logan ellipse parameters: (intensity, a, b, x0, y0, theta_deg)
    ellipses = [
        ( 1.00,  0.69,   0.92,   0.00,   0.000,   0),
        (-0.80,  0.6624, 0.874,  0.00,  -0.0184,  0),
        (-0.20,  0.11,   0.31,  -0.22,   0.000,  -18),
        (-0.20,  0.16,   0.41,   0.22,   0.000,   18),
        ( 0.10,  0.21,   0.25,   0.00,   0.350,   0),
        ( 0.10,  0.046,  0.046,  0.00,   0.100,   0),
        ( 0.10,  0.046,  0.046,  0.00,  -0.100,   0),
        (-0.02,  0.046,  0.023, -0.08,  -0.605,   0),
        ( 0.01,  0.023,  0.023,  0.00,  -0.606,   0),
        ( 0.01,  0.023,  0.046,  0.06,  -0.605,   0),
    ]

    phantom = np.zeros((H, W), dtype=np.float64)
    for intens, a, b, x0, y0, theta_deg in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = cos_t * (XX - x0) + sin_t * (YY - y0)
        yr = -sin_t * (XX - x0) + cos_t * (YY - y0)
        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        phantom[inside] += intens

    phantom = np.clip(phantom, 0.0, None)
    phantom /= phantom.max() + 1e-6

    # 11 contrast variants for distinct public samples
    contrasts = [1.00, 0.85, 0.95, 1.10, 0.90, 1.05, 0.80, 0.75, 1.15, 1.20, 0.88]
    phantom = np.clip(phantom * contrasts[variant % len(contrasts)], 0.0, 1.0)

    return phantom.astype(np.float32)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"Generating {n} {mode} MRI phantoms...")

    for name, x, recipe in generate_batch(n, mode=mode):
        print(f"  {name}: shape={x.shape} range=[{x.min():.3f}, {x.max():.3f}] recipe={recipe}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scenes = generate_batch(n, mode=mode)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, x, recipe) in zip(axes, scenes):
            ax.imshow(x, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name}\n{recipe}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig("_mri_preview.png", dpi=100)
        print(f"\n  Preview saved to _mri_preview.png")
    except Exception as e:
        print(f"  (No preview: {e})")
