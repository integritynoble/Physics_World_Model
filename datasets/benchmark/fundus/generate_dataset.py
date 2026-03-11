#!/usr/bin/env python3
"""Generate anatomically realistic Retinal Fundus Photography benchmark dataset.

Ground-truth phantoms mimic green-channel fundus images from DRIVE/STARE/HRF
databases:
  - Retinal pigment epithelium (RPE) background with spatially varying texture
  - Bright optic disc with physiological cup, neuroretinal rim
  - Fractal branching vessel tree (arteries + veins, 5-7 generations)
  - Anti-aliased vessels with Gaussian cross-section and width tapering
  - Macula and foveal pit (dark central region)
  - Circular fundus field-of-view mask with soft edge

Forward model (fundus camera degradation):
    y = scatter_field + illumination * conv(x_true, psf) + noise

where:
    x_true       -- ground-truth retinal image (256x256 grayscale, green channel)
    psf          -- optical PSF (Gaussian / Airy-like, simulates defocus)
    illumination -- non-uniform illumination field (brighter center, darker periphery)
    scatter_field-- low-frequency additive haze (media opacity / cataract)
    noise        -- Poisson-Gaussian sensor noise

Mismatch parameters:
    psf_sigma                  : PSF width / aberration severity (pixels)
    scatter_intensity          : scatter haze level [0,1]
    illumination_falloff       : vignetting severity [0,1]
    noise_level                : sensor noise std [0,1]

Tiers:
    public : 12 samples (varied healthy retinas, mild degradation)
    dev    : 20 samples (different vasculature seeds, medium degradation)
    hidden : 20 samples (different vasculature seeds, severe degradation)

CPU reconstruction: Wiener deconvolution + TV denoising + illumination correction.

Usage:
    cd datasets/benchmark/fundus
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# ---------------------------------------------------------------------------
# Physics / reflectance constants (green channel, typical fundus camera)
# ---------------------------------------------------------------------------
REFLECTANCE_RPE_BG    = 0.35   # retinal pigment epithelium background
REFLECTANCE_DISC      = 0.88   # optic disc (neuroretinal rim)
REFLECTANCE_CUP       = 0.93   # optic cup (paler center)
REFLECTANCE_FOVEA     = 0.18   # foveal pit (darkest retinal point)
REFLECTANCE_MACULA    = 0.27   # macular region
REFLECTANCE_ARTERY    = 0.52   # oxygenated blood (brighter red channel, mid green)
REFLECTANCE_VEIN      = 0.22   # deoxygenated blood (darker)
REFLECTANCE_NERVE_FL  = 0.40   # nerve fiber layer streaks (near disc)

# ---------------------------------------------------------------------------
# Mismatch spec ranges per tier
# ---------------------------------------------------------------------------
SPEC = {
    "public": {
        "psf_sigma":              {"min": 0.5, "max": 2.5, "unit": "pixels"},
        "scatter_intensity":      {"min": 0.00, "max": 0.06, "unit": ""},
        "illumination_falloff":   {"min": 0.00, "max": 0.15, "unit": ""},
        "noise_level":            {"min": 0.005, "max": 0.015, "unit": ""},
    },
    "dev": {
        "psf_sigma":              {"min": 1.0, "max": 4.0, "unit": "pixels"},
        "scatter_intensity":      {"min": 0.00, "max": 0.12, "unit": ""},
        "illumination_falloff":   {"min": 0.00, "max": 0.25, "unit": ""},
        "noise_level":            {"min": 0.008, "max": 0.025, "unit": ""},
    },
    "hidden": {
        "psf_sigma":              {"min": 1.5, "max": 6.0, "unit": "pixels"},
        "scatter_intensity":      {"min": 0.00, "max": 0.20, "unit": ""},
        "illumination_falloff":   {"min": 0.00, "max": 0.40, "unit": ""},
        "noise_level":            {"min": 0.010, "max": 0.040, "unit": ""},
    },
}

TIER_CONFIG = {
    "public": {"n_samples": 12, "base_seed": 0},
    "dev":    {"n_samples": 20, "base_seed": 10000},
    "hidden": {"n_samples": 20, "base_seed": 20000},
}


# ===========================================================================
# Low-level drawing primitives
# ===========================================================================

def _perlin_noise(H: int, W: int, scale: float, rng: np.random.Generator,
                  octaves: int = 4) -> np.ndarray:
    """Multi-octave smooth noise field normalized to [0,1]."""
    noise = np.zeros((H, W), dtype=np.float64)
    amp = 1.0
    s = scale
    for _ in range(octaves):
        raw = rng.standard_normal((H, W))
        noise += amp * gaussian_filter(raw, sigma=s)
        amp *= 0.5
        s = max(1.0, s * 0.5)
    lo, hi = noise.min(), noise.max()
    if hi - lo > 1e-10:
        noise = (noise - lo) / (hi - lo)
    return noise


def _soft_disc(H: int, W: int, cy: float, cx: float, radius: float,
               edge_width: float = 2.0) -> np.ndarray:
    """Smooth circular mask with configurable soft edge."""
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    return np.clip(1.0 - (dist - radius) / max(edge_width, 0.5), 0.0, 1.0)


def _soft_ellipse(H: int, W: int, cy: float, cx: float,
                  ry: float, rx: float, edge_width: float = 2.0) -> np.ndarray:
    """Smooth elliptical mask."""
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    dist = np.sqrt(((yy - cy) / max(ry, 1.0))**2 + ((xx - cx) / max(rx, 1.0))**2)
    return np.clip(1.0 - (dist - 1.0) / max(edge_width / min(ry, rx), 0.01), 0.0, 1.0)


# ===========================================================================
# Fractal retinal vessel tree generation
# ===========================================================================

def _draw_vessel_segment(canvas: np.ndarray, weight_map: np.ndarray,
                         y0: float, x0: float, y1: float, x1: float,
                         width: float, reflectance: float) -> None:
    """Draw a single anti-aliased vessel segment with Gaussian cross-section.

    Uses a distance-based alpha for smooth blending (no staircase artifacts).
    The weight_map accumulates coverage so overlapping vessels blend properly.
    """
    H, W = canvas.shape
    length = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    if length < 0.5 or width < 0.2:
        return

    n_pts = max(int(length * 4), 20)
    ts = np.linspace(0, 1, n_pts)
    ys = y0 + ts * (y1 - y0)
    xs = x0 + ts * (x1 - x0)

    # Width tapers linearly from start to end
    ws = np.linspace(width, width * 0.92, n_pts)

    for y, x, w in zip(ys, xs, ws):
        r_int = max(1, int(np.ceil(w + 1.5)))
        iy, ix = int(round(y)), int(round(x))
        y_lo = max(0, iy - r_int)
        y_hi = min(H, iy + r_int + 1)
        x_lo = max(0, ix - r_int)
        x_hi = min(W, ix + r_int + 1)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue

        # Sub-pixel distance computation for anti-aliasing
        yy = np.arange(y_lo, y_hi, dtype=np.float64)
        xx = np.arange(x_lo, x_hi, dtype=np.float64)
        dy2 = (yy - y)**2
        dx2 = (xx - x)**2
        dist2 = dy2[:, None] + dx2[None, :]
        sigma_v = max(w * 0.5, 0.3)
        alpha = np.exp(-dist2 / (2.0 * sigma_v**2))
        alpha = np.clip(alpha, 0.0, 1.0)

        # Alpha-blend: vessel reflectance overwrites background where alpha > 0
        old_weight = weight_map[y_lo:y_hi, x_lo:x_hi]
        new_weight = np.maximum(old_weight, alpha)
        # Only apply the incremental alpha
        inc = new_weight - old_weight
        canvas[y_lo:y_hi, x_lo:x_hi] = (
            canvas[y_lo:y_hi, x_lo:x_hi] * (1.0 - inc) + reflectance * inc
        )
        weight_map[y_lo:y_hi, x_lo:x_hi] = new_weight


def _bezier_midpoint(y0: float, x0: float, y1: float, x1: float,
                     tortuosity: float, rng: np.random.Generator
                     ) -> Tuple[float, float]:
    """Compute a random control point for a quadratic Bezier curve to add
    tortuosity (natural vessel curvature)."""
    mid_y = (y0 + y1) / 2.0
    mid_x = (x0 + x1) / 2.0
    length = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    # Perpendicular displacement
    dx = x1 - x0
    dy = y1 - y0
    perp_y = -dx / (length + 1e-8)
    perp_x = dy / (length + 1e-8)
    displacement = rng.normal(0, tortuosity * length * 0.15)
    return mid_y + perp_y * displacement, mid_x + perp_x * displacement


def _draw_curved_segment(canvas: np.ndarray, weight_map: np.ndarray,
                         y0: float, x0: float, y1: float, x1: float,
                         width: float, reflectance: float,
                         tortuosity: float, rng: np.random.Generator) -> None:
    """Draw a curved vessel segment using a quadratic Bezier midpoint."""
    ctrl_y, ctrl_x = _bezier_midpoint(y0, x0, y1, x1, tortuosity, rng)
    # Subdivide into 2 straight segments through the control point
    _draw_vessel_segment(canvas, weight_map, y0, x0, ctrl_y, ctrl_x,
                         width, reflectance)
    _draw_vessel_segment(canvas, weight_map, ctrl_y, ctrl_x, y1, x1,
                         width * 0.97, reflectance)


def generate_retinal_vessels(canvas: np.ndarray, weight_map: np.ndarray,
                             disc_cy: float, disc_cx: float,
                             seed: int, n_major: int = 8,
                             max_depth: int = 6,
                             tortuosity: float = 0.5,
                             density_scale: float = 1.0) -> None:
    """Generate a complete retinal vessel tree using recursive fractal branching.

    Arteries and veins alternate, originating from the optic disc. Each major
    vessel branches recursively with:
      - Width tapering (Murray's law: child_w ~ parent_w * 0.7)
      - Angular spread with randomness
      - Tortuosity (gentle curves via Bezier midpoints)
      - 5-7 branching generations for realistic density

    Parameters
    ----------
    canvas : (H, W) float64 array -- retinal image being built
    weight_map : (H, W) float64 array -- vessel coverage accumulator
    disc_cy, disc_cx : optic disc center
    seed : random seed for reproducibility
    n_major : number of major arcades (typically 6-10 for realistic retinas)
    max_depth : maximum recursion depth (branching generations)
    tortuosity : vessel curvature amplitude [0=straight, 1=very curved]
    density_scale : multiplier on branch count per generation
    """
    rng = np.random.default_rng(seed)
    H, W = canvas.shape

    # FOV mask to keep vessels inside the circular fundus field
    fov_cy, fov_cx = H / 2.0, W / 2.0
    fov_radius = H * 0.44

    def _in_fov(y, x):
        return ((y - fov_cy)**2 + (x - fov_cx)**2) < (fov_radius * 0.95)**2

    def _grow(y0, x0, angle_deg, length, width, refl, depth):
        """Recursive vessel growth with continuation + side-branches.

        At each step:
          1. Draw the current segment (curved)
          2. Continue the main trunk with a slight angle perturbation
          3. Spawn 1-2 side branches with wider angle deviation
        This produces the characteristic arterial/venous tree where a major
        vessel continues across the retina with smaller branches peeling off,
        matching the appearance of DRIVE/STARE/HRF datasets.
        """
        if depth > max_depth or length < 3.0 or width < 0.20:
            return
        if not _in_fov(y0, x0):
            return

        angle_rad = np.radians(angle_deg)
        end_y = y0 + length * np.sin(angle_rad)
        end_x = x0 + length * np.cos(angle_rad)

        # Clip to FOV
        if not _in_fov(end_y, end_x):
            for frac in np.linspace(0.9, 0.1, 9):
                ey = y0 + frac * length * np.sin(angle_rad)
                ex = x0 + frac * length * np.cos(angle_rad)
                if _in_fov(ey, ex):
                    end_y, end_x = ey, ex
                    length *= frac
                    break
            else:
                return

        # Draw this segment with curvature
        _draw_curved_segment(canvas, weight_map, y0, x0, end_y, end_x,
                             width, refl, tortuosity, rng)

        # -- Continuation branch (main trunk keeps going) --
        if depth < max_depth and width * 0.82 >= 0.20:
            cont_angle = angle_deg + float(rng.uniform(-12, 12))
            cont_width = width * float(rng.uniform(0.78, 0.88))
            cont_length = length * float(rng.uniform(0.60, 0.82))
            cont_refl = refl + float(rng.uniform(-0.01, 0.01))
            cont_refl = np.clip(cont_refl, 0.10, 0.70)
            _grow(end_y, end_x, cont_angle, cont_length,
                  cont_width, cont_refl, depth + 1)

        # -- Side branches (smaller vessels peeling off) --
        n_side = int(rng.integers(
            max(1, int(1 * density_scale)),
            max(2, int(3 * density_scale)) + 1
        ))
        if depth < 2:
            n_side = int(rng.integers(2, 4))
        elif depth >= max_depth - 1:
            n_side = int(rng.integers(0, 2))

        for _ in range(n_side):
            # Side branches have wider angular deviation
            side_sign = 1.0 if rng.random() > 0.5 else -1.0
            angle_offset = float(rng.uniform(25, 60)) * side_sign
            child_angle = angle_deg + angle_offset

            # Murray's law: side branch width ~ parent * 0.50-0.70
            child_width = width * float(rng.uniform(0.45, 0.70))
            child_length = length * float(rng.uniform(0.40, 0.65))

            child_refl = refl + float(rng.uniform(-0.02, 0.02))
            child_refl = np.clip(child_refl, 0.10, 0.70)

            _grow(end_y, end_x, child_angle, child_length,
                  child_width, child_refl, depth + 1)

    # ---- Generate major arcade vessels ----
    # Real retinal anatomy: 4 major arcades (STA, ITA, SNA, INA) plus
    # additional smaller vessels. Arteries and veins run roughly parallel.
    for i in range(n_major):
        base_angle = i * (360.0 / n_major)
        angle = base_angle + float(rng.uniform(-12, 12))

        is_artery = (i % 2 == 0)
        refl = REFLECTANCE_ARTERY if is_artery else REFLECTANCE_VEIN
        refl += float(rng.uniform(-0.04, 0.04))

        init_width = float(rng.uniform(3.2, 5.0)) if is_artery else float(rng.uniform(3.8, 5.5))
        init_length = float(rng.uniform(70, 120))

        _grow(disc_cy, disc_cx, angle, init_length, init_width, refl, 0)

    # Peri-papillary capillaries (small vessels close to disc)
    n_small = int(rng.integers(6, 12))
    for _ in range(n_small):
        angle = float(rng.uniform(0, 360))
        angle_rad = np.radians(angle)
        start_r = float(rng.uniform(5, 18))
        sy = disc_cy + start_r * np.sin(angle_rad)
        sx = disc_cx + start_r * np.cos(angle_rad)
        _draw_vessel_segment(
            canvas, weight_map, disc_cy, disc_cx, sy, sx,
            float(rng.uniform(0.6, 1.5)),
            float(rng.uniform(0.25, 0.45))
        )

    # Secondary arcade: a few extra vessels that arch around the macula
    # (temporal arcades curve around the fovea in real retinas)
    mac_cy_approx = H / 2.0
    mac_cx_approx = W * 0.42
    n_arcade = int(rng.integers(2, 5))
    for _ in range(n_arcade):
        # Start from disc, curve toward macula region
        start_angle = float(rng.uniform(-50, 50))  # mostly horizontal
        arc_width = float(rng.uniform(1.5, 3.0))
        arc_length = float(rng.uniform(40, 80))
        arc_refl = float(rng.uniform(0.25, 0.50))
        _grow(disc_cy, disc_cx, 180.0 + start_angle, arc_length,
              arc_width, arc_refl, 2)  # start at depth 2 for shorter trees


# ===========================================================================
# Retinal fundus phantom generator
# ===========================================================================

def generate_fundus_phantom(H: int, W: int, seed: int,
                            n_major_vessels: int = 8,
                            vessel_depth: int = 6,
                            tortuosity: float = 0.5,
                            vessel_density: float = 1.0,
                            disc_offset_y: float = 0.0,
                            disc_offset_x: float = 0.0,
                            bg_variation: float = 0.08,
                            pathology: str = "none"
                            ) -> Tuple[np.ndarray, str]:
    """Generate a single anatomically realistic retinal fundus image.

    Parameters
    ----------
    H, W : image dimensions
    seed : random seed
    n_major_vessels : number of major arcades from the optic disc
    vessel_depth : max branching depth (5-7 for DRIVE-like density)
    tortuosity : vessel curvature amplitude
    vessel_density : multiplier on number of child branches per generation
    disc_offset_y, disc_offset_x : optic disc position offset from default
    bg_variation : amplitude of RPE background texture variation
    pathology : "none", "mild", "moderate", or "severe"

    Returns
    -------
    image : (H, W) float64 array in [0, 1]
    name : descriptive string for the sample
    """
    rng = np.random.default_rng(seed)

    # ----- 1. RPE background texture -----
    # Multi-scale Perlin noise to simulate retinal pigment epithelium
    texture = _perlin_noise(H, W, scale=25.0, rng=rng, octaves=5)
    fine_tex = _perlin_noise(H, W, scale=8.0, rng=rng, octaves=3)
    image = np.full((H, W), REFLECTANCE_RPE_BG, dtype=np.float64)
    image += (texture - 0.5) * bg_variation
    image += (fine_tex - 0.5) * (bg_variation * 0.3)

    # ----- 2. Circular FOV mask -----
    fov_cy, fov_cx = H / 2.0, W / 2.0
    fov_radius = H * 0.45
    fov_mask = _soft_disc(H, W, fov_cy, fov_cx, fov_radius, edge_width=3.0)
    image *= fov_mask

    # ----- 3. Optic disc (nasal side, slightly above/below center) -----
    disc_cy = H * (0.48 + float(rng.uniform(-0.03, 0.03)) + disc_offset_y)
    disc_cx = W * (0.72 + float(rng.uniform(-0.04, 0.04)) + disc_offset_x)
    disc_ry = float(rng.uniform(16, 22))
    disc_rx = float(rng.uniform(15, 21))
    disc_mask = _soft_ellipse(H, W, disc_cy, disc_cx, disc_ry, disc_rx,
                              edge_width=2.5)
    image = image * (1.0 - disc_mask) + disc_mask * REFLECTANCE_DISC

    # Optic cup (paler center, ~0.3-0.6 of disc size)
    cup_ratio = float(rng.uniform(0.30, 0.55))
    cup_mask = _soft_ellipse(H, W, disc_cy, disc_cx,
                             disc_ry * cup_ratio, disc_rx * cup_ratio,
                             edge_width=2.0)
    image = image * (1.0 - cup_mask * 0.5) + cup_mask * 0.5 * REFLECTANCE_CUP

    # ----- 4. Nerve fiber layer streaks radiating from disc -----
    nfl_rng = np.random.default_rng(seed + 7777)
    n_streaks = int(nfl_rng.integers(8, 16))
    nfl_canvas = np.zeros((H, W), dtype=np.float64)
    nfl_weight = np.zeros((H, W), dtype=np.float64)
    for _ in range(n_streaks):
        angle = float(nfl_rng.uniform(0, 360))
        angle_rad = np.radians(angle)
        streak_len = float(nfl_rng.uniform(20, 50))
        end_y = disc_cy + streak_len * np.sin(angle_rad)
        end_x = disc_cx + streak_len * np.cos(angle_rad)
        _draw_vessel_segment(nfl_canvas, nfl_weight, disc_cy, disc_cx,
                             end_y, end_x,
                             float(nfl_rng.uniform(1.5, 4.0)),
                             REFLECTANCE_NERVE_FL)
    # Blend nerve fiber layer subtly
    nfl_alpha = np.clip(nfl_weight * 0.25, 0.0, 0.25)
    image = image * (1.0 - nfl_alpha) + nfl_canvas * nfl_alpha

    # ----- 5. Macula and fovea -----
    mac_cy = H * (0.50 + float(rng.uniform(-0.02, 0.02)))
    mac_cx = W * (0.42 + float(rng.uniform(-0.03, 0.03)))
    mac_radius = float(rng.uniform(30, 42))
    mac_mask = _soft_disc(H, W, mac_cy, mac_cx, mac_radius, edge_width=8.0)
    image = image * (1.0 - mac_mask * 0.20) + mac_mask * 0.20 * REFLECTANCE_MACULA

    # Foveal pit (small, very dark)
    fov_radius_small = float(rng.uniform(6, 10))
    fov_mask_small = _soft_disc(H, W, mac_cy, mac_cx, fov_radius_small,
                                edge_width=3.0)
    image = image * (1.0 - fov_mask_small * 0.65) + fov_mask_small * 0.65 * REFLECTANCE_FOVEA

    # Foveal reflex (tiny bright spot at center)
    reflex_mask = _soft_disc(H, W, mac_cy, mac_cx, 2.0, edge_width=1.5)
    image = image * (1.0 - reflex_mask * 0.3) + reflex_mask * 0.3 * 0.55

    # ----- 6. Vessel tree (fractal branching) -----
    vessel_weight = np.zeros((H, W), dtype=np.float64)
    generate_retinal_vessels(
        image, vessel_weight, disc_cy, disc_cx,
        seed=seed + 50000,
        n_major=n_major_vessels,
        max_depth=vessel_depth,
        tortuosity=tortuosity,
        density_scale=vessel_density,
    )

    # ----- 7. Pathological features -----
    name_parts = []
    if pathology != "none":
        path_rng = np.random.default_rng(seed + 90000)
        severity_map = {"mild": 1.0, "moderate": 2.0, "severe": 3.0}
        sev = severity_map.get(pathology, 1.0)

        # Microaneurysms (tiny dark dots near vessels)
        n_ma = int(path_rng.integers(3, int(8 * sev)))
        for _ in range(n_ma):
            my = float(path_rng.uniform(30, H - 30))
            mx = float(path_rng.uniform(30, W - 30))
            if fov_mask[int(my), int(mx)] < 0.5:
                continue
            dot_r = float(path_rng.uniform(1.0, 2.5))
            dot = _soft_disc(H, W, my, mx, dot_r, edge_width=1.0)
            image = image * (1.0 - dot * 0.8) + dot * 0.8 * 0.10

        # Hemorrhages (irregular dark patches)
        n_hem = int(path_rng.integers(1, int(4 * sev)))
        for _ in range(n_hem):
            hy = float(path_rng.uniform(40, H - 40))
            hx = float(path_rng.uniform(40, W - 40))
            if fov_mask[int(hy), int(hx)] < 0.5:
                continue
            hr = float(path_rng.uniform(4, 12 * sev))
            hem = _soft_disc(H, W, hy, hx, hr, edge_width=3.0)
            noise_mask = _perlin_noise(H, W, scale=8, rng=path_rng, octaves=2)
            hem *= (noise_mask > 0.4).astype(np.float64)
            hem = gaussian_filter(hem, sigma=1.5)
            hem = np.clip(hem, 0.0, 1.0)
            image = image * (1.0 - hem * 0.7) + hem * 0.7 * 0.08

        # Hard exudates (bright yellowish deposits)
        n_exu = int(path_rng.integers(2, int(6 * sev)))
        for _ in range(n_exu):
            ey = float(path_rng.uniform(50, H - 50))
            ex = float(path_rng.uniform(50, W - 50))
            if fov_mask[int(ey), int(ex)] < 0.5:
                continue
            er = float(path_rng.uniform(2, 7))
            exu = _soft_disc(H, W, ey, ex, er, edge_width=1.5)
            image = image * (1.0 - exu * 0.65) + exu * 0.65 * float(path_rng.uniform(0.72, 0.90))

        # Drusen (pale deposits, larger)
        n_dru = int(path_rng.integers(1, int(5 * sev)))
        for _ in range(n_dru):
            dy = float(path_rng.uniform(H // 4, 3 * H // 4))
            dx = float(path_rng.uniform(W // 4, 3 * W // 4))
            if fov_mask[int(dy), int(dx)] < 0.5:
                continue
            dr = float(path_rng.uniform(3, 9))
            dru = _soft_disc(H, W, dy, dx, dr, edge_width=2.0)
            image = image * (1.0 - dru * 0.55) + dru * 0.55 * float(path_rng.uniform(0.60, 0.80))

        name_parts.append(f"path_{pathology[0]}")
    else:
        name_parts.append("healthy")

    # ----- 8. Final masking and smoothing -----
    image *= fov_mask
    # Very mild smoothing to simulate optical diffraction limit
    image = gaussian_filter(image, sigma=0.4)
    image = np.clip(image, 0.0, 1.0)

    name = f"{'_'.join(name_parts)}_{seed:05d}"
    return image, name


# ===========================================================================
# Phantom pools per tier
# ===========================================================================

def _generate_phantoms(n: int, base_seed: int, seed_offset: int,
                       pathology_options: list,
                       tortuosity_range: Tuple[float, float] = (0.3, 0.8),
                       depth_range: Tuple[int, int] = (5, 7),
                       n_major_range: Tuple[int, int] = (6, 10),
                       density_range: Tuple[float, float] = (0.8, 1.3),
                       ) -> list:
    """Generate a pool of diverse retinal phantoms."""
    phantoms = []
    meta_rng = np.random.default_rng(base_seed + 99)
    for i in range(n):
        seed = base_seed + seed_offset + i * 137  # spread seeds for diversity
        pathology = pathology_options[i % len(pathology_options)]
        tort = float(meta_rng.uniform(*tortuosity_range))
        depth = int(meta_rng.integers(depth_range[0], depth_range[1] + 1))
        n_maj = int(meta_rng.integers(n_major_range[0], n_major_range[1] + 1))
        dens = float(meta_rng.uniform(*density_range))
        disc_oy = float(meta_rng.uniform(-0.04, 0.04))
        disc_ox = float(meta_rng.uniform(-0.06, 0.06))
        bg_var = float(meta_rng.uniform(0.05, 0.12))

        img, name = generate_fundus_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed,
            n_major_vessels=n_maj,
            vessel_depth=depth,
            tortuosity=tort,
            vessel_density=dens,
            disc_offset_y=disc_oy,
            disc_offset_x=disc_ox,
            bg_variation=bg_var,
            pathology=pathology,
        )
        phantoms.append((img, name))
    return phantoms


def generate_phantoms_public(n: int = 12) -> list:
    return _generate_phantoms(
        n, base_seed=100, seed_offset=0,
        pathology_options=["none", "none", "none", "mild",
                           "none", "none", "mild", "none",
                           "none", "none", "none", "mild"],
        tortuosity_range=(0.3, 0.7),
        depth_range=(5, 7),
        n_major_range=(6, 10),
        density_range=(0.8, 1.2),
    )


def generate_phantoms_dev(n: int = 20) -> list:
    return _generate_phantoms(
        n, base_seed=5000, seed_offset=10000,
        pathology_options=["none", "mild", "moderate", "none", "mild",
                           "none", "moderate", "none", "mild", "none",
                           "none", "mild", "none", "moderate", "none",
                           "mild", "none", "none", "moderate", "mild"],
        tortuosity_range=(0.2, 0.9),
        depth_range=(5, 7),
        n_major_range=(6, 10),
        density_range=(0.7, 1.4),
    )


def generate_phantoms_hidden(n: int = 20) -> list:
    return _generate_phantoms(
        n, base_seed=8000, seed_offset=20000,
        pathology_options=["none", "moderate", "severe", "mild", "moderate",
                           "none", "severe", "moderate", "mild", "none",
                           "severe", "moderate", "none", "mild", "severe",
                           "moderate", "none", "severe", "mild", "moderate"],
        tortuosity_range=(0.2, 1.0),
        depth_range=(5, 7),
        n_major_range=(6, 12),
        density_range=(0.7, 1.5),
    )


# ===========================================================================
# PSF generation
# ===========================================================================

def make_defocus_psf(psf_sigma: float, size: int = 31) -> np.ndarray:
    """Generate Gaussian defocus PSF."""
    if size % 2 == 0:
        size += 1
    sigma = max(0.3, min(psf_sigma, size / 3.0))
    c = size // 2
    yy = np.arange(size, dtype=np.float64) - c
    yy, xx = np.meshgrid(yy, yy, indexing='ij')
    psf = np.exp(-(yy**2 + xx**2) / (2.0 * sigma**2 + 1e-10))
    psf /= psf.sum()
    return psf


# ===========================================================================
# Illumination field
# ===========================================================================

def make_illumination_field(H: int, W: int, falloff: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Non-uniform illumination: brighter center, darker periphery (vignetting).

    Models cosine-4th-law falloff of a fundus camera.
    """
    cy = H / 2.0 + float(rng.uniform(-5, 5))
    cx = W / 2.0 + float(rng.uniform(-5, 5))
    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    r2 = yy**2 + xx**2
    r_max2 = cy**2 + cx**2 + 1e-6
    cos_theta = np.clip(1.0 - 0.5 * r2 / r_max2, 0.1, 1.0)
    return 1.0 - falloff * (1.0 - cos_theta**4)


# ===========================================================================
# Forward model
# ===========================================================================

def _make_H_ideal(psf: np.ndarray, H: int, W: int) -> np.ndarray:
    """Embed PSF kernel into a full-sized array for Fourier-domain operations."""
    psf_pad = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    y0 = (H - kh) // 2
    x0 = (W - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)
    return psf_pad.astype(np.float32)


def fundus_forward_model(x_true: np.ndarray, psf_sigma: float,
                         scatter_intensity: float, illumination_falloff: float,
                         noise_level: float,
                         rng: np.random.Generator) -> dict:
    """Apply the fundus camera forward model:

        y = scatter_field + illumination * conv(x_true, psf) + noise

    Returns dict with all intermediate products.
    """
    H, W = x_true.shape

    # PSF
    psf_size = min(61, max(11, int(6 * psf_sigma + 1) | 1))
    if psf_size % 2 == 0:
        psf_size += 1
    psf = make_defocus_psf(psf_sigma, size=psf_size)

    # Illumination field
    illum = make_illumination_field(H, W, illumination_falloff, rng)

    # Blur
    if psf_sigma > 0.3:
        blurred = fftconvolve(x_true, psf, mode='same')
    else:
        blurred = x_true.copy()

    # Apply illumination
    lit = blurred * illum

    # Scatter / media opacity haze
    scatter = np.zeros((H, W), dtype=np.float64)
    if scatter_intensity > 0.001:
        # Low-frequency haze from vitreous / cataract
        haze_base = gaussian_filter(blurred, sigma=40) * 0.6 + 0.4
        scatter = scatter_intensity * haze_base
        degraded = (1.0 - scatter_intensity) * lit + scatter
    else:
        degraded = lit

    degraded = np.clip(degraded, 0.0, 1.0)
    image_ideal = degraded.astype(np.float32)

    # Poisson-Gaussian noise
    peak = max(1.0 / (noise_level**2 + 1e-10), 100.0)
    poisson = rng.poisson(np.clip(degraded * peak, 0.01, None)).astype(np.float64) / peak
    readout = rng.normal(0, noise_level * 0.3, (H, W))
    measured = np.clip(poisson + readout, 0.0, 1.0).astype(np.float32)

    return {
        "image_ideal": image_ideal,
        "image_measured": measured,
        "y": measured,
        "H_ideal": _make_H_ideal(psf, H, W),
        "psf": psf.astype(np.float32),
        "illumination_field": illum.astype(np.float32),
        "scatter_field": scatter.astype(np.float32),
    }


# ===========================================================================
# CPU reconstruction (Wiener + TV + illumination correction)
# ===========================================================================

def wiener_deconvolution(image: np.ndarray, psf: np.ndarray,
                         noise_power: float = 0.01) -> np.ndarray:
    H_img, W_img = image.shape
    psf_pad = np.zeros((H_img, W_img), dtype=np.float64)
    kh, kw = psf.shape
    y0 = (H_img - kh) // 2
    x0 = (W_img - kw) // 2
    psf_pad[y0:y0 + kh, x0:x0 + kw] = psf
    psf_pad = np.roll(psf_pad, -(y0 + kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(x0 + kw // 2), axis=1)
    Y = np.fft.fft2(image.astype(np.float64))
    Hf = np.fft.fft2(psf_pad)
    return np.real(np.fft.ifft2(np.conj(Hf) / (np.abs(Hf)**2 + noise_power) * Y))


def _tv_denoise(image: np.ndarray, weight: float, n_iter: int = 30) -> np.ndarray:
    """Chambolle TV denoising."""
    img = image.astype(np.float64)
    H, W = img.shape
    px = np.zeros((H, W), dtype=np.float64)
    py = np.zeros((H, W), dtype=np.float64)
    tau = 0.25
    for _ in range(n_iter):
        div_p = np.zeros_like(img)
        div_p[:-1, :] += px[:-1, :]
        div_p[1:, :]  -= px[:-1, :]
        div_p[:, :-1] += py[:, :-1]
        div_p[:, 1:]  -= py[:, :-1]
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
    div_p[1:, :]  -= px[:-1, :]
    div_p[:, :-1] += py[:, :-1]
    div_p[:, 1:]  -= py[:, :-1]
    return img - weight * div_p


def reconstruct_cpu(image_measured: np.ndarray, psf: np.ndarray,
                    noise_sigma: float) -> np.ndarray:
    """Wiener deconvolution + illumination correction + TV denoising."""
    img = image_measured.astype(np.float64)

    # Estimate and correct non-uniform illumination
    illum_est = gaussian_filter(img, sigma=50)
    illum_est = np.clip(illum_est, 0.05, None)
    illum_est /= (illum_est.mean() + 1e-8)
    img_c = np.clip(img / illum_est, 0.0, 1.0)

    # Wiener deconvolution
    kh = psf.shape[0]
    yy = np.arange(kh, dtype=np.float64) - kh // 2
    psf64 = psf.astype(np.float64)
    psf_var = np.sum(psf64 * yy[:, None]**2) + np.sum(psf64 * yy[None, :]**2)
    psf_sig = max(0.5, np.sqrt(max(psf_var, 0.0)))
    reg = max(noise_sigma**2 * (1.0 + 0.003 * psf_sig**2), 5e-5)
    recon = np.clip(wiener_deconvolution(img_c, psf, noise_power=reg), 0.0, 1.0)

    # TV denoising
    tv_w = max(0.02, noise_sigma * 3.0)
    recon = np.clip(_tv_denoise(recon, weight=tv_w, n_iter=40), 0.0, 1.0)

    return recon.astype(np.float32)


# ===========================================================================
# Metrics
# ===========================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64))**2)
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr**2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    g = gt.astype(np.float64)
    r = recon.astype(np.float64)
    dr = g.max() - g.min()
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr)**2
    c2 = (0.03 * dr)**2
    mx, my = g.mean(), r.mean()
    vx, vy = g.var(), r.var()
    cov = np.mean((g - mx) * (r - my))
    return float((2 * mx * my + c1) * (2 * cov + c2) /
                 ((mx**2 + my**2 + c1) * (vx + vy + c2)))


# ===========================================================================
# Image I/O helpers
# ===========================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, img_ideal, img_meas, recon, path):
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(img_ideal)
    ov[:, 2*tw:3*tw] = _r(img_meas)
    ov[:, 3*tw:4*tw] = _r(recon)
    _save_png(ov, path)


# ===========================================================================
# Tier generation
# ===========================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier: str, phantoms: list, base_seed: int) -> None:
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
            f"(PSF blur + scatter + illumination non-uniformity + noise)")
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "fov_deg": 30.0,
            "pixel_size_um": 23.4,
        })
        f.attrs["forward_model"] = (
            "y = scatter_field + illumination * conv(x_true, psf) + noise")

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            result = fundus_forward_model(
                x_true,
                psf_sigma=mis["psf_sigma"],
                scatter_intensity=mis["scatter_intensity"],
                illumination_falloff=mis["illumination_falloff"],
                noise_level=mis["noise_level"],
                rng=rng,
            )

            recon = reconstruct_cpu(result["image_measured"], result["psf"],
                                    mis["noise_level"])
            psnr = compute_psnr(x_true.astype(np.float32), recon)
            ssim = compute_ssim(x_true.astype(np.float32), recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=result["y"], compression="gzip")
            grp.create_dataset("H_ideal", data=result["H_ideal"],
                               compression="gzip")
            grp.create_dataset("image_ideal", data=result["image_ideal"],
                               compression="gzip")
            grp.create_dataset("image_measured", data=result["image_measured"],
                               compression="gzip")
            grp.create_dataset("psf", data=result["psf"], compression="gzip")
            grp.create_dataset("illumination_field",
                               data=result["illumination_field"],
                               compression="gzip")
            grp.create_dataset("scatter_field", data=result["scatter_field"],
                               compression="gzip")
            grp.create_dataset("reconstruction_wiener", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psf_shape": list(result["psf"].shape),
                "psnr_wiener": float(psnr),
                "ssim_wiener": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Per-sample images
            sd = images_dir / f"sample_{idx:02d}_{scene_name}"
            sd.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sd / "gt.png")
            _save_png(result["image_measured"], sd / "measurement.png")
            _save_png(recon, sd / "recon.png")
            _save_overview(x_true, result["image_ideal"],
                           result["image_measured"], recon,
                           sd / "overview.png")
            with open(sd / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_wiener": psnr,
                    "ssim_wiener": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} SSIM={ssim:.3f}")

    # Tier-level spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    avg_psnr = float(np.mean(all_psnrs))
    avg_ssim = float(np.mean(all_ssims))
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={avg_psnr:.2f} dB | Mean SSIM={avg_ssim:.3f}")


# ===========================================================================
# Gallery images for the platform
# ===========================================================================

def generate_gallery_images() -> None:
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "fundus")
    h5_path = BENCHMARK_DIR / "public" / "fundus_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] public HDF5 not found, skipping gallery")
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


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("Retinal Fundus Photography Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Forward model: y = scatter + illumination * conv(x, psf) + noise")
    print(f"Vessel generation: fractal branching (5-7 generations)")
    print()

    print("Generating public tier (12 samples)...")
    generate_tier("public", generate_phantoms_public(12), base_seed=0)

    print("\nGenerating dev tier (20 samples)...")
    generate_tier("dev", generate_phantoms_dev(20), base_seed=10000)

    print("\nGenerating hidden tier (20 samples)...")
    generate_tier("hidden", generate_phantoms_hidden(20), base_seed=20000)

    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Fundus benchmark complete!")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
