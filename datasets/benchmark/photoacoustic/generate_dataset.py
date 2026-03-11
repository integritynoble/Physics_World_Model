#!/usr/bin/env python3
"""Generate the photoacoustic tomography (PAT) benchmark dataset.

Photoacoustic imaging: a pulsed laser causes thermoelastic expansion in
tissue chromophores (hemoglobin, melanin); the resulting broadband
ultrasound pressure waves are recorded by a circular transducer array
and reconstructed into an initial-pressure map proportional to the
optical absorption coefficient times the local fluence.

Forward model
-------------
    y(d, t) = sum over pixels on arc of radius c*t centred at detector d
              of p_0(pixel)

    The measurement y is a (N_det, N_t) sinogram-like array.
    H_ideal is stored as a geometry descriptor (transducer positions +
    time axis) rather than a dense matrix, to keep file sizes manageable.

Mismatch parameters per tier
-----------------------------
    speed_of_sound_error_pct : +/- % perturbation of assumed c
    angular_coverage_deg     : actual transducer arc (< 360 -> limited-view)
    bandwidth_limit_MHz      : low-pass cutoff (finite transducer response)
    noise_level              : additive Gaussian sigma relative to signal peak

Phantom types
-------------
Vascular networks: procedural blood-vessel trees, tumor vasculature
clusters, and superficial skin-vessel layers.  High-absorption structures
in a low-absorption background, normalised to [0, 1].

CPU baseline
------------
Universal Back-Projection (UBP): time-reversal adjoint summation over the
transducer arc.  Expected ~20-26 dB depending on limited-view severity.

Tiers
-----
    public : 12 samples, moderate mismatch
    dev    : 20 samples, medium mismatch + augmentation
    hidden : 20 samples, heavy mismatch + adversarial phantoms

Seeds: public=0, dev=10000, hidden=20000

Usage:
    cd datasets/benchmark/photoacoustic
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# -- Physics constants -------------------------------------------------------

SPEED_OF_SOUND = 1540.0          # m/s nominal (soft tissue / water)
PIXEL_SIZE_MM = 0.10             # mm per pixel  => FOV = 25.6 mm
N_DET_DEFAULT = 64               # transducer elements on the arc
N_TIME_DEFAULT = 128             # time samples per detector
DETECTOR_RADIUS_PX = 140.0       # radius of transducer ring (pixels)
ANGULAR_COVERAGE_DEG = 240.0     # < 360 -> limited-view
BANDWIDTH_MHZ = 10.0             # transducer bandwidth cutoff
NOISE_LEVEL = 0.02               # sigma / signal_peak

# -- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "speed_of_sound_error_pct": {"min": -2.0,  "max":  2.0, "unit": "%"},
        "angular_coverage_deg":     {"min": 200.0, "max": 300.0, "unit": "deg"},
        "bandwidth_limit_MHz":      {"min": 6.0,   "max": 12.0, "unit": "MHz"},
        "noise_level":              {"min": 0.01,  "max": 0.04, "unit": ""},
    },
    "dev": {
        "speed_of_sound_error_pct": {"min": -4.0,  "max":  4.0, "unit": "%"},
        "angular_coverage_deg":     {"min": 160.0, "max": 280.0, "unit": "deg"},
        "bandwidth_limit_MHz":      {"min": 4.0,   "max": 10.0, "unit": "MHz"},
        "noise_level":              {"min": 0.02,  "max": 0.08, "unit": ""},
    },
    "hidden": {
        "speed_of_sound_error_pct": {"min": -6.0,  "max":  6.0, "unit": "%"},
        "angular_coverage_deg":     {"min": 120.0, "max": 260.0, "unit": "deg"},
        "bandwidth_limit_MHz":      {"min": 3.0,   "max": 8.0,  "unit": "MHz"},
        "noise_level":              {"min": 0.04,  "max": 0.15, "unit": ""},
    },
}


# ============================================================================
# Phantom generators -- vascular networks
# ============================================================================

def _draw_vessel_segment(
    canvas: np.ndarray,
    cy: float, cx: float,
    thick: float, amp: float,
    size: int,
) -> None:
    """Draw a single vessel cross-section (Gaussian profile) at (cy, cx)."""
    iy, ix = int(round(cy)), int(round(cx))
    r = max(1, int(np.ceil(thick * 2.5)))
    y_lo = max(0, iy - r)
    y_hi = min(size, iy + r + 1)
    x_lo = max(0, ix - r)
    x_hi = min(size, ix + r + 1)
    if y_lo >= y_hi or x_lo >= x_hi:
        return
    yy = np.arange(y_lo, y_hi, dtype=np.float64) - cy
    xx = np.arange(x_lo, x_hi, dtype=np.float64) - cx
    YY, XX = np.meshgrid(yy, xx, indexing="ij")
    profile = amp * np.exp(-(YY ** 2 + XX ** 2) / (2.0 * thick ** 2))
    canvas[y_lo:y_hi, x_lo:x_hi] = np.maximum(
        canvas[y_lo:y_hi, x_lo:x_hi], profile)


def _make_vessel_tree(
    rng: np.random.Generator,
    size: int,
    n_branches: int = 60,
    thickness_range: tuple = (1.0, 4.0),
    amplitude_range: tuple = (0.5, 1.0),
) -> np.ndarray:
    """Procedural vessel tree via recursive branching random walk.

    Roots start inside the image (at 15-25% from an edge) pointing toward
    the centre, guaranteeing they always generate visible vessels.
    """
    canvas = np.zeros((size, size), dtype=np.float64)
    centre = size / 2.0

    # Root vessels: start inside the image, point roughly toward centre
    n_roots = int(rng.integers(2, 5))
    queue = []
    for _ in range(n_roots):
        side = rng.integers(0, 4)
        margin = rng.integers(size // 7, size // 4)
        if side == 0:      # near top
            ry = float(margin)
            rx = float(rng.integers(size // 5, 4 * size // 5))
        elif side == 1:    # near bottom
            ry = float(size - 1 - margin)
            rx = float(rng.integers(size // 5, 4 * size // 5))
        elif side == 2:    # near left
            ry = float(rng.integers(size // 5, 4 * size // 5))
            rx = float(margin)
        else:              # near right
            ry = float(rng.integers(size // 5, 4 * size // 5))
            rx = float(size - 1 - margin)

        # Angle pointing toward centre with some jitter
        angle = np.arctan2(centre - ry, centre - rx) + rng.normal(0.0, 0.3)
        thick = rng.uniform(
            max(thickness_range[0], thickness_range[1] * 0.6),
            thickness_range[1])
        amp = rng.uniform(
            max(amplitude_range[0], amplitude_range[1] * 0.6),
            amplitude_range[1])
        queue.append((ry, rx, angle, thick, amp, 0))

    branches_drawn = 0
    while queue and branches_drawn < n_branches:
        cy, cx, angle, thick, amp, depth = queue.pop(0)
        walk_len = int(rng.integers(20, 70))
        for _ in range(walk_len):
            angle += rng.normal(0.0, 0.08)
            ny = cy + np.sin(angle)
            nx = cx + np.cos(angle)
            if ny < 1 or ny >= size - 1 or nx < 1 or nx >= size - 1:
                break
            _draw_vessel_segment(canvas, ny, nx, thick, amp, size)
            cy, cx = ny, nx
            if depth < 5 and rng.random() < 0.04 * max(1, 3 - depth):
                new_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
                new_thick = thick * rng.uniform(0.5, 0.8)
                new_amp = amp * rng.uniform(0.6, 0.9)
                if new_thick >= thickness_range[0] * 0.5:
                    queue.append((cy, cx, new_angle, new_thick,
                                  new_amp, depth + 1))
        branches_drawn += 1

    # Safety: if canvas is still empty, place a central star pattern
    if canvas.max() < 0.01:
        n_star = rng.integers(3, 6)
        for _ in range(n_star):
            ang = rng.uniform(0, 2 * np.pi)
            cy, cx = centre, centre
            thick = rng.uniform(1.5, 3.0)
            amp = rng.uniform(0.5, 0.9)
            for s in range(int(size * 0.35)):
                ny = cy + np.sin(ang)
                nx = cx + np.cos(ang)
                ang += rng.normal(0.0, 0.05)
                if ny < 1 or ny >= size - 1 or nx < 1 or nx >= size - 1:
                    break
                _draw_vessel_segment(canvas, ny, nx, thick, amp, size)
                cy, cx = ny, nx

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def _make_tumor_vasculature(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Tumor vasculature: dense chaotic vessels near a tumour core."""
    canvas = np.zeros((size, size), dtype=np.float64)

    tc_y = rng.integers(size // 4, 3 * size // 4)
    tc_x = rng.integers(size // 4, 3 * size // 4)
    t_radius = rng.integers(25, 55)

    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - tc_y) ** 2 + (xx - tc_x) ** 2)
    canvas[dist < t_radius] += rng.uniform(0.05, 0.15)

    # Dense tortuous vessels inside tumour
    for _ in range(int(rng.integers(20, 40))):
        angle = rng.uniform(0, 2 * np.pi)
        start_r = rng.uniform(0, t_radius * 0.8)
        cy = float(tc_y + start_r * np.sin(angle))
        cx = float(tc_x + start_r * np.cos(angle))
        walk_angle = rng.uniform(0, 2 * np.pi)
        thick = rng.uniform(0.8, 2.5)
        amp = rng.uniform(0.5, 1.0)
        for _ in range(int(rng.integers(8, 30))):
            walk_angle += rng.normal(0.0, 0.2)
            cy += np.sin(walk_angle)
            cx += np.cos(walk_angle)
            if cy < 1 or cy >= size - 1 or cx < 1 or cx >= size - 1:
                break
            _draw_vessel_segment(canvas, cy, cx, thick, amp, size)

    # Feeding vessels
    tree = _make_vessel_tree(
        rng, size, n_branches=25, thickness_range=(1.5, 3.5),
        amplitude_range=(0.4, 0.8))
    canvas = np.maximum(canvas, tree)
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def _make_skin_vessels(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Superficial skin vessel phantom: thin parallel vessels in layers."""
    canvas = np.zeros((size, size), dtype=np.float64)

    # Dermal background
    depth_profile = np.linspace(0.0, 1.0, size)[:, None]
    canvas += 0.05 * np.exp(-3.0 * depth_profile)

    def _draw_horizontal_vessel(vy, thick, amp):
        base_angle = rng.uniform(-0.15, 0.15)
        cx = float(rng.integers(0, size // 6))
        cy = float(vy)
        while cx < size - 1:
            base_angle += rng.normal(0.0, 0.02)
            cx += 1.0
            cy += np.sin(base_angle) * 0.5
            if cy < 0 or cy >= size:
                break
            _draw_vessel_segment(canvas, cy, cx, thick, amp, size)

    # Superficial plexus
    for _ in range(int(rng.integers(6, 14))):
        _draw_horizontal_vessel(
            rng.integers(size // 10, size // 3),
            rng.uniform(0.6, 1.5),
            rng.uniform(0.4, 0.9))

    # Deep plexus
    for _ in range(int(rng.integers(4, 8))):
        _draw_horizontal_vessel(
            rng.integers(size // 3, 2 * size // 3),
            rng.uniform(1.5, 3.0),
            rng.uniform(0.5, 0.85))

    # Connecting perforators
    for _ in range(int(rng.integers(3, 7))):
        px_val = rng.integers(size // 6, 5 * size // 6)
        py_start = rng.integers(size // 8, size // 3)
        py_end = rng.integers(size // 3, 2 * size // 3)
        thick = rng.uniform(0.8, 1.8)
        amp = rng.uniform(0.4, 0.75)
        for py in range(py_start, py_end):
            _draw_vessel_segment(canvas, float(py), float(px_val),
                                 thick, amp, size)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def _make_mixed_phantom(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Mixed phantom: vessel tree + point absorbers + diffuse background."""
    canvas = _make_vessel_tree(
        rng, size, n_branches=rng.integers(30, 50),
        thickness_range=(0.8, 3.0), amplitude_range=(0.4, 0.9))

    # Point absorbers
    for _ in range(int(rng.integers(5, 20))):
        py = rng.integers(15, size - 15)
        px_val = rng.integers(15, size - 15)
        pr = rng.uniform(1.0, 3.5)
        amp = rng.uniform(0.5, 1.0)
        yy, xx = np.ogrid[:size, :size]
        dist_sq = (yy - py) ** 2 + (xx - px_val) ** 2
        canvas += amp * np.exp(-dist_sq / (2.0 * pr ** 2))

    # Faint diffuse background
    bg = gaussian_filter(rng.standard_normal((size, size)), sigma=20.0)
    canvas += 0.03 * (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Scene name tables
# ============================================================================

PUBLIC_SCENE_NAMES = [
    "vessel_tree_01", "vessel_tree_02", "vessel_tree_03", "vessel_tree_04",
    "tumor_vasc_01", "tumor_vasc_02", "tumor_vasc_03", "tumor_vasc_04",
    "skin_vessel_01", "skin_vessel_02", "skin_vessel_03", "skin_vessel_04",
]

DEV_SCENE_NAMES = [f"dev_vascular_{i:02d}" for i in range(20)]
HIDDEN_SCENE_NAMES = [f"hidden_adversarial_{i:02d}" for i in range(20)]


# ============================================================================
# Phantom generation per tier
# ============================================================================

def generate_public_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """12 public phantoms: 4 vessel trees + 4 tumour + 4 skin vessels."""
    phantoms = []
    for i in range(4):
        x = _make_vessel_tree(rng, IMAGE_SIZE, n_branches=rng.integers(40, 70))
        phantoms.append((PUBLIC_SCENE_NAMES[i], x))
    for i in range(4):
        x = _make_tumor_vasculature(rng, IMAGE_SIZE)
        phantoms.append((PUBLIC_SCENE_NAMES[4 + i], x))
    for i in range(4):
        x = _make_skin_vessels(rng, IMAGE_SIZE)
        phantoms.append((PUBLIC_SCENE_NAMES[8 + i], x))
    return phantoms


def generate_dev_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """20 dev phantoms: mixed types + augmentation."""
    generators = [
        lambda r: _make_vessel_tree(r, IMAGE_SIZE,
                                     n_branches=r.integers(30, 60)),
        lambda r: _make_tumor_vasculature(r, IMAGE_SIZE),
        lambda r: _make_skin_vessels(r, IMAGE_SIZE),
        lambda r: _make_mixed_phantom(r, IMAGE_SIZE),
    ]
    phantoms = []
    for i in range(20):
        gen = generators[i % len(generators)]
        x = gen(rng)
        k = rng.integers(0, 4)
        if k > 0:
            x = np.rot90(x, k).copy()
        if rng.random() < 0.5:
            x = np.fliplr(x).copy()
        if rng.random() < 0.3:
            x = np.flipud(x).copy()
        phantoms.append((DEV_SCENE_NAMES[i], x))
    return phantoms


def generate_hidden_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """20 hidden phantoms: adversarial, complex anatomy."""
    phantoms = []
    for i in range(20):
        if i < 6:
            x = _make_vessel_tree(rng, IMAGE_SIZE, n_branches=rng.integers(60, 90),
                                   thickness_range=(0.5, 2.0),
                                   amplitude_range=(0.3, 1.0))
        elif i < 12:
            x = _make_tumor_vasculature(rng, IMAGE_SIZE)
            for _ in range(int(rng.integers(10, 30))):
                py = rng.integers(10, IMAGE_SIZE - 10)
                px_val = rng.integers(10, IMAGE_SIZE - 10)
                pr = rng.uniform(0.5, 2.0)
                amp = rng.uniform(0.3, 0.8)
                yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                d2 = (yy - py) ** 2 + (xx - px_val) ** 2
                x = np.maximum(x, (amp * np.exp(
                    -d2 / (2.0 * pr ** 2))).astype(np.float32))
        elif i < 16:
            x = _make_skin_vessels(rng, IMAGE_SIZE)
            ly, lx = rng.integers(20, IMAGE_SIZE - 20, size=2)
            lr = rng.uniform(5, 15)
            yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            d2 = (yy - ly) ** 2 + (xx - lx) ** 2
            x = np.maximum(x, (0.9 * np.exp(
                -d2 / (2.0 * lr ** 2))).astype(np.float32))
        else:
            x = _make_mixed_phantom(rng, IMAGE_SIZE)

        k = rng.integers(0, 4)
        if k > 0:
            x = np.rot90(x, k).copy()
        if rng.random() < 0.7:
            x = np.fliplr(x).copy()
        if rng.random() < 0.5:
            x = np.flipud(x).copy()
        gamma = rng.uniform(0.7, 1.4)
        x = np.clip(x ** gamma, 0.0, 1.0).astype(np.float32)

        phantoms.append((HIDDEN_SCENE_NAMES[i], x))
    return phantoms


# ============================================================================
# Forward model: circular-arc transducer array (on-the-fly, no dense matrix)
# ============================================================================

def _get_detector_positions(
    n_det: int,
    angular_coverage_deg: float,
    detector_radius_px: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (det_y, det_x) arrays for transducers on a circular arc."""
    half = image_size / 2.0
    arc_start = np.deg2rad((360.0 - angular_coverage_deg) / 2.0)
    arc_end = np.deg2rad((360.0 + angular_coverage_deg) / 2.0)
    det_angles = np.linspace(arc_start, arc_end, n_det, endpoint=False)
    det_y = half + detector_radius_px * np.sin(det_angles)
    det_x = half + detector_radius_px * np.cos(det_angles)
    return det_y.astype(np.float64), det_x.astype(np.float64)


def _get_time_radii(
    n_time: int,
    detector_radius_px: float,
) -> np.ndarray:
    """Time-sample radii in pixels (distance = c * t mapped to pixels)."""
    max_travel_px = 2.0 * detector_radius_px * 1.15
    dt_px = max_travel_px / n_time
    return (np.arange(n_time) + 0.5) * dt_px


def _forward_op(
    x: np.ndarray,
    det_y: np.ndarray,
    det_x: np.ndarray,
    time_radii: np.ndarray,
    n_arc: int = 120,
) -> np.ndarray:
    """Apply the PA forward operator: for each (detector, time) pair,
    integrate x along a circular arc of radius time_radii[t] centred
    at the detector.

    Returns y of shape (n_det, n_time).
    """
    n_det = len(det_y)
    n_time = len(time_radii)
    size = x.shape[0]
    x64 = x.astype(np.float64)

    arc_thetas = np.linspace(0, 2.0 * np.pi, n_arc, endpoint=False)
    cos_arc = np.cos(arc_thetas)  # (n_arc,)
    sin_arc = np.sin(arc_thetas)  # (n_arc,)

    y = np.zeros((n_det, n_time), dtype=np.float64)

    for di in range(n_det):
        dy, dx = det_y[di], det_x[di]
        for ti in range(n_time):
            r_t = time_radii[ti]
            if r_t < 0.5:
                continue
            # Sample points on the integration circle
            sy = dy + r_t * sin_arc
            sx = dx + r_t * cos_arc

            # Nearest-neighbour sampling (fast)
            iy = np.round(sy).astype(np.intp)
            ix = np.round(sx).astype(np.intp)
            valid = (iy >= 0) & (iy < size) & (ix >= 0) & (ix < size)
            if valid.any():
                y[di, ti] = x64[iy[valid], ix[valid]].sum() / n_arc

    return y.astype(np.float32)


def _adjoint_op(
    y: np.ndarray,
    det_y: np.ndarray,
    det_x: np.ndarray,
    time_radii: np.ndarray,
    image_size: int,
    n_arc: int = 120,
) -> np.ndarray:
    """Adjoint (back-projection) of the PA forward operator.

    For each (detector, time) pair, spread y[d,t] back along the
    circular arc of radius time_radii[t] centred at detector d.

    Returns x_bp of shape (image_size, image_size).
    """
    n_det = y.shape[0]
    n_time = y.shape[1]

    arc_thetas = np.linspace(0, 2.0 * np.pi, n_arc, endpoint=False)
    cos_arc = np.cos(arc_thetas)
    sin_arc = np.sin(arc_thetas)

    x_bp = np.zeros((image_size, image_size), dtype=np.float64)

    for di in range(n_det):
        dy, dx = det_y[di], det_x[di]
        for ti in range(n_time):
            val = float(y[di, ti])
            if abs(val) < 1e-12:
                continue
            r_t = time_radii[ti]
            if r_t < 0.5:
                continue
            sy = dy + r_t * sin_arc
            sx = dx + r_t * cos_arc
            iy = np.round(sy).astype(np.intp)
            ix = np.round(sx).astype(np.intp)
            valid = (iy >= 0) & (iy < image_size) & (ix >= 0) & (ix < image_size)
            weight = val / n_arc
            np.add.at(x_bp, (iy[valid], ix[valid]), weight)

    return x_bp.astype(np.float32)


def _apply_bandwidth_filter(
    y: np.ndarray,
    bandwidth_MHz: float,
    sampling_rate_MHz: float = 40.0,
) -> np.ndarray:
    """Low-pass filter in the time direction to simulate finite transducer
    bandwidth."""
    n_det, n_time = y.shape
    Y = np.fft.rfft(y.astype(np.float64), axis=1)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate_MHz)
    order = 4
    H = 1.0 / (1.0 + (freqs / max(bandwidth_MHz, 0.5)) ** (2 * order))
    Y *= H[None, :]
    y_filt = np.fft.irfft(Y, n=n_time, axis=1)
    return y_filt.astype(np.float32)


def forward_model(
    x_true: np.ndarray,
    n_det: int,
    n_time: int,
    detector_radius_px: float,
    angular_coverage_deg: float,
    speed_of_sound_error_pct: float,
    bandwidth_limit_MHz: float,
    noise_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete photoacoustic forward model.

    Returns:
        y_measured : (n_det, n_time) -- measured signals with mismatch + noise
        y_ideal    : (n_det, n_time) -- ideal signals (nominal geometry,
                     no mismatch, no noise)
    """
    N = x_true.shape[0]

    # Ideal geometry (nominal coverage, nominal SoS)
    det_y_ideal, det_x_ideal = _get_detector_positions(
        n_det, ANGULAR_COVERAGE_DEG, detector_radius_px, N)
    time_radii_ideal = _get_time_radii(n_time, detector_radius_px)

    y_ideal = _forward_op(x_true, det_y_ideal, det_x_ideal,
                          time_radii_ideal)

    # Mismatched geometry (actual coverage, perturbed SoS)
    det_y_mis, det_x_mis = _get_detector_positions(
        n_det, angular_coverage_deg, detector_radius_px, N)
    # SoS error stretches/compresses the time-to-radius mapping
    sos_factor = 1.0 + speed_of_sound_error_pct / 100.0
    time_radii_mis = time_radii_ideal * sos_factor

    y_mismatch = _forward_op(x_true, det_y_mis, det_x_mis,
                             time_radii_mis)

    # Bandwidth limiting
    y_mismatch = _apply_bandwidth_filter(y_mismatch, bandwidth_limit_MHz)

    # Additive Gaussian noise
    sig_peak = np.abs(y_mismatch).max() + 1e-10
    noise = rng.normal(0.0, noise_level * sig_peak,
                       y_mismatch.shape).astype(np.float32)
    y_measured = y_mismatch + noise

    return y_measured, y_ideal


# ============================================================================
# CPU Baseline: Universal Back-Projection (UBP)
# ============================================================================

def reconstruct_ubp(
    y_measured: np.ndarray,
    n_det: int,
    n_time: int,
    detector_radius_px: float,
    angular_coverage_deg: float,
    image_size: int,
) -> np.ndarray:
    """Universal back-projection: adjoint of the forward operator.

    Uses the ideal (nominal) geometry for back-projection.
    """
    det_y, det_x = _get_detector_positions(
        n_det, angular_coverage_deg, detector_radius_px, image_size)
    time_radii = _get_time_radii(n_time, detector_radius_px)

    y_2d = y_measured.reshape(n_det, n_time) if y_measured.ndim == 1 \
        else y_measured

    x_bp = _adjoint_op(y_2d, det_y, det_x, time_radii, image_size)
    x_bp = np.clip(x_bp, 0.0, None)
    x_max = x_bp.max()
    if x_max > 1e-10:
        x_bp /= x_max
    return x_bp.astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Simple global SSIM (no skimage dependency)."""
    g = gt.astype(np.float64)
    r = recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_g, mu_r = g.mean(), r.mean()
    var_g, var_r = g.var(), r.var()
    cov = float(np.mean((g - mu_g) * (r - mu_r)))
    return float(((2 * mu_g * mu_r + c1) * (2 * cov + c2)) /
                 ((mu_g ** 2 + mu_r ** 2 + c1) * (var_g + var_r + c2)))


# ============================================================================
# Image helpers
# ============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _save_png(arr: np.ndarray, path: Path) -> None:
    normed = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(normed, "L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img.save(str(path))


def _save_sinogram_png(y: np.ndarray, path: Path) -> None:
    """Save (n_det, n_time) sinogram as a grayscale PNG."""
    _save_png(y, path)


def _save_overview(
    x_true: np.ndarray,
    y_meas: np.ndarray,
    recon: np.ndarray,
    path: Path,
) -> None:
    """Save 1x3 overview: GT | Sinogram | Recon."""
    th, tw = 128, 128

    def _r(a: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS))

    ov = np.hstack([_r(x_true), _r(y_meas), _r(recon)])
    Image.fromarray(ov, "L").save(str(path))


# ============================================================================
# Mismatch sampling
# ============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        lo, hi = v["min"], v["max"]
        result[k] = float(rng.uniform(lo, hi))
    return result


# ============================================================================
# H_ideal geometry descriptor (compact representation)
# ============================================================================

def _make_h_ideal_descriptor(
    n_det: int,
    n_time: int,
    detector_radius_px: float,
    image_size: int,
) -> dict:
    """Create a JSON-serializable descriptor of the ideal system geometry.

    This replaces the full dense system matrix (which would be
    n_det*n_time x image_size^2 = ~500 MB per sample) with a compact
    parametric description that allows exact reconstruction of the
    forward/adjoint operator.
    """
    det_y, det_x = _get_detector_positions(
        n_det, ANGULAR_COVERAGE_DEG, detector_radius_px, image_size)
    time_radii = _get_time_radii(n_time, detector_radius_px)
    return {
        "type": "circular_arc_2d",
        "n_det": n_det,
        "n_time": n_time,
        "detector_radius_px": detector_radius_px,
        "angular_coverage_deg": ANGULAR_COVERAGE_DEG,
        "speed_of_sound_m_s": SPEED_OF_SOUND,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "image_size": image_size,
        "det_y": det_y.tolist(),
        "det_x": det_x.tolist(),
        "time_radii_px": time_radii.tolist(),
        "n_arc_samples": 120,
        "note": ("To apply: for each (det_i, time_j), integrate image along "
                 "circle of radius time_radii_px[j] centred at "
                 "(det_y[i], det_x[i]). Adjoint = back-project."),
    }


# ============================================================================
# Tier generator
# ============================================================================

def generate_tier(
    tier: str,
    phantoms: list[tuple[str, np.ndarray]],
    base_seed: int,
) -> None:
    """Generate one tier of the photoacoustic benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"photoacoustic_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    rows, true_specs = [], {}
    psnr_list, ssim_list = [], []

    n_det = N_DET_DEFAULT
    n_time = N_TIME_DEFAULT

    h_ideal_desc = _make_h_ideal_descriptor(
        n_det, n_time, DETECTOR_RADIUS_PX, IMAGE_SIZE)

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Photoacoustic Tomography benchmark -- {tier} tier "
            f"(circular transducer array, limited-view)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "fov_mm": IMAGE_SIZE * PIXEL_SIZE_MM,
            "n_det": n_det,
            "n_time": n_time,
            "detector_radius_px": DETECTOR_RADIUS_PX,
            "speed_of_sound_m_s": SPEED_OF_SOUND,
            "nominal_angular_coverage_deg": ANGULAR_COVERAGE_DEG,
            "nominal_bandwidth_MHz": BANDWIDTH_MHZ,
        })
        f.attrs["H_ideal"] = json.dumps(h_ideal_desc)

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            print(f"  [{tier}] {key} {scene_name} -- forward model...",
                  end="", flush=True)

            y_measured, y_ideal = forward_model(
                x_true,
                n_det=n_det,
                n_time=n_time,
                detector_radius_px=DETECTOR_RADIUS_PX,
                angular_coverage_deg=mis["angular_coverage_deg"],
                speed_of_sound_error_pct=mis["speed_of_sound_error_pct"],
                bandwidth_limit_MHz=mis["bandwidth_limit_MHz"],
                noise_level=mis["noise_level"],
                rng=rng,
            )

            # CPU baseline reconstruction (UBP with ideal geometry)
            recon = reconstruct_ubp(
                y_measured, n_det, n_time, DETECTOR_RADIUS_PX,
                ANGULAR_COVERAGE_DEG, IMAGE_SIZE)
            gt_norm = _norm(x_true)
            recon_norm = _norm(recon)
            psnr = compute_psnr(gt_norm, recon_norm)
            ssim = compute_ssim(gt_norm, recon_norm)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y_measured, compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "n_det": n_det,
                "n_time": n_time,
                "y_shape": list(y_measured.shape),
                "ubp_psnr_dB": round(psnr, 2),
                "ubp_ssim": round(ssim, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps({**mis, "scene": scene_name})

            # Per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_sinogram_png(y_measured, sample_dir / "sinogram_measured.png")
            _save_sinogram_png(y_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(recon, sample_dir / "reconstruction.png")
            _save_overview(x_true, y_measured, recon,
                           sample_dir / "overview.png")

            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "ubp_psnr_dB": round(psnr, 2),
                    "ubp_ssim": round(ssim, 4),
                }, sf, indent=2)

            rows.append((key, scene_name, mis, psnr, ssim))
            print(f"  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  "
                  f"cov={mis['angular_coverage_deg']:.0f}deg  "
                  f"noise={mis['noise_level']:.3f}")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows)
    print(f"  [{tier}] HDF5 -> {h5_path.name}  "
          f"avg PSNR={np.mean(psnr_list):.2f} dB  "
          f"avg SSIM={np.mean(ssim_list):.4f}")


# ============================================================================
# README writers
# ============================================================================

def _write_tier_readme(tier: str, tier_dir: Path, rows: list) -> None:
    spec = SPEC[tier]
    if tier == "public":
        access = "Full (GT + true spec + ideal sinogram)"
        source = ("Synthetic vascular phantoms: vessel trees, tumour "
                  "vasculature, skin vessels (12 samples)")
    elif tier == "dev":
        access = "Blind (measured signals + spec ranges only)"
        source = "Augmented synthetic phantoms -- mixed types (20 samples)"
    else:
        access = "Server-only"
        source = ("Adversarial synthetic phantoms -- dense vasculature, "
                  "bright lesions, intensity perturbations (20 samples)")

    param_desc = {
        "speed_of_sound_error_pct": "Acoustic SoS error",
        "angular_coverage_deg": "Transducer angular coverage",
        "bandwidth_limit_MHz": "Transducer bandwidth cutoff",
        "noise_level": "Additive noise sigma / peak",
    }

    lines = [
        f"# Photoacoustic Tomography {tier.capitalize()} Tier\n\n",
        f"**Source:** {source}\n\n",
        f"**Access:** {access}\n\n",
        "## Mismatch Parameters\n\n",
        "| Parameter | Description | Range |\n",
        "|-----------|-------------|-------|\n",
    ]
    for k, v in spec.items():
        lo, hi, u = v["min"], v["max"], v.get("unit", "")
        lines.append(
            f"| `{k}` | {param_desc.get(k, k)} | [{lo}, {hi}] {u} |\n")

    lines += [
        "\n## Samples\n\n",
        "| Sample | Scene | SoS Err (%) | Coverage (deg) | BW (MHz) | "
        "Noise | PSNR (dB) | SSIM |\n",
        "|--------|-------|-------------|----------------|----------|"
        "-------|-----------|------|\n",
    ]
    for key, scene, mis, psnr, ssim in rows:
        lines.append(
            f"| {key} | {scene}"
            f" | {mis['speed_of_sound_error_pct']:.1f}"
            f" | {mis['angular_coverage_deg']:.0f}"
            f" | {mis['bandwidth_limit_MHz']:.1f}"
            f" | {mis['noise_level']:.3f}"
            f" | {psnr:.2f}"
            f" | {ssim:.4f} |\n")

    lines += [
        "\n## HDF5 Datasets per Sample\n\n",
        "| Key | Shape | Dtype | Description |\n",
        "|-----|-------|-------|-------------|\n",
        "| `x_true` | (256, 256) | float32 | "
        "Ground-truth initial pressure distribution |\n",
        f"| `y` | ({N_DET_DEFAULT}, {N_TIME_DEFAULT}) | float32 | "
        "Measured time-domain pressure signals |\n",
        "\n`H_ideal` is stored as a JSON geometry descriptor in file-level "
        "attributes (not per-sample) to avoid ~500 MB dense matrices.\n",
    ]

    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme() -> None:
    txt = f"""# Photoacoustic Tomography Benchmark Dataset

## Overview

Photoacoustic tomography (PAT) benchmark with circular transducer array
geometry, limited-view artifacts, and realistic vascular phantoms.

A pulsed laser causes thermoelastic expansion in absorbing tissue
structures; the resulting ultrasound waves are detected by a transducer
array and reconstructed into a 2-D initial-pressure map.

## Forward Model

```
y(d, t) = integral of p_0(r) along circle of radius c*t
          centred at detector d, plus noise

where:
  p_0(r)  -- (256, 256) initial pressure distribution
  d       -- transducer element index (0..{N_DET_DEFAULT - 1})
  t       -- time sample index (0..{N_TIME_DEFAULT - 1})
  c       -- speed of sound ({SPEED_OF_SOUND} m/s nominal)
```

## Transducer Geometry

| Parameter | Value |
|-----------|-------|
| Image size | {IMAGE_SIZE} x {IMAGE_SIZE} px |
| Pixel size | {PIXEL_SIZE_MM} mm |
| FOV | {IMAGE_SIZE * PIXEL_SIZE_MM:.1f} mm |
| Detector radius | {DETECTOR_RADIUS_PX:.0f} px ({DETECTOR_RADIUS_PX * PIXEL_SIZE_MM:.0f} mm) |
| N detectors | {N_DET_DEFAULT} |
| N time samples | {N_TIME_DEFAULT} |
| Speed of sound | {SPEED_OF_SOUND:.0f} m/s (nominal) |
| Angular coverage | {ANGULAR_COVERAGE_DEG:.0f} deg (nominal, limited-view) |
| Bandwidth | {BANDWIDTH_MHZ:.0f} MHz (nominal) |

## Mismatch Parameters (ThetaSpace)

| Knob | Description | Public | Dev | Hidden |
|------|-------------|--------|-----|--------|
| `speed_of_sound_error_pct` | Acoustic SoS error | +-2% | +-4% | +-6% |
| `angular_coverage_deg` | Transducer arc | 200-300 | 160-280 | 120-260 |
| `bandwidth_limit_MHz` | Low-pass cutoff | 6-12 | 4-10 | 3-8 |
| `noise_level` | sigma / peak | 0.01-0.04 | 0.02-0.08 | 0.04-0.15 |

## Phantom Types

| Type | Description | Tier |
|------|-------------|------|
| Vessel tree | Branching vascular network | Public, Dev |
| Tumor vasculature | Dense tortuous vessels near tumour core | Public, Dev, Hidden |
| Skin vessels | Superficial / deep plexus layers | Public, Dev, Hidden |
| Mixed | Vessels + point absorbers + diffuse bg | Dev, Hidden |

## CPU Reconstruction

Universal Back-Projection (UBP): adjoint of the forward operator (time-
reversal summation over the transducer arc).
Expected ~20-26 dB depending on limited-view and noise severity.

## H_ideal

The ideal system matrix is stored as a JSON geometry descriptor in the
HDF5 file-level attribute `H_ideal`, containing detector positions,
time-sample radii, and reconstruction parameters.  This avoids storing
a dense matrix of ~500 MB per sample.

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## Dataset Structure

```
photoacoustic/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (4 vessel + 4 tumour + 4 skin)
|   +-- photoacoustic_challenge_public.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- dev/       20 samples (augmented, medium mismatch)
|   +-- photoacoustic_challenge_dev.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- hidden/    20 samples (adversarial, wide mismatch)
    +-- photoacoustic_challenge_hidden.h5
    +-- spec.json / true_spec.json
    +-- images/sample_XX_*/
```

## References

- Xu & Wang (2005) Universal Back-Projection, IEEE Trans. Med. Imaging 24:1208-1221.
- Treeby & Cox (2010) k-Wave, J. Biomedical Optics 15:021314.
- Arridge et al. (2016) Model-based TV, Inverse Problems 32:115012.
- Antholzer et al. (2019) PAT-Net, Photoacoustics 14:1-9.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ============================================================================
# Gallery image generation
# ============================================================================

def generate_gallery_images(
    phantoms: list[tuple[str, np.ndarray]],
    gallery_dir: Path,
    n_scenes: int = 4,
    seed: int = 5555,
) -> None:
    """Generate gallery images for the platform benchmark page.

    Each scene: gt.png, measurement_I.png, measurement_II.png,
                recon_I.png, recon_II.png, recon_III.png
    All 256x256 grayscale PNGs.
    """
    rng = np.random.default_rng(seed)

    # Diverse phantoms: vessel_tree, tumour, skin, vessel_tree
    gallery_indices = [0, 4, 8, 1]
    n_det = N_DET_DEFAULT
    n_time = N_TIME_DEFAULT

    for si, pidx in enumerate(gallery_indices[:n_scenes]):
        if pidx >= len(phantoms):
            pidx = si % len(phantoms)
        scene_name, x_true = phantoms[pidx]
        scene_dir = gallery_dir / f"scene_{si:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        _save_png(x_true, scene_dir / "gt.png")

        # Measurement I: mild conditions (wide coverage, low noise)
        y_m1, _ = forward_model(
            x_true, n_det, n_time, DETECTOR_RADIUS_PX,
            angular_coverage_deg=280.0,
            speed_of_sound_error_pct=1.0,
            bandwidth_limit_MHz=10.0,
            noise_level=0.02,
            rng=rng,
        )
        _save_sinogram_png(y_m1, scene_dir / "measurement_I.png")

        # Measurement II: harsh conditions (narrow coverage, high noise)
        y_m2, _ = forward_model(
            x_true, n_det, n_time, DETECTOR_RADIUS_PX,
            angular_coverage_deg=160.0,
            speed_of_sound_error_pct=4.0,
            bandwidth_limit_MHz=5.0,
            noise_level=0.08,
            rng=rng,
        )
        _save_sinogram_png(y_m2, scene_dir / "measurement_II.png")

        # Recon I: UBP from mild
        r1 = reconstruct_ubp(y_m1, n_det, n_time, DETECTOR_RADIUS_PX,
                             ANGULAR_COVERAGE_DEG, IMAGE_SIZE)
        _save_png(r1, scene_dir / "recon_I.png")

        # Recon II: UBP from harsh
        r2 = reconstruct_ubp(y_m2, n_det, n_time, DETECTOR_RADIUS_PX,
                             ANGULAR_COVERAGE_DEG, IMAGE_SIZE)
        _save_png(r2, scene_dir / "recon_II.png")

        # Recon III: UBP from harsh with mismatched coverage
        r3 = reconstruct_ubp(y_m2, n_det, n_time, DETECTOR_RADIUS_PX,
                             280.0, IMAGE_SIZE)  # wrong coverage assumption
        _save_png(r3, scene_dir / "recon_III.png")

        print(f"  Gallery scene_{si:02d} ({scene_name}): 6 images saved")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Photoacoustic Tomography Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Output: {BENCHMARK_DIR}\n")

    # -- Public tier (12 samples) -----------------------------------------
    print("Generating public tier (12 samples)...")
    rng_pub = np.random.default_rng(0)
    public_phantoms = generate_public_phantoms(rng_pub)
    generate_tier("public", public_phantoms, base_seed=0)

    # -- Dev tier (20 samples) --------------------------------------------
    print("\nGenerating dev tier (20 samples, augmented)...")
    rng_dev = np.random.default_rng(10000)
    dev_phantoms = generate_dev_phantoms(rng_dev)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # -- Hidden tier (20 samples) -----------------------------------------
    print("\nGenerating hidden tier (20 samples, adversarial)...")
    rng_hid = np.random.default_rng(20000)
    hidden_phantoms = generate_hidden_phantoms(rng_hid)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # -- Gallery images ---------------------------------------------------
    gallery_dir = (Path(__file__).resolve().parent.parent.parent.parent /
                   "platform" / "pwm_platform" / "static" / "img" /
                   "benchmark_gallery" / "photoacoustic")
    print(f"\nGenerating gallery images at {gallery_dir}...")
    generate_gallery_images(public_phantoms, gallery_dir, n_scenes=4)

    # -- Top-level README -------------------------------------------------
    _write_top_readme()

    print(f"\n{'=' * 60}")
    print(f"Done -- Photoacoustic benchmark ready at {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
