#!/usr/bin/env python3
"""
Generate photoacoustic imaging benchmark dataset.

Physics:
  - Pulsed laser excites tissue -> acoustic waves detected by transducers
  - x_true: optical absorption map / initial pressure distribution, (256,256) float32, [0,1]
  - Forward: simplified back-projection from ring transducer array
      * 128 transducer positions on circle of radius 100px around 128x128 image centre
      * Time-of-flight: y shape (128, 200) = (n_transducers, n_time_steps)
      * Add Gaussian noise to y
  - H_ideal: transducer positions array (128, 2) -- columns [row, col]
  - reconstruction_baseline: delay-and-sum back projection
  - Mismatch: speed of sound error, transducer position error, noise

NOTE: y shape is (128, 200) NOT (256, 256).

Tiers:
  public:  12 samples
  dev:     20 samples
  hidden:  20 samples

Output:
  datasets/benchmark/photoacoustic/{tier}/photoacoustic_challenge_{tier}.h5
  datasets/benchmark/photoacoustic/{tier}/spec.json
  datasets/benchmark/photoacoustic/{tier}/true_spec.json
  datasets/benchmark/photoacoustic/{tier}/images/sample_NN/...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "datasets" / "benchmark" / "photoacoustic"

IMAGE_SIZE = 256
N_TRANSDUCERS = 128
N_TIME_STEPS = 200
TRANSDUCER_RADIUS_PX = 100.0        # pixels, around image centre (128, 128)
IMAGE_CENTRE = IMAGE_SIZE / 2.0     # 128.0
SPEED_OF_SOUND = 1540.0             # m/s nominal (soft tissue)
PIXEL_SIZE_MM = 0.10                # 100 um/pixel => FOV = 25.6 mm

# Maximum detectable radius: 2*R * 1.1 (to cover far corners)
MAX_TRAVEL_PX = 2.0 * TRANSDUCER_RADIUS_PX * 1.15


# ── Mismatch spec ranges ────────────────────────────────────────────────────

SPEC = {
    "public": {
        "speed_of_sound_error_pct":  {"min": -2.0,  "max":  2.0,  "unit": "%"},
        "transducer_position_error": {"min":  0.0,  "max":  2.0,  "unit": "pixels"},
        "noise_sigma":               {"min":  0.01, "max":  0.04, "unit": "fraction of peak"},
    },
    "dev": {
        "speed_of_sound_error_pct":  {"min": -4.0,  "max":  4.0,  "unit": "%"},
        "transducer_position_error": {"min":  0.0,  "max":  4.0,  "unit": "pixels"},
        "noise_sigma":               {"min":  0.02, "max":  0.08, "unit": "fraction of peak"},
    },
    "hidden": {
        "speed_of_sound_error_pct":  {"min": -6.0,  "max":  6.0,  "unit": "%"},
        "transducer_position_error": {"min":  0.0,  "max":  6.0,  "unit": "pixels"},
        "noise_sigma":               {"min":  0.04, "max":  0.15, "unit": "fraction of peak"},
    },
}


# ── Geometry helpers ─────────────────────────────────────────────────────────

def get_transducer_positions(
    n: int = N_TRANSDUCERS,
    radius_px: float = TRANSDUCER_RADIUS_PX,
    centre: float = IMAGE_CENTRE,
    position_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns (n, 2) array of [row, col] transducer positions on a full circle.
    Transducers are equally spaced at 360/n degrees.
    Optional Gaussian position noise in pixels.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = centre + radius_px * np.sin(angles)
    cols = centre + radius_px * np.cos(angles)
    positions = np.column_stack([rows, cols])  # (n, 2)

    if position_noise > 0.0 and rng is not None:
        noise = rng.normal(0.0, position_noise, positions.shape)
        positions = positions + noise

    return positions.astype(np.float32)


def get_time_radii(
    n_time: int = N_TIME_STEPS,
    max_travel_px: float = MAX_TRAVEL_PX,
) -> np.ndarray:
    """
    Returns (n_time,) array of radii in pixels corresponding to time samples.
    Linear spacing from 0 to max_travel_px.
    """
    dt_px = max_travel_px / n_time
    return (np.arange(n_time) + 0.5) * dt_px


# ── Forward model: circular arc integration ──────────────────────────────────

def forward_operator(
    x: np.ndarray,
    transducer_positions: np.ndarray,
    time_radii: np.ndarray,
    n_arc: int = 120,
) -> np.ndarray:
    """
    Photoacoustic forward operator.

    For each transducer d and time step t, integrate x along a circle of
    radius time_radii[t] centred at transducer_positions[d].

    Args:
        x                   : (H, W) initial pressure / absorption map
        transducer_positions: (n_det, 2) [row, col] in pixels
        time_radii          : (n_time,) radii in pixels
        n_arc               : number of arc sample points per circle

    Returns:
        y : (n_det, n_time) float32
    """
    n_det = len(transducer_positions)
    n_time = len(time_radii)
    H, W = x.shape
    x64 = x.astype(np.float64)

    arc_thetas = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
    sin_arc = np.sin(arc_thetas)
    cos_arc = np.cos(arc_thetas)

    y = np.zeros((n_det, n_time), dtype=np.float64)

    for di in range(n_det):
        dr, dc = float(transducer_positions[di, 0]), float(transducer_positions[di, 1])
        for ti in range(n_time):
            r_t = float(time_radii[ti])
            if r_t < 0.5:
                continue
            # Arc sample points
            sample_rows = dr + r_t * sin_arc
            sample_cols = dc + r_t * cos_arc
            ir = np.round(sample_rows).astype(np.intp)
            ic = np.round(sample_cols).astype(np.intp)
            valid = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
            if valid.any():
                y[di, ti] = x64[ir[valid], ic[valid]].sum() / n_arc

    return y.astype(np.float32)


def backproject_operator(
    y: np.ndarray,
    transducer_positions: np.ndarray,
    time_radii: np.ndarray,
    image_size: int,
    n_arc: int = 120,
) -> np.ndarray:
    """
    Delay-and-sum back projection (adjoint of forward_operator).

    For each (transducer, time) pair, spread y[d,t] back along the
    arc of radius time_radii[t] centred at transducer d.

    Args:
        y                   : (n_det, n_time) float32
        transducer_positions: (n_det, 2) [row, col] in pixels
        time_radii          : (n_time,) radii in pixels
        image_size          : output image is (image_size, image_size)
        n_arc               : arc sample density

    Returns:
        x_bp : (image_size, image_size) float32
    """
    n_det, n_time = y.shape
    arc_thetas = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
    sin_arc = np.sin(arc_thetas)
    cos_arc = np.cos(arc_thetas)

    x_bp = np.zeros((image_size, image_size), dtype=np.float64)

    for di in range(n_det):
        dr, dc = float(transducer_positions[di, 0]), float(transducer_positions[di, 1])
        for ti in range(n_time):
            val = float(y[di, ti])
            if abs(val) < 1e-12:
                continue
            r_t = float(time_radii[ti])
            if r_t < 0.5:
                continue
            sample_rows = dr + r_t * sin_arc
            sample_cols = dc + r_t * cos_arc
            ir = np.round(sample_rows).astype(np.intp)
            ic = np.round(sample_cols).astype(np.intp)
            valid = (ir >= 0) & (ir < image_size) & (ic >= 0) & (ic < image_size)
            weight = val / n_arc
            np.add.at(x_bp, (ir[valid], ic[valid]), weight)

    x_bp = np.clip(x_bp, 0, None)
    x_max = float(x_bp.max())
    if x_max > 1e-10:
        x_bp /= x_max
    return x_bp.astype(np.float32)


def full_forward_model(
    x_true: np.ndarray,
    speed_of_sound_error_pct: float,
    transducer_position_error: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete PA forward model with mismatch.

    Speed-of-sound error stretches/compresses the time axis (scales radii).
    Transducer position error adds Gaussian noise to detector locations.
    Gaussian noise is added to y.

    Returns:
        y         : (N_TRANSDUCERS, N_TIME_STEPS) float32 noisy measurement
        y_ideal   : (N_TRANSDUCERS, N_TIME_STEPS) float32 ideal (no noise, nominal SoS)
    """
    # Ideal geometry (nominal, no position error)
    positions_ideal = get_transducer_positions(N_TRANSDUCERS, TRANSDUCER_RADIUS_PX,
                                               IMAGE_CENTRE)
    radii_ideal = get_time_radii(N_TIME_STEPS, MAX_TRAVEL_PX)

    # Ideal forward (no mismatch, no noise)
    y_ideal = forward_operator(x_true, positions_ideal, radii_ideal)

    # Mismatched geometry
    positions_mis = get_transducer_positions(
        N_TRANSDUCERS, TRANSDUCER_RADIUS_PX, IMAGE_CENTRE,
        position_noise=transducer_position_error, rng=rng)

    # SoS error stretches time-to-radius mapping
    sos_factor = 1.0 + speed_of_sound_error_pct / 100.0
    radii_mis = radii_ideal * sos_factor

    y_mismatch = forward_operator(x_true, positions_mis, radii_mis)

    # Add Gaussian noise
    sig_peak = float(np.abs(y_mismatch).max()) + 1e-10
    noise = rng.normal(0.0, noise_sigma * sig_peak,
                       y_mismatch.shape).astype(np.float32)
    y_measured = y_mismatch + noise

    return y_measured, y_ideal


# ── Phantom generators ───────────────────────────────────────────────────────

def make_vessel_tree_phantom(rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """Branching vascular network phantom."""
    canvas = np.zeros((size, size), dtype=np.float64)
    centre = size / 2.0
    n_roots = rng.integers(2, 5)
    queue = []
    for _ in range(n_roots):
        side = rng.integers(0, 4)
        margin = rng.integers(size // 7, size // 4)
        if side == 0:
            ry, rx = float(margin), float(rng.integers(size // 5, 4 * size // 5))
        elif side == 1:
            ry, rx = float(size - 1 - margin), float(rng.integers(size // 5, 4 * size // 5))
        elif side == 2:
            ry, rx = float(rng.integers(size // 5, 4 * size // 5)), float(margin)
        else:
            ry, rx = float(rng.integers(size // 5, 4 * size // 5)), float(size - 1 - margin)
        angle = np.arctan2(centre - ry, centre - rx) + rng.normal(0, 0.3)
        thick = rng.uniform(1.5, 4.0)
        amp = rng.uniform(0.6, 1.0)
        queue.append((ry, rx, angle, thick, amp, 0))

    n_branches = 0
    max_branches = 60
    while queue and n_branches < max_branches:
        cy, cx, angle, thick, amp, depth = queue.pop(0)
        walk_len = int(rng.integers(20, 70))
        for _ in range(walk_len):
            angle += rng.normal(0, 0.08)
            ny = cy + np.sin(angle)
            nx = cx + np.cos(angle)
            if ny < 1 or ny >= size - 1 or nx < 1 or nx >= size - 1:
                break
            iy, ix = int(round(ny)), int(round(nx))
            r = max(1, int(np.ceil(thick * 2.5)))
            y0, y1 = max(0, iy - r), min(size, iy + r + 1)
            x0, x1 = max(0, ix - r), min(size, ix + r + 1)
            if y1 > y0 and x1 > x0:
                yg = np.arange(y0, y1, dtype=np.float64)[:, None] - ny
                xg = np.arange(x0, x1, dtype=np.float64)[None, :] - nx
                canvas[y0:y1, x0:x1] = np.maximum(
                    canvas[y0:y1, x0:x1],
                    amp * np.exp(-(yg ** 2 + xg ** 2) / (2.0 * thick ** 2)))
            cy, cx = ny, nx
            if depth < 5 and rng.random() < 0.04:
                new_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
                new_thick = thick * rng.uniform(0.5, 0.8)
                if new_thick >= 0.5:
                    queue.append((cy, cx, new_angle, new_thick, amp * 0.75, depth + 1))
        n_branches += 1

    # Fallback if empty
    if canvas.max() < 0.01:
        for _ in range(rng.integers(3, 6)):
            ang = rng.uniform(0, 2 * np.pi)
            cy, cx = centre, centre
            thick = rng.uniform(1.5, 3.0)
            for s in range(int(size * 0.35)):
                ny, nx = cy + np.sin(ang), cx + np.cos(ang)
                ang += rng.normal(0, 0.05)
                if not (1 <= ny < size - 1 and 1 <= nx < size - 1):
                    break
                iy, ix = int(round(ny)), int(round(nx))
                r = max(1, int(np.ceil(thick * 2)))
                y0, y1 = max(0, iy - r), min(size, iy + r + 1)
                x0, x1 = max(0, ix - r), min(size, ix + r + 1)
                if y1 > y0 and x1 > x0:
                    yg = np.arange(y0, y1, dtype=np.float64)[:, None] - ny
                    xg = np.arange(x0, x1, dtype=np.float64)[None, :] - nx
                    canvas[y0:y1, x0:x1] = np.maximum(
                        canvas[y0:y1, x0:x1],
                        0.8 * np.exp(-(yg ** 2 + xg ** 2) / (2.0 * thick ** 2)))
                cy, cx = ny, nx
    return np.clip(canvas, 0, 1).astype(np.float32)


def make_point_absorbers_phantom(rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """Point absorbers + diffuse background (mimics point-source PAT targets)."""
    canvas = np.zeros((size, size), dtype=np.float64)
    n_pts = rng.integers(10, 35)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_pts):
        py = rng.integers(10, size - 10)
        px = rng.integers(10, size - 10)
        pr = rng.uniform(1.0, 5.0)
        amp = rng.uniform(0.3, 1.0)
        d2 = (yy - py) ** 2 + (xx - px) ** 2
        canvas += amp * np.exp(-d2 / (2.0 * pr ** 2))
    # Diffuse background
    bg = gaussian_filter(rng.standard_normal((size, size)), sigma=15)
    canvas += 0.05 * (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    return np.clip(canvas, 0, 1).astype(np.float32)


def make_tumor_phantom(rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """Tumor vasculature: dense chaotic vessels near a tumour core."""
    canvas = np.zeros((size, size), dtype=np.float64)
    tc_y = rng.integers(size // 4, 3 * size // 4)
    tc_x = rng.integers(size // 4, 3 * size // 4)
    t_radius = rng.integers(20, 50)
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - tc_y) ** 2 + (xx - tc_x) ** 2)
    canvas[dist < t_radius] += rng.uniform(0.05, 0.15)
    # Tortuous vessels in tumour
    for _ in range(rng.integers(20, 40)):
        angle = rng.uniform(0, 2 * np.pi)
        start_r = rng.uniform(0, t_radius * 0.8)
        cy = float(tc_y + start_r * np.sin(angle))
        cx = float(tc_x + start_r * np.cos(angle))
        walk_angle = rng.uniform(0, 2 * np.pi)
        thick = rng.uniform(0.8, 2.5)
        amp = rng.uniform(0.5, 1.0)
        for _ in range(rng.integers(8, 30)):
            walk_angle += rng.normal(0, 0.2)
            cy += np.sin(walk_angle)
            cx += np.cos(walk_angle)
            if not (1 <= cy < size - 1 and 1 <= cx < size - 1):
                break
            iy, ix = int(round(cy)), int(round(cx))
            r = max(1, int(np.ceil(thick * 2)))
            y0, y1 = max(0, iy - r), min(size, iy + r + 1)
            x0, x1 = max(0, ix - r), min(size, ix + r + 1)
            if y1 > y0 and x1 > x0:
                yg = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
                xg = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
                canvas[y0:y1, x0:x1] = np.maximum(
                    canvas[y0:y1, x0:x1],
                    amp * np.exp(-(yg ** 2 + xg ** 2) / (2.0 * thick ** 2)))
    # Feeding vessels
    tree = make_vessel_tree_phantom(rng, size)
    canvas = np.maximum(canvas, tree * 0.5)
    return np.clip(canvas, 0, 1).astype(np.float32)


PHANTOM_FNS = [
    make_vessel_tree_phantom,
    make_point_absorbers_phantom,
    make_tumor_phantom,
]
PHANTOM_NAMES = ["vessel_tree", "point_absorbers", "tumor_vasculature"]


def make_phantom(idx: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    fn = PHANTOM_FNS[idx % len(PHANTOM_FNS)]
    name = PHANTOM_NAMES[idx % len(PHANTOM_NAMES)]
    x_raw = fn(rng)
    xmin, xmax = float(x_raw.min()), float(x_raw.max())
    if xmax - xmin > 1e-8:
        return ((x_raw - xmin) / (xmax - xmin)).astype(np.float32), name
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32), name


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    g, r = gt.astype(np.float64), recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_g, mu_r = g.mean(), r.mean()
    cov = float(np.mean((g - mu_g) * (r - mu_r)))
    return float(
        ((2 * mu_g * mu_r + c1) * (2 * cov + c2))
        / ((mu_g ** 2 + mu_r ** 2 + c1) * (g.var() + r.var() + c2))
    )


# ── PNG helpers ──────────────────────────────────────────────────────────────

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, target_size: int = IMAGE_SIZE) -> None:
    if not HAS_PIL:
        return
    img = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "L")
    if pil.size != (target_size, target_size):
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
    pil.save(str(path))


# ── Mismatch sampler ─────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


# ── Tier generator ───────────────────────────────────────────────────────────

def generate_tier(tier: str, n_samples: int, seed_offset: int, mismatch_seed: int) -> None:
    """Generate one tier of the photoacoustic benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = OUT_BASE / tier
    images_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"photoacoustic_challenge_{tier}.h5"
    rng = np.random.default_rng(mismatch_seed)
    true_specs: dict = {}
    psnrs, ssims = [], []

    # Ideal transducer positions (128, 2) -- stored as H_ideal
    H_ideal_positions = get_transducer_positions(N_TRANSDUCERS, TRANSDUCER_RADIUS_PX,
                                                 IMAGE_CENTRE)
    time_radii_nominal = get_time_radii(N_TIME_STEPS, MAX_TRAVEL_PX)

    print(f"\n[{tier}] Generating {n_samples} samples -> {h5_path}")
    print(f"  H_ideal shape: {H_ideal_positions.shape}  (transducer [row,col] positions)")
    print(f"  y shape per sample: ({N_TRANSDUCERS}, {N_TIME_STEPS})")

    with h5py.File(h5_path, "w") as f:
        f.attrs["modality"] = "photoacoustic_tomography"
        f.attrs["tier"] = tier
        f.attrs["n_samples"] = n_samples
        f.attrs["image_size"] = IMAGE_SIZE
        f.attrs["n_transducers"] = N_TRANSDUCERS
        f.attrs["n_time_steps"] = N_TIME_STEPS
        f.attrs["transducer_radius_px"] = TRANSDUCER_RADIUS_PX
        f.attrs["image_centre_px"] = IMAGE_CENTRE
        f.attrs["speed_of_sound_m_s"] = SPEED_OF_SOUND
        f.attrs["pixel_size_mm"] = PIXEL_SIZE_MM
        f.attrs["max_travel_px"] = MAX_TRAVEL_PX
        f.attrs["forward_model"] = (
            "y(d,t) = integral of x along circle of radius time_radii[t] "
            "centred at transducer_positions[d] + Gaussian noise. "
            "y shape: (128, 200). H_ideal: (128,2) ideal transducer [row,col]."
        )
        f.attrs["baseline_method"] = (
            "Delay-and-sum back projection (UBP adjoint) with nominal geometry"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["H_ideal_description"] = (
            "H_ideal stored per sample as (128, 2) float32 array of ideal "
            "transducer positions [row, col] in pixel coordinates. "
            "Nominal radius=100px around image centre (128,128)."
        )

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            x_true, phantom_name = make_phantom(idx, seed_offset + idx)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "phantom_seed": seed_offset + idx,
                               "phantom_type": phantom_name}

            sample_rng = np.random.default_rng(mismatch_seed + idx + 100)

            print(f"  [{tier}] {key} ({phantom_name}) -- forward model...",
                  end="", flush=True)

            y_measured, y_ideal = full_forward_model(
                x_true,
                speed_of_sound_error_pct=mis["speed_of_sound_error_pct"],
                transducer_position_error=mis["transducer_position_error"],
                noise_sigma=mis["noise_sigma"],
                rng=sample_rng,
            )

            # Delay-and-sum back projection (baseline)
            recon = backproject_operator(
                y_measured, H_ideal_positions, time_radii_nominal, IMAGE_SIZE)

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            psnrs.append(psnr)
            ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y_measured, compression="gzip")
            # H_ideal: (128, 2) transducer positions array
            grp.create_dataset("H_ideal", data=H_ideal_positions, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")

            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["metadata"] = json.dumps({
                "scene": phantom_name,
                "shapes": {
                    "x_true": list(x_true.shape),
                    "y": list(y_measured.shape),
                    "H_ideal": list(H_ideal_positions.shape),
                    "reconstruction_baseline": list(recon.shape),
                },
                "x_true_range": [float(x_true.min()), float(x_true.max())],
                "y_range": [float(y_measured.min()), float(y_measured.max())],
                "H_ideal_description": "transducer [row, col] positions in pixel coords",
                "recon_range": [float(recon.min()), float(recon.max())],
                "psnr_baseline_db": round(psnr, 2),
                "ssim_baseline": round(ssim, 4),
            })

            # Images
            sample_img_dir = images_dir / f"sample_{idx:02d}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_img_dir / "x_true.png")
            # Save sinogram as PNG (resized to IMAGE_SIZE for display)
            _save_png(y_measured, sample_img_dir / "y_sinogram.png")
            _save_png(recon, sample_img_dir / "reconstruction.png")
            with open(sample_img_dir / "spec.json", "w") as sf:
                json.dump({"true_spec": mis, "spec_ranges": spec_ranges,
                           "psnr_db": psnr, "ssim": ssim}, sf, indent=2)

            print(f"  x_true {x_true.shape} [{x_true.min():.4f},{x_true.max():.4f}] "
                  f"y {y_measured.shape} [{y_measured.min():.4f},{y_measured.max():.4f}] "
                  f"H_ideal {H_ideal_positions.shape} "
                  f"recon {recon.shape} [{recon.min():.4f},{recon.max():.4f}] "
                  f"SoS_err={mis['speed_of_sound_error_pct']:.1f}%  "
                  f"pos_err={mis['transducer_position_error']:.2f}px  "
                  f"noise={mis['noise_sigma']:.3f}  "
                  f"PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

        f.attrs["mean_psnr_baseline_db"] = float(np.mean(psnrs))
        f.attrs["mean_ssim_baseline"] = float(np.mean(ssims))

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    print(f"  [{tier}] Done: {n_samples} samples | "
          f"mean PSNR={np.mean(psnrs):.2f} dB | mean SSIM={np.mean(ssims):.4f}")
    print(f"  [{tier}] HDF5: {h5_path}  "
          f"({os.path.getsize(h5_path) / 1e6:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Photoacoustic Imaging Benchmark Dataset Generator")
    print(f"Output: {OUT_BASE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Transducers: {N_TRANSDUCERS} on ring, radius={TRANSDUCER_RADIUS_PX}px, "
          f"centre=({IMAGE_CENTRE},{IMAGE_CENTRE})")
    print(f"y shape: ({N_TRANSDUCERS}, {N_TIME_STEPS}) = (n_transducers, n_time_steps)")
    print(f"H_ideal shape: ({N_TRANSDUCERS}, 2) = transducer [row,col] positions")
    print(f"Speed of sound: {SPEED_OF_SOUND} m/s nominal")
    print(f"Baseline: Delay-and-sum back projection (UBP)")
    print("=" * 70)

    generate_tier("public",  n_samples=12, seed_offset=0,     mismatch_seed=1000)
    generate_tier("dev",     n_samples=20, seed_offset=10000, mismatch_seed=11000)
    generate_tier("hidden",  n_samples=20, seed_offset=20000, mismatch_seed=21000)

    print("\n" + "=" * 70)
    print("Photoacoustic benchmark generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
