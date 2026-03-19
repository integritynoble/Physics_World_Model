#!/usr/bin/env python3
"""Generate PALM/STORM (Single-Molecule Localization Microscopy) benchmark dataset.

Forward model (per frame):
    y_t = Poisson(PSF * x_sparse_t + background) + readout_noise

where:
    x_sparse_t  : sparse activation map (only ~1% of molecules active per frame)
    PSF         : Gaussian point spread function (sigma ~2-3 px, ~250 nm FWHM)
    background  : autofluorescence background (Poisson)
    readout_noise : Gaussian camera noise (sCMOS/EMCCD)

The ground truth x_true is the super-resolved image (all molecule positions
rendered at high resolution). The measurement y is a single noisy widefield
frame with sparse molecular activations.

Mismatch parameters:
    psf_sigma_px        : PSF standard deviation in pixels (2.0 - 4.0)
    photon_count        : mean photons per molecule (200 - 2000)
    background_level    : background photons per pixel (5 - 50)
    activation_density  : fraction of molecules active per frame (0.005 - 0.05)

Phantoms:
    Subcellular structures: microtubules, mitochondria, nuclear pore complexes,
    membrane structures. 50-200 molecule positions per sample.

CPU Baseline reconstruction:
    Gaussian fitting localization + histogram binning (simple SMLM pipeline).
    Expected: ~18-24 dB PSNR.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/palm_storm
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as nd_label
from scipy.optimize import least_squares

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256           # super-resolved ground truth
MEAS_SIZE = 256            # measurement (widefield) -- same pixel grid
PIXEL_SIZE_NM = 100.0      # 100 nm/pixel for the super-resolution image
CAMERA_PIXEL_NM = 100.0    # camera pixel size (same grid; PSF blurs molecules)

# -- Mismatch ranges per tier ------------------------------------------------

SPEC = {
    "public": {
        "psf_sigma_px":       {"min": 2.0, "max": 3.0, "unit": "pixels"},
        "photon_count":       {"min": 500, "max": 2000, "unit": "photons"},
        "background_level":   {"min": 5,   "max": 20,  "unit": "photons/pixel"},
        "activation_density": {"min": 0.01, "max": 0.03, "unit": "fraction"},
    },
    "dev": {
        "psf_sigma_px":       {"min": 2.0, "max": 3.5, "unit": "pixels"},
        "photon_count":       {"min": 300, "max": 2000, "unit": "photons"},
        "background_level":   {"min": 5,   "max": 35,  "unit": "photons/pixel"},
        "activation_density": {"min": 0.01, "max": 0.04, "unit": "fraction"},
    },
    "hidden": {
        "psf_sigma_px":       {"min": 2.5, "max": 4.0, "unit": "pixels"},
        "photon_count":       {"min": 200, "max": 1500, "unit": "photons"},
        "background_level":   {"min": 10,  "max": 50,  "unit": "photons/pixel"},
        "activation_density": {"min": 0.01, "max": 0.05, "unit": "fraction"},
    },
}


# -- Phantom generators (subcellular structures) -----------------------------

def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line rasterization."""
    pts = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y1 > y0 else -1
    sx = 1 if x1 > x0 else -1
    err = dx - dy
    while True:
        pts.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts


def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 300
) -> np.ndarray:
    """Generate smooth curve via cubic interpolation of control points.

    Args:
        control_pts: (N, 2) array of (y, x) control points
        n_interp: number of interpolated points

    Returns:
        (n_interp, 2) array of (y, x) interpolated points
    """
    from scipy.interpolate import CubicSpline

    n = len(control_pts)
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def make_microtubule_phantom(
    H: int, W: int, rng: np.random.Generator, n_molecules: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Microtubules: thin curved filaments radiating from cell center.

    Returns:
        positions: (N, 2) float64 molecule positions (y, x) in pixel coords
        x_true: (H, W) float64 super-resolved ground truth image
    """
    positions = []
    n_filaments = rng.integers(5, 12)

    for _ in range(n_filaments):
        # Start near center, extend outward
        cy = H / 2 + rng.uniform(-H * 0.15, H * 0.15)
        cx = W / 2 + rng.uniform(-W * 0.15, W * 0.15)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.2, H * 0.45)

        # Generate curved path with 4-6 control points
        n_ctrl = rng.integers(4, 7)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / (n_ctrl - 1)
            ctrl[j, 0] = cy + length * t * np.sin(angle) + rng.uniform(-15, 15)
            ctrl[j, 1] = cx + length * t * np.cos(angle) + rng.uniform(-15, 15)

        curve = _smooth_curve_points(ctrl, n_interp=500)
        # Filter points within bounds
        valid = (
            (curve[:, 0] >= 2) & (curve[:, 0] < H - 2)
            & (curve[:, 1] >= 2) & (curve[:, 1] < W - 2)
        )
        curve = curve[valid]
        if len(curve) < 10:
            continue

        # Sample molecule positions along filament with small transverse jitter
        n_mol_this = max(5, n_molecules // n_filaments)
        indices = rng.choice(len(curve), size=min(n_mol_this, len(curve)), replace=False)
        for idx in indices:
            jitter_y = rng.normal(0, 0.5)
            jitter_x = rng.normal(0, 0.5)
            py = curve[idx, 0] + jitter_y
            px = curve[idx, 1] + jitter_x
            if 0 <= py < H and 0 <= px < W:
                positions.append([py, px])

    positions = np.array(positions) if positions else np.zeros((0, 2))

    # Trim or pad to desired count
    if len(positions) > n_molecules:
        idx = rng.choice(len(positions), n_molecules, replace=False)
        positions = positions[idx]

    x_true = _render_positions(positions, H, W, sigma=0.8)
    return positions, x_true


def make_mitochondria_phantom(
    H: int, W: int, rng: np.random.Generator, n_molecules: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Mitochondria: elongated blob-like organelles scattered in cytoplasm.

    Returns:
        positions: (N, 2) float64 molecule positions
        x_true: (H, W) float64 ground truth
    """
    positions = []
    n_mito = rng.integers(5, 15)

    for _ in range(n_mito):
        # Each mitochondrion: elongated ellipse with membrane labeling
        cy = rng.uniform(H * 0.15, H * 0.85)
        cx = rng.uniform(W * 0.15, W * 0.85)
        a = rng.uniform(8, 30)  # semi-major axis (pixels)
        b = rng.uniform(3, 8)   # semi-minor axis
        angle = rng.uniform(0, np.pi)

        # Generate molecules along the membrane (ellipse boundary)
        n_mol_this = max(4, n_molecules // n_mito)
        thetas = rng.uniform(0, 2 * np.pi, n_mol_this)
        for theta in thetas:
            r_y = a * np.sin(theta)
            r_x = b * np.cos(theta)
            # Rotate
            py = cy + r_y * np.cos(angle) - r_x * np.sin(angle) + rng.normal(0, 0.8)
            px = cx + r_y * np.sin(angle) + r_x * np.cos(angle) + rng.normal(0, 0.8)
            if 0 <= py < H and 0 <= px < W:
                positions.append([py, px])

    positions = np.array(positions) if positions else np.zeros((0, 2))
    if len(positions) > n_molecules:
        idx = rng.choice(len(positions), n_molecules, replace=False)
        positions = positions[idx]

    x_true = _render_positions(positions, H, W, sigma=0.8)
    return positions, x_true


def make_nuclear_pore_phantom(
    H: int, W: int, rng: np.random.Generator, n_molecules: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Nuclear pore complexes: small ring patterns (~120 nm diameter).

    Returns:
        positions: (N, 2) float64 molecule positions
        x_true: (H, W) float64 ground truth
    """
    positions = []
    # Nuclear envelope: large circle
    nuc_cy = H / 2 + rng.uniform(-10, 10)
    nuc_cx = W / 2 + rng.uniform(-10, 10)
    nuc_r = rng.uniform(H * 0.25, H * 0.38)

    # Place 8-20 nuclear pore complexes on the envelope
    n_pores = rng.integers(8, 21)
    pore_angles = np.sort(rng.uniform(0, 2 * np.pi, n_pores))

    for pa in pore_angles:
        # Pore center on nuclear envelope
        pore_cy = nuc_cy + nuc_r * np.sin(pa) + rng.normal(0, 2)
        pore_cx = nuc_cx + nuc_r * np.cos(pa) + rng.normal(0, 2)

        # Each NPC: 8-fold symmetric ring, radius ~6 pixels (~60 nm)
        pore_r = rng.uniform(4, 8)
        n_subunits = 8
        for k in range(n_subunits):
            theta = 2 * np.pi * k / n_subunits + rng.uniform(-0.1, 0.1)
            py = pore_cy + pore_r * np.sin(theta) + rng.normal(0, 0.3)
            px = pore_cx + pore_r * np.cos(theta) + rng.normal(0, 0.3)
            if 0 <= py < H and 0 <= px < W:
                positions.append([py, px])

    positions = np.array(positions) if positions else np.zeros((0, 2))
    if len(positions) > n_molecules:
        idx = rng.choice(len(positions), n_molecules, replace=False)
        positions = positions[idx]

    x_true = _render_positions(positions, H, W, sigma=0.8)
    return positions, x_true


def make_membrane_phantom(
    H: int, W: int, rng: np.random.Generator, n_molecules: int = 130,
) -> tuple[np.ndarray, np.ndarray]:
    """Membrane structures: cell boundary with clustered receptors.

    Returns:
        positions: (N, 2) float64 molecule positions
        x_true: (H, W) float64 ground truth
    """
    positions = []

    # Main cell membrane: irregular closed curve (deformed circle)
    n_ctrl = rng.integers(8, 16)
    base_r = rng.uniform(H * 0.25, H * 0.40)
    ctrl_angles = np.sort(rng.uniform(0, 2 * np.pi, n_ctrl))
    ctrl_angles = np.append(ctrl_angles, ctrl_angles[0] + 2 * np.pi)  # close loop
    ctrl_r = base_r + rng.uniform(-base_r * 0.2, base_r * 0.2, len(ctrl_angles))
    ctrl_r[-1] = ctrl_r[0]  # close loop

    cy = H / 2
    cx = W / 2

    ctrl_pts = np.column_stack([
        cy + ctrl_r * np.sin(ctrl_angles),
        cx + ctrl_r * np.cos(ctrl_angles),
    ])
    curve = _smooth_curve_points(ctrl_pts, n_interp=800)

    # Molecule positions along membrane
    valid = (
        (curve[:, 0] >= 2) & (curve[:, 0] < H - 2)
        & (curve[:, 1] >= 2) & (curve[:, 1] < W - 2)
    )
    curve = curve[valid]

    # Uniform labeling along membrane
    n_uniform = int(n_molecules * 0.6)
    if len(curve) > 0:
        indices = rng.choice(len(curve), size=min(n_uniform, len(curve)), replace=False)
        for idx in indices:
            jitter_y = rng.normal(0, 0.5)
            jitter_x = rng.normal(0, 0.5)
            positions.append([curve[idx, 0] + jitter_y, curve[idx, 1] + jitter_x])

    # Receptor clusters (bright spots on membrane)
    n_clusters = rng.integers(3, 8)
    n_remaining = n_molecules - len(positions)
    for _ in range(n_clusters):
        if len(curve) == 0:
            break
        center_idx = rng.integers(0, len(curve))
        cluster_cy = curve[center_idx, 0]
        cluster_cx = curve[center_idx, 1]
        n_in_cluster = max(2, n_remaining // n_clusters)
        for _ in range(n_in_cluster):
            py = cluster_cy + rng.normal(0, 2.0)
            px = cluster_cx + rng.normal(0, 2.0)
            if 0 <= py < H and 0 <= px < W:
                positions.append([py, px])

    positions = np.array(positions) if positions else np.zeros((0, 2))
    if len(positions) > n_molecules:
        idx = rng.choice(len(positions), n_molecules, replace=False)
        positions = positions[idx]

    x_true = _render_positions(positions, H, W, sigma=0.8)
    return positions, x_true


def _render_positions(
    positions: np.ndarray, H: int, W: int, sigma: float = 0.8
) -> np.ndarray:
    """Render molecule positions as Gaussian spots on a high-res image.

    Each molecule is a tiny Gaussian spot (sigma ~ 0.8 px) at sub-pixel position.
    The result represents the "true" super-resolved image.
    """
    x_true = np.zeros((H, W), dtype=np.float64)
    if len(positions) == 0:
        return x_true.astype(np.float32)

    r = int(np.ceil(3 * sigma))
    for py, px in positions:
        iy = int(round(py))
        ix = int(round(px))
        y0 = max(0, iy - r)
        y1 = min(H, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(W, ix + r + 1)
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        gaussian = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
        x_true[y0:y1, x0:x1] += gaussian

    # Normalize to [0, 1]
    if x_true.max() > 0:
        x_true /= x_true.max()

    return x_true.astype(np.float32)


# -- PALM/STORM forward model ------------------------------------------------

def palm_storm_forward(
    positions: np.ndarray,
    H: int,
    W: int,
    psf_sigma_px: float,
    photon_count: float,
    background_level: float,
    activation_density: float,
    rng: np.random.Generator,
    readout_noise_std: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """PALM/STORM forward model: generate a single noisy widefield frame.

    Args:
        positions: (N, 2) molecule positions (y, x) in pixels
        H, W: image dimensions
        psf_sigma_px: PSF Gaussian sigma in pixels
        photon_count: mean photons per active molecule
        background_level: mean background photons per pixel
        activation_density: fraction of molecules active this frame
        rng: random generator
        readout_noise_std: Gaussian readout noise std (electrons)

    Returns:
        y: (H, W) float32 measured frame (noisy)
        H_ideal: (H, W) float32 ideal (noiseless) PSF-convolved active molecules
    """
    n_molecules = len(positions)
    if n_molecules == 0:
        y = rng.poisson(background_level, (H, W)).astype(np.float64)
        y += rng.normal(0, readout_noise_std, (H, W))
        return np.maximum(y, 0).astype(np.float32), np.zeros((H, W), dtype=np.float32)

    # Stochastic activation: each molecule has probability = activation_density
    active = rng.random(n_molecules) < activation_density
    active_pos = positions[active]

    # Build ideal image: sum of PSF spots from active molecules
    H_ideal = np.zeros((H, W), dtype=np.float64)
    r = int(np.ceil(4 * psf_sigma_px))

    for py, px in active_pos:
        # Each molecule emits Poisson(photon_count) photons
        n_photons = rng.poisson(photon_count)
        if n_photons <= 0:
            continue

        iy = int(round(py))
        ix = int(round(px))
        y0 = max(0, iy - r)
        y1 = min(H, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        psf = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * psf_sigma_px ** 2))
        psf_sum = psf.sum()
        if psf_sum > 0:
            psf /= psf_sum  # normalize PSF to sum to 1
        H_ideal[y0:y1, x0:x1] += n_photons * psf

    # Add background
    signal_plus_bg = H_ideal + background_level

    # Poisson noise (shot noise)
    y = rng.poisson(np.maximum(signal_plus_bg, 0.01)).astype(np.float64)

    # Readout noise (Gaussian)
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal.astype(np.float32)


# -- CPU Baseline: Gaussian fitting localization + histogram binning ----------

def _fit_gaussian_2d(patch: np.ndarray, cy_init: float, cx_init: float,
                     sigma_init: float = 2.5) -> tuple[float, float, float, float]:
    """Fit a 2D Gaussian to a small patch via least-squares.

    Returns: (y_center, x_center, amplitude, sigma) in patch coordinates.
    """
    h, w = patch.shape
    yy, xx = np.mgrid[:h, :w]
    yy_flat = yy.ravel().astype(np.float64)
    xx_flat = xx.ravel().astype(np.float64)
    data = patch.ravel().astype(np.float64)
    bg = float(np.percentile(data, 10))

    def residuals(p):
        amp, y0, x0, sig = p
        model = amp * np.exp(-((yy_flat - y0) ** 2 + (xx_flat - x0) ** 2) / (2 * sig ** 2)) + bg
        return model - data

    p0 = [float(patch.max() - bg), cy_init, cx_init, sigma_init]
    try:
        result = least_squares(
            residuals, p0,
            bounds=([0, -1, -1, 0.5], [1e6, h + 1, w + 1, 10.0]),
            max_nfev=50,
        )
        amp, y0, x0, sig = result.x
        return float(y0), float(x0), float(amp), float(sig)
    except Exception:
        return cy_init, cx_init, float(patch.max() - bg), sigma_init


def smlm_baseline_reconstruct(
    y: np.ndarray,
    psf_sigma_est: float = 2.5,
    detection_threshold: float = 3.0,
    output_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Simple SMLM reconstruction: detect spots, fit Gaussians, bin to histogram.

    Pipeline:
        1. Subtract background (median filter)
        2. Detect bright spots above threshold
        3. Fit 2D Gaussian to each spot
        4. Accumulate localized positions into a histogram image

    Args:
        y: (H, W) single noisy frame
        psf_sigma_est: estimated PSF sigma for fitting
        detection_threshold: detection threshold in units of noise std
        output_size: output image size (same as gt)

    Returns:
        recon: (output_size, output_size) float32 reconstruction
    """
    y64 = y.astype(np.float64)
    H, W = y64.shape

    # Background estimation (large-scale median via Gaussian smoothing)
    bg = gaussian_filter(y64, sigma=max(psf_sigma_est * 5, 15))
    residual = y64 - bg

    # Noise estimation
    noise_std = max(np.median(np.abs(residual)) * 1.4826, 1.0)  # MAD estimator

    # Detection: threshold
    det_map = residual > detection_threshold * noise_std

    # Connected component labeling to find individual spots
    labeled, n_features = nd_label(det_map)

    localizations = []
    fit_radius = int(np.ceil(3 * psf_sigma_est))

    for label_id in range(1, n_features + 1):
        ys, xs = np.where(labeled == label_id)
        if len(ys) < 3:
            continue
        # Centroid of detected region
        cy = float(ys.mean())
        cx = float(xs.mean())
        iy = int(round(cy))
        ix = int(round(cx))

        # Extract patch for Gaussian fitting
        y0 = max(0, iy - fit_radius)
        y1 = min(H, iy + fit_radius + 1)
        x0 = max(0, ix - fit_radius)
        x1 = min(W, ix + fit_radius + 1)
        patch = y64[y0:y1, x0:x1]

        if patch.size < 4:
            continue

        # Fit Gaussian
        local_cy = cy - y0
        local_cx = cx - x0
        fit_y, fit_x, amp, sig = _fit_gaussian_2d(
            patch, local_cy, local_cx, psf_sigma_est
        )

        # Convert back to image coordinates
        loc_y = fit_y + y0
        loc_x = fit_x + x0

        # Quality filter: reasonable amplitude, sigma, and position
        if amp > noise_std * 2 and 0.5 < sig < 8.0 and 0 <= loc_y < H and 0 <= loc_x < W:
            localizations.append((loc_y, loc_x, amp))

    # Render localizations into histogram image
    recon = np.zeros((output_size, output_size), dtype=np.float64)
    render_sigma = 0.8  # sub-pixel Gaussian rendering

    for loc_y, loc_x, amp in localizations:
        # Map to output coordinates (same grid)
        oy = loc_y
        ox = loc_x
        iy = int(round(oy))
        ix = int(round(ox))
        r = 2
        y0 = max(0, iy - r)
        y1 = min(output_size, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(output_size, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = np.exp(-((yy - oy) ** 2 + (xx - ox) ** 2) / (2 * render_sigma ** 2))
        recon[y0:y1, x0:x1] += g

    if recon.max() > 0:
        recon /= recon.max()

    return recon.astype(np.float32)


# -- Metrics -----------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    mse = np.mean((gt64 - recon64) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt64.max() - gt64.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    data_range = gt64.max() - gt64.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gt64.mean()
    mu_y = recon64.mean()
    var_x = gt64.var()
    var_y = recon64.var()
    cov_xy = np.mean((gt64 - mu_x) * (recon64 - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# -- Image helpers -----------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            lo, hi = np.percentile(nonzero, [1, 99])
            arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# -- Phantom pool for each tier -----------------------------------------------

PHANTOM_FNS = [
    make_microtubule_phantom,
    make_mitochondria_phantom,
    make_nuclear_pore_phantom,
    make_membrane_phantom,
]

PHANTOM_NAMES = ["microtubule", "mitochondria", "nuclear_pore", "membrane"]


def generate_phantoms(
    n: int, seed_offset: int, tier: str,
) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    """Generate phantom set for a tier.

    Returns list of (positions, x_true, scene_name, n_molecules).
    """
    phantoms = []
    rng = np.random.default_rng(seed_offset + 42)

    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]
        n_molecules = int(rng.integers(50, 201))

        phantom_rng = np.random.default_rng(seed_offset + i)
        positions, x_true = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng, n_molecules)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((positions, x_true, scene_name, len(positions)))

    return phantoms


# -- Tier generation ---------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        # Round integer-valued params
        if k in ("photon_count", "background_level"):
            val = round(val)
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, str, int]],
    base_seed: int,
) -> None:
    """Generate one tier of the PALM/STORM benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"palm_storm_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM PALM/STORM benchmark -- {tier} tier "
            f"(single-molecule localization microscopy)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF * x_sparse + background) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "camera_pixel_nm": CAMERA_PIXEL_NM,
        })

        for idx, (positions, x_true, scene_name, n_mol) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name}, "
                  f"{n_mol} molecules)...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "n_molecules": n_mol}

            # Forward model: generate noisy widefield frame
            y, H_ideal = palm_storm_forward(
                positions, IMAGE_SIZE, IMAGE_SIZE,
                psf_sigma_px=mis["psf_sigma_px"],
                photon_count=mis["photon_count"],
                background_level=mis["background_level"],
                activation_density=mis["activation_density"],
                rng=rng,
            )

            # CPU baseline reconstruction
            recon = smlm_baseline_reconstruct(
                y, psf_sigma_est=mis["psf_sigma_px"],
                detection_threshold=3.0,
                output_size=IMAGE_SIZE,
            )

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("positions", data=positions.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "n_molecules": n_mol,
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps({**mis, "n_molecules": n_mol})
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "gt.png")
            _save_png(y, sample_dir / "measurement.png", percentile_clip=True)
            _save_png(recon, sample_dir / "recon.png")
            _save_png(H_ideal, sample_dir / "H_ideal.png", percentile_clip=True)
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": {**mis, "n_molecules": n_mol},
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"psf_sigma={mis['psf_sigma_px']:.2f}  "
                  f"photons={mis['photon_count']}  "
                  f"bg={mis['background_level']}  "
                  f"density={mis['activation_density']:.3f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    if tier == "public":
        with open(tier_dir / "true_spec.json", "w") as tf:
            json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# -- Gallery image generation ------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "palm_storm"
    )

    h5_path = BENCHMARK_DIR / "public" / "palm_storm_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: microtubule, mitochondria, nuclear_pore, membrane
    gallery_indices = [0, 1, 2, 3]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            y = grp["y"][:]
            H_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]

            # gt.png -- super-resolved ground truth
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy widefield frame
            _save_png(y, scene_dir / "measurement_I.png", percentile_clip=True)

            # measurement_II.png -- ideal PSF-convolved active molecules
            _save_png(H_ideal, scene_dir / "measurement_II.png", percentile_clip=True)

            # recon_I.png -- baseline SMLM reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# -- Main --------------------------------------------------------------------

def main() -> None:
    print("PALM/STORM Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, pixel size: {PIXEL_SIZE_NM} nm\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0, tier="public")
    generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000, tier="dev")
    generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000, tier="hidden")
    generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    # Spec files in public/
    print("\nWriting spec files...")
    pub_spec = BENCHMARK_DIR / "public" / "spec.json"
    if pub_spec.exists():
        print(f"  spec.json already exists at {pub_spec}")

    print(f"\n{'=' * 68}")
    print("PALM/STORM benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
