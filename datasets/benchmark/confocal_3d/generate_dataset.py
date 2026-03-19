#!/usr/bin/env python3
"""Generate Confocal 3D Microscopy benchmark dataset.

Forward model (single z-slice from 3D confocal volume):
    y = Poisson(PSF_confocal * x + out_of_focus_blur + bg) + readout_noise

where:
    x              : ground truth fluorescence at the focal z-plane (256x256)
    PSF_confocal   : confocal PSF = excitation PSF x detection PSF (narrower
                     than widefield due to pinhole). Lateral FWHM ~ 0.4*lambda/NA.
    out_of_focus_blur : contribution from adjacent z-planes convolved with a
                     wider (depth-dependent) PSF. Simulates incomplete optical
                     sectioning.
    bg             : autofluorescence / dark-current background (photons/pixel)
    readout_noise  : additive Gaussian (sCMOS camera readout electronics)

The confocal PSF is the product of excitation and detection PSFs, both modelled
as Gaussian with widths determined by NA, wavelength, and refractive index.
Spherical aberration from refractive-index mismatch broadens the PSF and
introduces asymmetry at deeper z-planes.

Mismatch parameters:
    pinhole_size_au         : pinhole diameter in Airy units (0.5 - 2.5 AU)
    refractive_index_mismatch : delta-n between immersion and sample (0.0 - 0.08)
    spherical_aberration_waves : peak-to-valley SA in waves (0.0 - 0.4)
    noise_level             : Poisson photon budget scale (100 - 2000 photons/pixel peak)

Phantoms:
    - Fluorescent beads at various depths (point-like emitters)
    - Branching dendrites (neuron-like filaments)
    - Nuclear staining patterns (DAPI-like, filled nuclei with varying intensity)

CPU Baseline:
    Richardson-Lucy deconvolution with estimated PSF (50 iterations).
    Expected ~22-28 dB.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/confocal_3d
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256          # 256x256 z-slice
PIXEL_SIZE_UM = 0.065     # 65 nm/pixel (typical confocal at 100x)
Z_SPACING_UM = 0.3        # 300 nm z-step between slices

# -- Optical parameters (defaults) ------------------------------------------

WAVELENGTH_EX_UM = 0.488   # excitation wavelength (488 nm, GFP)
WAVELENGTH_EM_UM = 0.525   # emission wavelength (525 nm)
NA = 1.4                   # numerical aperture (oil immersion)
N_IMMERSION = 1.515        # refractive index (oil)

# -- Mismatch ranges per tier -----------------------------------------------

SPEC = {
    "public": {
        "pinhole_size_au":            {"min": 0.8, "max": 1.5, "unit": "Airy units"},
        "refractive_index_mismatch":  {"min": 0.0, "max": 0.03, "unit": "delta-n"},
        "spherical_aberration_waves": {"min": 0.0, "max": 0.15, "unit": "waves"},
        "noise_level":                {"min": 500, "max": 2000, "unit": "peak photons"},
    },
    "dev": {
        "pinhole_size_au":            {"min": 0.6, "max": 2.0, "unit": "Airy units"},
        "refractive_index_mismatch":  {"min": 0.0, "max": 0.06, "unit": "delta-n"},
        "spherical_aberration_waves": {"min": 0.0, "max": 0.30, "unit": "waves"},
        "noise_level":                {"min": 200, "max": 2000, "unit": "peak photons"},
    },
    "hidden": {
        "pinhole_size_au":            {"min": 0.5, "max": 2.5, "unit": "Airy units"},
        "refractive_index_mismatch":  {"min": 0.0, "max": 0.08, "unit": "delta-n"},
        "spherical_aberration_waves": {"min": 0.0, "max": 0.40, "unit": "waves"},
        "noise_level":                {"min": 100, "max": 1500, "unit": "peak photons"},
    },
}


# ============================================================================
# PSF generation
# ============================================================================

def _airy_disk_radius_px() -> float:
    """First zero of the Airy disk in pixels: 0.61 * lambda_em / NA / pixel_size."""
    return 0.61 * WAVELENGTH_EM_UM / NA / PIXEL_SIZE_UM


def make_confocal_psf(
    size: int,
    pinhole_au: float,
    ri_mismatch: float,
    sa_waves: float,
) -> np.ndarray:
    """Generate a 2D confocal PSF (product of excitation and detection PSFs).

    The confocal PSF is the product of the excitation and emission PSFs. The
    pinhole clips the detection PSF, effectively multiplying it by a Gaussian
    whose width is proportional to the pinhole diameter.

    Spherical aberration and refractive-index mismatch broaden the PSF
    (modelled as additional Gaussian blur).

    Args:
        size: PSF patch size (pixels)
        pinhole_au: pinhole diameter in Airy units
        ri_mismatch: delta-n (refractive index mismatch)
        sa_waves: spherical aberration in waves (peak-to-valley)

    Returns:
        psf: (size, size) float64, normalised to sum=1
    """
    # Lateral resolution limit (Gaussian sigma in pixels)
    # Excitation PSF sigma: 0.21 * lambda_ex / NA  (Rayleigh criterion -> sigma)
    sigma_ex = 0.21 * WAVELENGTH_EX_UM / NA / PIXEL_SIZE_UM
    # Emission PSF sigma
    sigma_em = 0.21 * WAVELENGTH_EM_UM / NA / PIXEL_SIZE_UM

    # Pinhole effect: clips detection PSF. For pinhole < 1 AU, detection PSF
    # is smaller; for pinhole >> 1 AU, approaches widefield.
    # Effective detection sigma scales as: sigma_det = sigma_em * max(pinhole_au, 0.5)
    # (pinhole < 0.5 AU has negligible further improvement)
    sigma_det = sigma_em * np.clip(pinhole_au, 0.5, 5.0)

    # Confocal PSF = excitation * detection -> effective sigma is:
    # 1/sigma_conf^2 = 1/sigma_ex^2 + 1/sigma_det^2
    sigma_conf_sq = 1.0 / (1.0 / sigma_ex**2 + 1.0 / sigma_det**2)
    sigma_conf = np.sqrt(sigma_conf_sq)

    # Aberration broadening: SA and RI mismatch add extra blur
    # SA broadening ~ sa_waves * lambda / (2*pi) mapped to pixels
    sa_broadening = sa_waves * WAVELENGTH_EM_UM / PIXEL_SIZE_UM * 0.3
    ri_broadening = ri_mismatch * 50.0 * WAVELENGTH_EM_UM / PIXEL_SIZE_UM

    sigma_total = np.sqrt(sigma_conf**2 + sa_broadening**2 + ri_broadening**2)

    # Build 2D Gaussian PSF
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    psf = np.exp(-r2 / (2 * sigma_total**2))

    # Normalize
    psf /= psf.sum()
    return psf


def make_out_of_focus_psf(size: int, scale_factor: float = 3.0) -> np.ndarray:
    """Generate a wider PSF for out-of-focus light from adjacent z-planes.

    The out-of-focus contribution is modelled as a much broader Gaussian
    (the defocused PSF ring pattern averaged over multiple planes).

    Args:
        size: PSF patch size
        scale_factor: how much wider than the confocal PSF

    Returns:
        psf: (size, size) float64, normalised to sum=1
    """
    sigma_oof = (_airy_disk_radius_px() * scale_factor)
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    psf = np.exp(-r2 / (2 * sigma_oof**2))
    psf /= psf.sum()
    return psf


# ============================================================================
# Phantom generators (cell-like structures)
# ============================================================================

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 500
) -> np.ndarray:
    """Smooth curve via cubic interpolation of control points.

    Args:
        control_pts: (N, 2) array of (y, x) control points
        n_interp: number of output points

    Returns:
        (n_interp, 2) array of (y, x) interpolated points
    """
    from scipy.interpolate import CubicSpline

    n = len(control_pts)
    if n < 2:
        return control_pts
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def _draw_thick_curve(
    canvas: np.ndarray,
    curve: np.ndarray,
    thickness: float,
    intensity: float,
) -> None:
    """Draw a thick curve on a canvas using Gaussian cross-section."""
    H, W = canvas.shape
    for py, px in curve:
        iy, ix = int(round(py)), int(round(px))
        r = int(np.ceil(thickness * 3))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        d2 = (yy - py)**2 + (xx - px)**2
        canvas[y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * thickness**2))


def make_bead_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Fluorescent beads at various depths.

    Beads appear as bright spots of varying size (depending on axial position
    relative to the focal plane). Larger = more defocused (further from focus).
    """
    x = np.zeros((H, W), dtype=np.float64)
    n_beads = rng.integers(30, 80)

    for _ in range(n_beads):
        cy = rng.uniform(10, H - 10)
        cx = rng.uniform(10, W - 10)
        # Depth determines apparent size and brightness
        z_offset = rng.uniform(-2.0, 2.0)  # um from focal plane
        sigma = max(0.8, 0.8 + abs(z_offset) * 1.2)
        intensity = rng.uniform(0.3, 1.0) * np.exp(-z_offset**2 / 2.0)

        iy, ix = int(round(cy)), int(round(cx))
        r = int(np.ceil(sigma * 4))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = intensity * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
        x[y0:y1, x0:x1] += g

    return x


def make_dendrite_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Branching dendrites (neuron-like filaments).

    Primary dendrite branches from a cell body (soma), with secondary and
    tertiary branches splitting off at various angles. The soma is a bright
    circular region.
    """
    x = np.zeros((H, W), dtype=np.float64)

    # Soma (cell body)
    soma_y = rng.uniform(H * 0.3, H * 0.7)
    soma_x = rng.uniform(W * 0.3, W * 0.7)
    soma_r = rng.uniform(8, 18)
    yy, xx = np.mgrid[:H, :W]
    soma_mask = ((yy - soma_y)**2 + (xx - soma_x)**2) < soma_r**2
    x[soma_mask] += rng.uniform(0.5, 0.9)

    # Primary dendrites
    n_primary = rng.integers(3, 7)
    for _ in range(n_primary):
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.15, H * 0.4)
        n_ctrl = rng.integers(4, 8)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / (n_ctrl - 1)
            ctrl[j, 0] = soma_y + length * t * np.sin(angle) + rng.normal(0, 8)
            ctrl[j, 1] = soma_x + length * t * np.cos(angle) + rng.normal(0, 8)

        curve = _smooth_curve_points(ctrl, n_interp=400)
        valid = (
            (curve[:, 0] >= 1) & (curve[:, 0] < H - 1)
            & (curve[:, 1] >= 1) & (curve[:, 1] < W - 1)
        )
        curve = curve[valid]
        if len(curve) < 5:
            continue

        # Thickness tapers along dendrite
        thickness = rng.uniform(1.5, 3.0)
        intensity = rng.uniform(0.4, 0.8)
        _draw_thick_curve(x, curve, thickness, intensity)

        # Secondary branches
        n_secondary = rng.integers(1, 4)
        for _ in range(n_secondary):
            if len(curve) < 20:
                break
            branch_idx = rng.integers(len(curve) // 3, len(curve))
            branch_angle = angle + rng.uniform(-np.pi / 3, np.pi / 3)
            branch_len = rng.uniform(H * 0.05, H * 0.15)
            n_ctrl2 = rng.integers(3, 5)
            ctrl2 = np.zeros((n_ctrl2, 2))
            for j in range(n_ctrl2):
                t = j / (n_ctrl2 - 1)
                ctrl2[j, 0] = (curve[branch_idx, 0]
                                + branch_len * t * np.sin(branch_angle)
                                + rng.normal(0, 4))
                ctrl2[j, 1] = (curve[branch_idx, 1]
                                + branch_len * t * np.cos(branch_angle)
                                + rng.normal(0, 4))
            curve2 = _smooth_curve_points(ctrl2, n_interp=200)
            valid2 = (
                (curve2[:, 0] >= 1) & (curve2[:, 0] < H - 1)
                & (curve2[:, 1] >= 1) & (curve2[:, 1] < W - 1)
            )
            curve2 = curve2[valid2]
            if len(curve2) > 3:
                _draw_thick_curve(x, curve2, thickness * 0.6, intensity * 0.7)

    return x


def make_nuclear_stain_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Nuclear staining patterns (DAPI-like).

    Multiple cell nuclei with varying size, shape (slight ellipticity),
    and internal intensity variations (chromatin texture).
    """
    x = np.zeros((H, W), dtype=np.float64)

    n_nuclei = rng.integers(5, 20)
    yy, xx = np.mgrid[:H, :W]

    for _ in range(n_nuclei):
        cy = rng.uniform(15, H - 15)
        cx = rng.uniform(15, W - 15)
        # Semi-axes
        a = rng.uniform(8, 25)
        b = rng.uniform(0.6, 1.0) * a  # slight ellipticity
        angle = rng.uniform(0, np.pi)
        intensity = rng.uniform(0.3, 1.0)

        # Rotated ellipse
        dy = yy - cy
        dx = xx - cx
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        r_rot = ((dy * cos_a + dx * sin_a) / a)**2 + \
                ((-dy * sin_a + dx * cos_a) / b)**2
        nucleus_mask = r_rot < 1.0

        # Internal chromatin texture: smooth random pattern
        texture = rng.standard_normal((H, W))
        texture = gaussian_filter(texture, sigma=rng.uniform(2.0, 5.0))
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

        # Smooth edge falloff
        falloff = np.exp(-np.clip(r_rot - 0.7, 0, None) * 10)
        contribution = intensity * falloff * (0.5 + 0.5 * texture)
        x += contribution * nucleus_mask

    return x


PHANTOM_FNS = [make_bead_phantom, make_dendrite_phantom, make_nuclear_stain_phantom]
PHANTOM_NAMES = ["beads", "dendrites", "nuclei"]


# ============================================================================
# Confocal 3D forward model
# ============================================================================

def confocal_forward(
    x_true: np.ndarray,
    pinhole_au: float,
    ri_mismatch: float,
    sa_waves: float,
    noise_level: float,
    rng: np.random.Generator,
    readout_noise_std: float = 3.0,
    bg_level: float = 5.0,
    oof_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Confocal 3D forward model for a single z-slice.

    y = Poisson(PSF_confocal * x + out_of_focus_blur + bg) + readout_noise

    Args:
        x_true: (H, W) ground truth fluorescence (normalised to [0,1])
        pinhole_au: pinhole diameter in Airy units
        ri_mismatch: refractive index mismatch (delta-n)
        sa_waves: spherical aberration in waves
        noise_level: peak photon count (scales signal amplitude)
        rng: random generator
        readout_noise_std: sCMOS readout noise std (electrons)
        bg_level: background photons per pixel
        oof_fraction: fraction of signal from out-of-focus planes

    Returns:
        y: (H, W) float32 noisy measurement
        H_ideal: (H, W) float32 noiseless blurred image (for reference)
    """
    H, W = x_true.shape
    psf_size = 31  # odd-sized PSF kernel

    # Confocal PSF
    psf_conf = make_confocal_psf(psf_size, pinhole_au, ri_mismatch, sa_waves)

    # Out-of-focus PSF (wider, from adjacent z-planes)
    oof_scale = 2.5 + ri_mismatch * 20 + sa_waves * 3.0
    psf_oof = make_out_of_focus_psf(psf_size, scale_factor=oof_scale)

    # Scale ground truth to photon counts
    x_photons = x_true.astype(np.float64) * noise_level

    # In-focus signal: convolve with confocal PSF
    in_focus = fftconvolve(x_photons, psf_conf, mode="same")

    # Out-of-focus contribution: wider blur of a slightly different "plane"
    # Simulate as blurred version of ground truth with random depth variation
    oof_signal = fftconvolve(x_photons, psf_oof, mode="same") * oof_fraction

    # Total ideal signal (noiseless)
    ideal = np.maximum(in_focus + oof_signal + bg_level, 0.01)
    H_ideal = ideal.astype(np.float32)

    # Poisson noise (shot noise from photon counting)
    y = rng.poisson(np.maximum(ideal, 0.01)).astype(np.float64)

    # Readout noise (Gaussian, sCMOS camera)
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal


# ============================================================================
# CPU Baseline: Richardson-Lucy deconvolution
# ============================================================================

def richardson_lucy_deconv(
    y: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 50,
    clip: bool = True,
) -> np.ndarray:
    """Richardson-Lucy deconvolution (classic iterative algorithm).

    The RL algorithm iteratively estimates x from y = PSF * x + noise
    using the multiplicative update:
        x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k)))

    Args:
        y: (H, W) noisy measurement
        psf: (K, K) point spread function (normalised)
        n_iter: number of iterations
        clip: clip negative values

    Returns:
        recon: (H, W) float32 reconstruction
    """
    y64 = y.astype(np.float64)
    psf64 = psf.astype(np.float64)
    # Flipped PSF for the adjoint operation
    psf_flip = psf64[::-1, ::-1]

    # Initialise with uniform estimate
    x_est = np.ones_like(y64) * max(y64.mean(), 1e-3)

    eps = 1e-8
    for _ in range(n_iter):
        # Forward: convolve estimate with PSF
        y_est = fftconvolve(x_est, psf64, mode="same")
        y_est = np.maximum(y_est, eps)

        # Ratio
        ratio = y64 / y_est

        # Back-project ratio
        correction = fftconvolve(ratio, psf_flip, mode="same")

        # Multiplicative update
        x_est *= correction

        if clip:
            x_est = np.maximum(x_est, 0)

    return x_est.astype(np.float32)


def baseline_reconstruct(
    y: np.ndarray,
    pinhole_au: float = 1.0,
) -> np.ndarray:
    """CPU baseline: Richardson-Lucy with estimated confocal PSF.

    Uses a slightly mismatched PSF (no aberration knowledge) to simulate
    a realistic baseline.

    Args:
        y: noisy measurement
        pinhole_au: estimated pinhole size

    Returns:
        recon: (H, W) float32, normalised to [0,1]
    """
    # Estimated PSF (assume no aberrations -- mild mismatch)
    psf_est = make_confocal_psf(
        size=31,
        pinhole_au=pinhole_au,
        ri_mismatch=0.0,   # assume no RI mismatch
        sa_waves=0.0,       # assume no SA
    )

    recon = richardson_lucy_deconv(y, psf_est, n_iter=50)

    # Normalise to [0, 1]
    rmin, rmax = recon.min(), recon.max()
    if rmax - rmin > 1e-8:
        recon = (recon - rmin) / (rmax - rmin)
    else:
        recon = np.zeros_like(recon)

    return recon.astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

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


# ============================================================================
# Image helpers
# ============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    a = arr.astype(np.float64)
    if percentile_clip and a.max() > 0:
        lo, hi = np.percentile(a[a > 0], [1, 99]) if np.any(a > 0) else (0, 1)
        a = np.clip(a, lo, hi)
    Image.fromarray(
        np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# ============================================================================
# Phantom pool per tier
# ============================================================================

def generate_phantoms(
    n: int, seed_offset: int,
) -> list[tuple[np.ndarray, str]]:
    """Generate phantom ground truths.

    Returns list of (x_true, scene_name).
    """
    phantoms = []
    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]

        phantom_rng = np.random.default_rng(seed_offset + i)
        x_raw = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        # Normalise to [0, 1]
        xmin, xmax = x_raw.min(), x_raw.max()
        if xmax - xmin > 1e-8:
            x_true = ((x_raw - xmin) / (xmax - xmin)).astype(np.float32)
        else:
            x_true = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((x_true, scene_name))

    return phantoms


# ============================================================================
# Tier generation
# ============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        if k == "noise_level":
            val = round(val)
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the Confocal 3D benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"confocal_3d_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Confocal 3D Microscopy benchmark -- {tier} tier "
            f"(z-slice from 3D confocal volume)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF_confocal * x + out_of_focus_blur + bg) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_um": PIXEL_SIZE_UM,
            "z_spacing_um": Z_SPACING_UM,
            "wavelength_ex_um": WAVELENGTH_EX_UM,
            "wavelength_em_um": WAVELENGTH_EM_UM,
            "NA": NA,
            "n_immersion": N_IMMERSION,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis.copy()

            # Forward model
            y, H_ideal = confocal_forward(
                x_true,
                pinhole_au=mis["pinhole_size_au"],
                ri_mismatch=mis["refractive_index_mismatch"],
                sa_waves=mis["spherical_aberration_waves"],
                noise_level=mis["noise_level"],
                rng=rng,
            )

            # CPU baseline: Richardson-Lucy
            recon = baseline_reconstruct(
                y,
                pinhole_au=mis["pinhole_size_au"],
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
            grp.create_dataset("reconstruction_baseline", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
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
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"pinhole={mis['pinhole_size_au']:.2f} AU  "
                  f"RI_mis={mis['refractive_index_mismatch']:.4f}  "
                  f"SA={mis['spherical_aberration_waves']:.3f}  "
                  f"photons={mis['noise_level']}")

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


# ============================================================================
# Gallery image generation
# ============================================================================

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "confocal_3d"
    )

    h5_path = BENCHMARK_DIR / "public" / "confocal_3d_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 samples: indices 0,1,2 (beads, dendrites, nuclei) + 3 (beads again)
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
            y_meas = grp["y"][:]
            H_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]

            # gt.png -- ground truth fluorescence
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy confocal acquisition
            _save_png(y_meas, scene_dir / "measurement_I.png",
                      percentile_clip=True)

            # measurement_II.png -- noiseless ideal (H_ideal)
            _save_png(H_ideal, scene_dir / "measurement_II.png",
                      percentile_clip=True)

            # recon_I.png -- Richardson-Lucy baseline
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- |GT - recon| difference map
            diff = np.abs(x_true.astype(np.float64) - recon.astype(np.float64))
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Confocal 3D Microscopy Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel size: {PIXEL_SIZE_UM} um, Z-spacing: {Z_SPACING_UM} um")
    print(f"Optics: NA={NA}, lambda_ex={WAVELENGTH_EX_UM*1000:.0f} nm, "
          f"lambda_em={WAVELENGTH_EM_UM*1000:.0f} nm\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0)
    generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000)
    generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000)
    generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Confocal 3D benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
