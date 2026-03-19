#!/usr/bin/env python3
"""Generate Light-Sheet Fluorescence Microscopy (LSFM / SPIM) benchmark dataset.

Forward model:
    y = Poisson(PSF_det * (S(z) * x) + scatter + bg) + readout_noise

where:
    x            : 2D fluorescence ground truth (cleared tissue/embryo section)
    S(z)         : light-sheet illumination profile (Gaussian beam, non-uniform
                   thickness -> striping artifacts at edges)
    PSF_det      : detection PSF (widefield-like Gaussian, sigma ~2-3 px)
    scatter      : tissue scattering (depth-dependent exponential attenuation)
    bg           : out-of-focus background fluorescence
    readout_noise: Gaussian camera readout noise (sCMOS)

Mismatch parameters:
    sheet_thickness    : light-sheet waist in pixels (3-8 range)
    sheet_uniformity   : how uniform sheet is across FOV (0.5-1.0; 1=perfect)
    scattering_coeff   : tissue scattering coefficient (0.01-0.10 per pixel)
    noise_level        : Poisson noise scaling / peak photon count (100-2000)

Phantoms:
    Cleared tissue/embryo sections with:
    - Sparse fluorescent nuclei (round bright spots)
    - Vasculature networks (branching tubular structures)
    - Developing organ structures (layered tissue regions)
    - Combined tissue sections (nuclei + vasculature + organ layers)

CPU Baseline:
    Stripe removal (Fourier notch filter) + Richardson-Lucy deconvolution.
    Expected: ~22-28 dB PSNR.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/lightsheet
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256
PIXEL_SIZE_UM = 0.65  # ~0.65 um/pixel (typical LSFM)

# -- Mismatch ranges per tier ------------------------------------------------

SPEC = {
    "public": {
        "sheet_thickness":   {"min": 3.0, "max": 5.0, "unit": "pixels"},
        "sheet_uniformity":  {"min": 0.7, "max": 1.0, "unit": "fraction"},
        "scattering_coeff":  {"min": 0.01, "max": 0.05, "unit": "1/pixel"},
        "noise_level":       {"min": 500, "max": 2000, "unit": "peak_photons"},
    },
    "dev": {
        "sheet_thickness":   {"min": 3.0, "max": 6.5, "unit": "pixels"},
        "sheet_uniformity":  {"min": 0.6, "max": 1.0, "unit": "fraction"},
        "scattering_coeff":  {"min": 0.01, "max": 0.08, "unit": "1/pixel"},
        "noise_level":       {"min": 300, "max": 2000, "unit": "peak_photons"},
    },
    "hidden": {
        "sheet_thickness":   {"min": 4.0, "max": 8.0, "unit": "pixels"},
        "sheet_uniformity":  {"min": 0.5, "max": 0.9, "unit": "fraction"},
        "scattering_coeff":  {"min": 0.02, "max": 0.10, "unit": "1/pixel"},
        "noise_level":       {"min": 100, "max": 1500, "unit": "peak_photons"},
    },
}


# ---------------------------------------------------------------------------
# Phantom generators: cleared tissue / embryo sections
# ---------------------------------------------------------------------------

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 500,
) -> np.ndarray:
    """Generate smooth curve via cubic interpolation of control points."""
    from scipy.interpolate import CubicSpline

    n = len(control_pts)
    if n < 3:
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
    """Draw a thick curve on canvas by placing Gaussian blobs along it."""
    H, W = canvas.shape
    sigma = max(thickness / 2.0, 0.5)
    r = int(np.ceil(3 * sigma))
    for py, px in curve:
        iy, ix = int(round(py)), int(round(px))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
        canvas[y0:y1, x0:x1] += intensity * g


def make_nuclei_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sparse fluorescent nuclei -- round bright spots of varying size."""
    x = np.zeros((H, W), dtype=np.float64)
    n_nuclei = rng.integers(40, 120)
    for _ in range(n_nuclei):
        cy = rng.uniform(10, H - 10)
        cx = rng.uniform(10, W - 10)
        radius = rng.uniform(2.0, 6.0)
        intensity = rng.uniform(0.4, 1.0)
        sigma = radius / 2.0
        r = int(np.ceil(3 * sigma)) + 1
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        x[y0:y1, x0:x1] += intensity * g
    return x


def make_vasculature_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Branching vasculature network -- tubular structures with bifurcations."""
    x = np.zeros((H, W), dtype=np.float64)
    n_main = rng.integers(3, 7)

    for _ in range(n_main):
        # Main vessel: curved path across image
        n_ctrl = rng.integers(4, 8)
        ctrl = np.zeros((n_ctrl, 2))
        start_y = rng.uniform(0, H)
        start_x = rng.uniform(0, W * 0.3)
        end_y = rng.uniform(0, H)
        end_x = rng.uniform(W * 0.7, W)
        for j in range(n_ctrl):
            t = j / (n_ctrl - 1)
            ctrl[j, 0] = start_y + (end_y - start_y) * t + rng.uniform(-30, 30)
            ctrl[j, 1] = start_x + (end_x - start_x) * t + rng.uniform(-20, 20)
        ctrl = np.clip(ctrl, 2, [H - 3, W - 3])
        curve = _smooth_curve_points(ctrl, n_interp=600)
        valid = (curve[:, 0] >= 1) & (curve[:, 0] < H - 1) & \
                (curve[:, 1] >= 1) & (curve[:, 1] < W - 1)
        curve = curve[valid]
        if len(curve) < 20:
            continue
        thickness = rng.uniform(1.5, 4.0)
        intensity = rng.uniform(0.5, 1.0)
        _draw_thick_curve(x, curve, thickness, intensity)

        # Branches
        n_branches = rng.integers(1, 5)
        for _ in range(n_branches):
            if len(curve) < 10:
                break
            branch_start_idx = rng.integers(len(curve) // 4, 3 * len(curve) // 4)
            branch_start = curve[branch_start_idx]
            n_bctrl = rng.integers(3, 6)
            bctrl = np.zeros((n_bctrl, 2))
            angle = rng.uniform(0, 2 * np.pi)
            branch_len = rng.uniform(30, 80)
            for j in range(n_bctrl):
                t = j / (n_bctrl - 1)
                bctrl[j, 0] = branch_start[0] + branch_len * t * np.sin(angle) + rng.uniform(-10, 10)
                bctrl[j, 1] = branch_start[1] + branch_len * t * np.cos(angle) + rng.uniform(-10, 10)
            bctrl = np.clip(bctrl, 2, [H - 3, W - 3])
            bcurve = _smooth_curve_points(bctrl, n_interp=300)
            valid_b = (bcurve[:, 0] >= 1) & (bcurve[:, 0] < H - 1) & \
                      (bcurve[:, 1] >= 1) & (bcurve[:, 1] < W - 1)
            bcurve = bcurve[valid_b]
            if len(bcurve) > 10:
                _draw_thick_curve(x, bcurve, thickness * 0.6, intensity * 0.8)

    return x


def make_organ_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Developing organ structures -- layered tissue with boundaries."""
    x = np.zeros((H, W), dtype=np.float64)

    # Generate 3-6 tissue layers (roughly horizontal with curvature)
    n_layers = rng.integers(3, 7)
    layer_positions = np.sort(rng.uniform(H * 0.1, H * 0.9, n_layers))

    for i, layer_y in enumerate(layer_positions):
        intensity = rng.uniform(0.3, 1.0)
        layer_width = rng.uniform(3, 12)

        # Create curved layer boundary
        n_pts = 20
        xs = np.linspace(0, W - 1, n_pts)
        ys = layer_y + rng.uniform(-8, 8, n_pts)
        # Smooth the boundary
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(xs, ys, bc_type="natural")
        x_fine = np.arange(W)
        y_boundary = cs(x_fine)

        # Fill tissue layer region
        sigma = layer_width / 2.5
        for col in range(W):
            center_y = y_boundary[col]
            y_lo = max(0, int(center_y - 3 * sigma))
            y_hi = min(H, int(center_y + 3 * sigma) + 1)
            if y_hi <= y_lo:
                continue
            yy = np.arange(y_lo, y_hi)
            profile = np.exp(-((yy - center_y) ** 2) / (2 * sigma ** 2))
            x[y_lo:y_hi, col] += intensity * profile

    # Add some scattered bright cells within layers
    n_cells = rng.integers(10, 40)
    for _ in range(n_cells):
        cy = rng.uniform(10, H - 10)
        cx = rng.uniform(10, W - 10)
        r = rng.uniform(1.5, 3.5)
        amp = rng.uniform(0.2, 0.6)
        sigma = r / 2.0
        rad = int(np.ceil(3 * sigma))
        y0, y1 = max(0, int(cy) - rad), min(H, int(cy) + rad + 1)
        x0, x1 = max(0, int(cx) - rad), min(W, int(cx) + rad + 1)
        if y1 > y0 and x1 > x0:
            yy = np.arange(y0, y1)[:, None]
            xx = np.arange(x0, x1)[None, :]
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            x[y0:y1, x0:x1] += amp * g

    return x


def make_combined_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Combined tissue section: nuclei + vasculature + organ layers."""
    organ = make_organ_phantom(H, W, rng) * rng.uniform(0.3, 0.6)
    vasc = make_vasculature_phantom(H, W, rng) * rng.uniform(0.3, 0.7)
    nuclei = make_nuclei_phantom(H, W, rng) * rng.uniform(0.4, 0.8)
    return organ + vasc + nuclei


PHANTOM_FNS = [
    make_nuclei_phantom,
    make_vasculature_phantom,
    make_organ_phantom,
    make_combined_phantom,
]

PHANTOM_NAMES = ["nuclei", "vasculature", "organ", "combined"]


# ---------------------------------------------------------------------------
# Light-sheet forward model
# ---------------------------------------------------------------------------

def _make_sheet_profile(
    H: int, W: int,
    sheet_thickness: float,
    sheet_uniformity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate light-sheet illumination profile S(z).

    The sheet illumination is a Gaussian beam along y (detection axis).
    At the beam waist (center of FOV) the sheet is thinnest; at edges
    it is thicker, producing striping artifacts.

    Args:
        H, W: image dimensions (H = detection axis, W = illumination axis)
        sheet_thickness: waist thickness in pixels (sigma of Gaussian)
        sheet_uniformity: 1.0 = perfectly uniform, lower = more striping

    Returns:
        S: (H, W) illumination profile, values in [0, 1]
    """
    # Beam propagation: thickness varies as sigma(x) = waist * sqrt(1 + (x/x_R)^2)
    # x_R (Rayleigh range) controls how fast the beam diverges
    x_R = W * sheet_uniformity * 0.5  # higher uniformity -> longer Rayleigh range
    x_coords = np.arange(W) - W / 2.0
    sigma_x = sheet_thickness * np.sqrt(1.0 + (x_coords / max(x_R, 1.0)) ** 2)

    # Sheet profile: Gaussian in the detection (y) axis at each x position
    y_coords = np.arange(H) - H / 2.0
    S = np.zeros((H, W), dtype=np.float64)
    for j in range(W):
        S[:, j] = np.exp(-(y_coords ** 2) / (2 * sigma_x[j] ** 2))

    # Add mild intensity variations (striping from scattering in sheet path)
    n_stripes = rng.integers(3, 10)
    stripe_pattern = np.ones(W, dtype=np.float64)
    for _ in range(n_stripes):
        pos = rng.uniform(0, W)
        stripe_width = rng.uniform(5, 30)
        stripe_depth = rng.uniform(0.05, 0.3) * (1.0 - sheet_uniformity)
        stripe_pattern -= stripe_depth * np.exp(
            -(np.arange(W) - pos) ** 2 / (2 * stripe_width ** 2)
        )
    stripe_pattern = np.clip(stripe_pattern, 0.3, 1.0)
    S *= stripe_pattern[None, :]

    # Normalize to [0, 1]
    S_max = S.max()
    if S_max > 0:
        S /= S_max

    return S


def _make_detection_psf(sigma: float, size: int = 0) -> np.ndarray:
    """Create 2D Gaussian detection PSF.

    Args:
        sigma: PSF standard deviation in pixels
        size: kernel size (auto if 0)

    Returns:
        psf: (K, K) normalized PSF kernel
    """
    if size == 0:
        size = int(np.ceil(6 * sigma)) | 1  # ensure odd
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    return psf


def _apply_scattering(
    img: np.ndarray,
    scattering_coeff: float,
) -> np.ndarray:
    """Apply depth-dependent tissue scattering (exponential attenuation).

    In light-sheet microscopy, deeper tissue scatters/absorbs more of both
    excitation and emission light. We model this as attenuation along the
    illumination axis (columns = x direction).

    Args:
        img: (H, W) input image
        scattering_coeff: attenuation coefficient per pixel

    Returns:
        attenuated: (H, W) scattered image
    """
    H, W = img.shape
    # Attenuation increases with depth along the illumination axis
    x_coords = np.arange(W, dtype=np.float64)
    # Light enters from left; attenuation increases with x
    attenuation = np.exp(-scattering_coeff * x_coords)
    return img * attenuation[None, :]


def lightsheet_forward(
    x_true: np.ndarray,
    sheet_thickness: float,
    sheet_uniformity: float,
    scattering_coeff: float,
    noise_level: float,
    rng: np.random.Generator,
    det_psf_sigma: float = 2.5,
    bg_fraction: float = 0.02,
    readout_noise_std: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Light-sheet fluorescence microscopy forward model.

    y = Poisson(PSF_det * (S(z) * x) + scatter + bg) + readout_noise

    Args:
        x_true: (H, W) ground truth fluorescence (normalized to [0, 1])
        sheet_thickness: light-sheet waist sigma in pixels
        sheet_uniformity: beam uniformity (0.5 = bad, 1.0 = perfect)
        scattering_coeff: tissue scattering coefficient (per pixel)
        noise_level: peak photon count for Poisson noise
        rng: random number generator
        det_psf_sigma: detection PSF sigma in pixels
        bg_fraction: background fluorescence fraction of peak signal
        readout_noise_std: sCMOS readout noise standard deviation

    Returns:
        y: (H, W) noisy measurement
        H_ideal: (H, W) noiseless ideal image (for reference)
    """
    H, W = x_true.shape

    # Step 1: Light-sheet illumination
    S = _make_sheet_profile(H, W, sheet_thickness, sheet_uniformity, rng)
    illuminated = S * x_true

    # Step 2: Tissue scattering (depth-dependent attenuation)
    scattered = _apply_scattering(illuminated, scattering_coeff)

    # Step 3: Detection PSF convolution (widefield-like blurring)
    psf = _make_detection_psf(det_psf_sigma)
    blurred = fftconvolve(scattered, psf, mode="same")
    blurred = np.maximum(blurred, 0)

    # Step 4: Scale to photon counts
    blurred_max = blurred.max()
    if blurred_max > 0:
        signal_photons = blurred * (noise_level / blurred_max)
    else:
        signal_photons = blurred

    # Step 5: Add background fluorescence
    background = bg_fraction * noise_level
    # Spatially varying background (out-of-focus fluorescence)
    bg_map = background * (0.5 + 0.5 * gaussian_filter(
        rng.uniform(0.5, 1.5, (H, W)), sigma=20
    ))

    # Add scattering-induced background (depth-dependent)
    scatter_bg = scattering_coeff * noise_level * 0.1 * rng.uniform(0.8, 1.2, (H, W))
    scatter_bg = gaussian_filter(scatter_bg, sigma=15)

    total_signal = signal_photons + bg_map + scatter_bg

    # Ideal (noiseless) measurement
    H_ideal = total_signal.copy().astype(np.float32)

    # Step 6: Poisson noise (shot noise from photon detection)
    total_signal = np.maximum(total_signal, 0.01)
    y = rng.poisson(total_signal).astype(np.float64)

    # Step 7: sCMOS readout noise
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal


# ---------------------------------------------------------------------------
# CPU Baseline: Stripe removal + Richardson-Lucy deconvolution
# ---------------------------------------------------------------------------

def _remove_stripes(y: np.ndarray, sigma_notch: float = 3.0) -> np.ndarray:
    """Remove horizontal stripe artifacts via Fourier notch filtering.

    Stripes appear as strong horizontal components (low ky, varying kx).
    We attenuate these in the Fourier domain with a narrow notch filter.
    """
    H, W = y.shape
    Y = np.fft.fft2(y)
    Y_shifted = np.fft.fftshift(Y)

    # Create notch filter: suppress the horizontal band (ky near 0, all kx)
    ky = np.arange(H) - H // 2
    kx = np.arange(W) - W // 2
    KY, KX = np.meshgrid(ky, kx, indexing="ij")

    # Notch: attenuate components where |ky| is small but kx is not zero
    # This targets horizontal stripe patterns
    notch = 1.0 - np.exp(-(KY ** 2) / (2 * sigma_notch ** 2))
    # Don't touch DC component
    notch[H // 2, W // 2] = 1.0
    # Also preserve low-frequency content (don't remove overall gradients)
    low_freq_mask = np.exp(-(KX ** 2 + KY ** 2) / (2 * 10 ** 2))
    notch = np.maximum(notch, low_freq_mask)

    Y_filtered = Y_shifted * notch
    result = np.real(np.fft.ifft2(np.fft.ifftshift(Y_filtered)))
    return np.maximum(result, 0)


def _richardson_lucy(
    y: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 30,
) -> np.ndarray:
    """Richardson-Lucy deconvolution.

    Iterative ML estimate for Poisson noise model:
        x^{k+1} = x^k * (psf_flipped * (y / (psf * x^k)))
    """
    # Normalize input
    y64 = y.astype(np.float64)
    y_max = y64.max()
    if y_max > 0:
        y64 /= y_max

    psf64 = psf.astype(np.float64)
    psf_flip = psf64[::-1, ::-1]

    # Initialize with uniform image
    x_est = np.full_like(y64, y64.mean())
    x_est = np.maximum(x_est, 1e-10)

    eps = 1e-10
    for _ in range(n_iter):
        # Forward: convolve estimate with PSF
        y_est = fftconvolve(x_est, psf64, mode="same")
        y_est = np.maximum(y_est, eps)

        # Ratio
        ratio = y64 / y_est

        # Back-project
        correction = fftconvolve(ratio, psf_flip, mode="same")

        # Update
        x_est = x_est * correction
        x_est = np.maximum(x_est, eps)

    return x_est.astype(np.float64)


def baseline_reconstruct(
    y: np.ndarray,
    det_psf_sigma: float = 2.5,
    rl_iterations: int = 30,
) -> np.ndarray:
    """CPU baseline: stripe removal + Richardson-Lucy deconvolution.

    Pipeline:
        1. Stripe removal via Fourier notch filtering
        2. Background subtraction (rolling ball / uniform filter)
        3. Richardson-Lucy deconvolution with estimated detection PSF
        4. Normalize to [0, 1]

    Args:
        y: (H, W) noisy measurement
        det_psf_sigma: estimated detection PSF sigma
        rl_iterations: number of RL iterations

    Returns:
        recon: (H, W) float32 reconstruction
    """
    # Step 1: Stripe removal
    destriped = _remove_stripes(y.astype(np.float64), sigma_notch=2.5)

    # Step 2: Background subtraction (large-scale background)
    bg_est = uniform_filter(destriped, size=50)
    destriped_sub = np.maximum(destriped - bg_est * 0.5, 0)

    # Step 3: Richardson-Lucy deconvolution
    psf = _make_detection_psf(det_psf_sigma)
    recon = _richardson_lucy(destriped_sub, psf, n_iter=rl_iterations)

    # Normalize to [0, 1]
    r_max = recon.max()
    if r_max > 0:
        recon /= r_max

    return recon.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    # Windowed SSIM (11x11 uniform window, per convention)
    from scipy.ndimage import uniform_filter as _uf
    w = 11
    mu_x = _uf(gt64, size=w)
    mu_y = _uf(recon64, size=w)
    mu_x2 = _uf(gt64 ** 2, size=w)
    mu_y2 = _uf(recon64 ** 2, size=w)
    mu_xy = _uf(gt64 * recon64, size=w)
    var_x = mu_x2 - mu_x ** 2
    var_y = mu_y2 - mu_y ** 2
    cov_xy = mu_xy - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    a = arr.astype(np.float64).copy()
    if percentile_clip and a.max() > 0:
        nonzero = a[a > 0]
        if len(nonzero) > 0:
            lo, hi = np.percentile(nonzero, [1, 99])
            a = np.clip(a, lo, hi)
    Image.fromarray(
        np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# ---------------------------------------------------------------------------
# Phantom pool for each tier
# ---------------------------------------------------------------------------

def generate_phantoms(
    n: int, seed_offset: int,
) -> list[tuple[np.ndarray, str]]:
    """Generate phantom set for a tier.

    Returns list of (x_true, scene_name).
    """
    phantoms = []
    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]

        phantom_rng = np.random.default_rng(seed_offset + i)
        x_raw = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        # Normalize to [0, 1]
        x_max = x_raw.max()
        if x_max > 0:
            x_true = (x_raw / x_max).astype(np.float32)
        else:
            x_true = x_raw.astype(np.float32)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((x_true, scene_name))

    return phantoms


# ---------------------------------------------------------------------------
# Tier generation
# ---------------------------------------------------------------------------

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
) -> dict:
    """Generate one tier of the light-sheet benchmark.

    Returns dict of per-sample metrics for summary.
    """
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"lightsheet_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Light-Sheet Fluorescence Microscopy benchmark -- {tier} tier"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF_det * (S(z) * x) + scatter + bg) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_um": PIXEL_SIZE_UM,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis.copy()

            # Detection PSF sigma (varies slightly)
            det_psf_sigma = rng.uniform(2.0, 3.0)

            # Forward model
            y, H_ideal = lightsheet_forward(
                x_true,
                sheet_thickness=mis["sheet_thickness"],
                sheet_uniformity=mis["sheet_uniformity"],
                scattering_coeff=mis["scattering_coeff"],
                noise_level=mis["noise_level"],
                rng=rng,
                det_psf_sigma=det_psf_sigma,
                bg_fraction=rng.uniform(0.01, 0.05),
                readout_noise_std=rng.uniform(3.0, 8.0),
            )

            # CPU baseline reconstruction
            recon = baseline_reconstruct(
                y,
                det_psf_sigma=det_psf_sigma,
                rl_iterations=30,
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
                "det_psf_sigma": det_psf_sigma,
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
                    "det_psf_sigma": det_psf_sigma,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"sheet_th={mis['sheet_thickness']:.2f}  "
                  f"unif={mis['sheet_uniformity']:.2f}  "
                  f"scatter={mis['scattering_coeff']:.3f}  "
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

    return {"mean_psnr": mean_psnr, "mean_ssim": mean_ssim,
            "psnrs": all_psnrs, "ssims": all_ssims}


# ---------------------------------------------------------------------------
# Gallery image generation
# ---------------------------------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "lightsheet"
    )

    h5_path = BENCHMARK_DIR / "public" / "lightsheet_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: nuclei, vasculature, organ, combined
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
            h_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]

            # gt.png -- ground truth fluorescence
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy light-sheet image
            _save_png(y_meas, scene_dir / "measurement_I.png", percentile_clip=True)

            # measurement_II.png -- ideal (noiseless) image
            _save_png(h_ideal, scene_dir / "measurement_II.png", percentile_clip=True)

            # recon_I.png -- baseline RL reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true.astype(np.float64) - recon.astype(np.float64))
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Light-Sheet Fluorescence Microscopy Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, pixel size: {PIXEL_SIZE_UM} um\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0)
    pub_metrics = generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000)
    dev_metrics = generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000)
    hid_metrics = generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    # Summary
    print(f"\n{'=' * 68}")
    print("Light-Sheet Fluorescence Microscopy benchmark generation complete!")
    print(f"  Public : PSNR={pub_metrics['mean_psnr']:.2f} dB, "
          f"SSIM={pub_metrics['mean_ssim']:.3f}")
    print(f"  Dev    : PSNR={dev_metrics['mean_psnr']:.2f} dB, "
          f"SSIM={dev_metrics['mean_ssim']:.3f}")
    print(f"  Hidden : PSNR={hid_metrics['mean_psnr']:.2f} dB, "
          f"SSIM={hid_metrics['mean_ssim']:.3f}")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
