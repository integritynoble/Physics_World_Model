"""
Generate benchmark datasets for SEM and TEM modalities.

SEM (Scanning Electron Microscopy):
  - Material contrast phantoms (Shepp-Logan style)
  - Gaussian PSF convolution + charging artifacts + Poisson + Gaussian noise
  - Mismatch params: psf_sigma, charging_coeff, scan_distortion

TEM (Transmission Electron Microscopy):
  - Phase contrast via CTF (Contrast Transfer Function)
  - CTF * FFT(x_true) -> IFFT -> |.|^2 + noise
  - Mismatch params: defocus_error_nm, astigmatism_nm, ctf_phase_offset

Output layout (per modality):
  datasets/benchmark/{modality}/
    {tier}/
      {modality}_challenge_{tier}.h5   <- HDF5 with sample_NN groups
      spec.json                         <- mismatch parameter ranges
      true_spec.json                    <- actual per-sample values
      images/
        sample_NN/
          x_true.png
          y.png
          H_ideal.png
          reconstruction_baseline.png

Tiers: public (12), dev (20), hidden (20)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

# ── Root paths ────────────────────────────────────────────────────────────────

ROOT = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model")
OUT_BASE = ROOT / "datasets" / "benchmark"

SHAPE = (256, 256)

TIER_SIZES = {"public": 12, "dev": 20, "hidden": 20}

# ── Phantom generation ────────────────────────────────────────────────────────

def make_shepp_logan_like(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Multi-ellipse phantom with random positions, sizes, and intensities.

    Mimics SEM material contrast: discrete materials with distinct gray levels.
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy = (yy / H - 0.5) * 2.0   # [-1, 1]
    xx = (xx / W - 0.5) * 2.0

    # Background = random low-intensity substrate
    bg_level = rng.uniform(0.05, 0.20)
    img = np.full((H, W), bg_level, dtype=np.float32)

    # Randomly place 4–10 ellipses with distinct material contrasts
    n_ellipses = rng.integers(4, 11)
    for _ in range(n_ellipses):
        cy = rng.uniform(-0.7, 0.7)
        cx = rng.uniform(-0.7, 0.7)
        ry = rng.uniform(0.05, 0.35)
        rx = rng.uniform(0.05, 0.35)
        angle = rng.uniform(0, np.pi)
        level = rng.uniform(0.15, 1.0)

        # Rotated ellipse mask
        dy = yy - cy
        dx = xx - cx
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy_rot = cos_a * dy + sin_a * dx
        dx_rot = -sin_a * dy + cos_a * dx
        inside = (dy_rot / ry) ** 2 + (dx_rot / rx) ** 2 <= 1.0
        img[inside] = level

    # Add sub-structure texture (grain/surface roughness)
    texture = rng.standard_normal(shape).astype(np.float32)
    texture = gaussian_filter(texture, sigma=rng.uniform(0.5, 2.0))
    texture -= texture.min()
    texture /= texture.max() + 1e-8
    img = img + rng.uniform(0.02, 0.08) * texture
    img = np.clip(img, 0.0, 1.0)
    return img


def make_tem_phase_object(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Weak-phase object for TEM: projected potential of a thin specimen.

    Returns projected phase in [0, 1], representing a thin amorphous+crystalline
    specimen (random blobs = amorphous regions, periodic = lattice-like).
    """
    H, W = shape

    # Base: fBm amorphous background
    noise = np.zeros((H, W), dtype=np.float32)
    amp, sig = 1.0, 2.0
    for _ in range(5):
        layer = rng.standard_normal((H, W)).astype(np.float32)
        layer = gaussian_filter(layer, sigma=sig)
        noise += amp * layer
        amp *= 0.55
        sig *= 1.8
    noise -= noise.min()
    noise /= noise.max() + 1e-8

    # Crystalline inclusions: regular lattice-like blobs
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    crystal = np.zeros((H, W), dtype=np.float32)
    n_crystals = rng.integers(1, 4)
    for _ in range(n_crystals):
        cx = rng.uniform(0.2, 0.8) * W
        cy = rng.uniform(0.2, 0.8) * H
        size = rng.uniform(20, 60)
        spacing = rng.uniform(8, 20)
        inside = ((yy - cy) ** 2 + (xx - cx) ** 2) < size ** 2
        lattice_x = np.cos(2 * np.pi * xx / spacing)
        lattice_y = np.cos(2 * np.pi * yy / spacing)
        lattice = 0.5 + 0.5 * lattice_x * lattice_y
        crystal += inside.astype(np.float32) * lattice

    crystal = np.clip(crystal, 0, 1)
    x = 0.6 * noise + 0.4 * crystal
    x -= x.min()
    x /= x.max() + 1e-8
    return x.astype(np.float32)


def make_phantom(modality: str, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    if modality == "sem":
        return make_shepp_logan_like(rng, shape)
    else:
        return make_tem_phase_object(rng, shape)


# ── SEM forward model ─────────────────────────────────────────────────────────

def sem_psf(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """2D Gaussian PSF, centred, normalised to sum=1."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    psf = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum() + 1e-12
    return psf.astype(np.float32)


def apply_charging_artifact(img: np.ndarray, coeff: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate charging: locally brightened region + streak along scan direction."""
    H, W = img.shape
    out = img.copy()
    # Random bright charging blob
    n_sites = rng.integers(1, 4)
    for _ in range(n_sites):
        cy = rng.integers(10, H - 10)
        cx = rng.integers(10, W - 10)
        radius = rng.uniform(5, 20)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mask = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2))
        out = np.clip(out + coeff * mask, 0, 1)
    # Horizontal scan streak from charging site
    streak_row = cy
    streak = np.zeros((H, W), dtype=np.float32)
    streak[streak_row, cx:] = coeff * 0.5 * np.exp(
        -np.arange(W - cx) / (W * 0.15)
    )
    out = np.clip(out + streak, 0, 1)
    return out.astype(np.float32)


def apply_scan_distortion(img: np.ndarray, distortion_amp: float, rng: np.random.Generator) -> np.ndarray:
    """Slow scan-axis distortion: sinusoidal row jitter."""
    H, W = img.shape
    rows = np.arange(H, dtype=np.float32)
    cols = np.arange(W, dtype=np.float32)
    freq = rng.uniform(0.5, 3.0)  # cycles across image
    phase = rng.uniform(0, 2 * np.pi)
    jitter = distortion_amp * np.sin(2 * np.pi * freq * rows / H + phase)  # shape (H,)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx_distorted = xx + jitter[:, np.newaxis]
    xx_distorted = np.clip(xx_distorted, 0, W - 1)

    coords = [yy.ravel(), xx_distorted.ravel()]
    out = map_coordinates(img, coords, order=1, mode='nearest')
    return out.reshape(H, W).astype(np.float32)


def sem_forward(
    x_true: np.ndarray,
    psf_sigma: float,
    charging_coeff: float,
    scan_distortion: float,
    poisson_scale: float,
    readout_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SEM forward model, return (y, H_ideal)."""
    H, W = x_true.shape

    # 1. PSF blur (convolve via FFT)
    psf = sem_psf((H, W), psf_sigma)
    H_ideal = psf  # store centred PSF as forward model

    X_f = np.fft.fft2(x_true.astype(np.float64))
    PSF_f = np.fft.fft2(psf.astype(np.float64))
    blurred = np.real(np.fft.ifft2(X_f * PSF_f))
    blurred = np.roll(blurred, -H // 2, axis=0)
    blurred = np.roll(blurred, -W // 2, axis=1)
    blurred = np.clip(blurred, 0, 1).astype(np.float32)

    # 2. Scan distortion
    if scan_distortion > 0:
        blurred = apply_scan_distortion(blurred, scan_distortion, rng)

    # 3. Charging artifacts
    if charging_coeff > 0:
        blurred = apply_charging_artifact(blurred, charging_coeff, rng)

    # 4. Poisson noise (scale to photon counts then back)
    signal_counts = blurred * poisson_scale
    signal_counts = np.clip(signal_counts, 0, None)
    noisy = rng.poisson(signal_counts).astype(np.float32) / poisson_scale

    # 5. Gaussian readout noise
    noisy = noisy + rng.normal(0, readout_sigma, size=noisy.shape).astype(np.float32)
    y = np.clip(noisy, 0, 1).astype(np.float32)

    return y, H_ideal


def sem_baseline_reconstruction(y: np.ndarray, H_ideal: np.ndarray) -> np.ndarray:
    """Wiener-like deconvolution baseline for SEM."""
    H, W = y.shape
    Y_f = np.fft.fft2(y.astype(np.float64))
    PSF_f = np.fft.fft2(H_ideal.astype(np.float64))
    # Wiener filter: H* / (|H|^2 + epsilon)
    eps = 1e-3
    PSF_conj = np.conj(PSF_f)
    Wiener = PSF_conj / (np.abs(PSF_f) ** 2 + eps)
    recon = np.real(np.fft.ifft2(Y_f * Wiener))
    recon = np.roll(recon, H // 2, axis=0)
    recon = np.roll(recon, W // 2, axis=1)
    recon -= recon.min()
    recon /= recon.max() + 1e-8
    return np.clip(recon, 0, 1).astype(np.float32)


# ── TEM forward model ─────────────────────────────────────────────────────────

def compute_ctf(
    shape: tuple[int, int],
    defocus_nm: float,
    Cs_mm: float,
    wavelength_pm: float,
    pixel_size_nm: float,
    astigmatism_nm: float = 0.0,
    astig_angle_rad: float = 0.0,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """Compute the Contrast Transfer Function (CTF) on frequency grid.

    CTF(k) = cos(chi(k))
    chi(k) = pi/2 * Cs * lambda^3 * k^4 - pi * lambda * defocus * k^2 + phase_offset

    All in consistent units (nm):
      defocus_nm  : defocus in nm  (positive = underfocus)
      Cs_mm       : spherical aberration in mm
      wavelength_pm: electron wavelength in pm
      pixel_size_nm: real-space pixel size in nm
    """
    H, W = shape
    # Spatial frequency grid in 1/nm
    fy = np.fft.fftfreq(H, d=pixel_size_nm).astype(np.float64)  # cycles/nm
    fx = np.fft.fftfreq(W, d=pixel_size_nm).astype(np.float64)
    FX, FY = np.meshgrid(fx, fy)

    # With astigmatism: effective defocus varies with angle
    theta_k = np.arctan2(FY, FX)
    k2 = FX ** 2 + FY ** 2
    defocus_eff = defocus_nm + astigmatism_nm * np.cos(2 * (theta_k - astig_angle_rad))

    lam_nm = wavelength_pm * 1e-3  # pm -> nm
    Cs_nm = Cs_mm * 1e6             # mm -> nm

    chi = (np.pi / 2.0) * Cs_nm * (lam_nm ** 3) * (k2 ** 2) \
          - np.pi * lam_nm * defocus_eff * k2 \
          + phase_offset

    ctf = np.cos(chi).astype(np.float32)
    return ctf


def tem_forward(
    x_true: np.ndarray,
    defocus_nm: float,
    Cs_mm: float,
    wavelength_pm: float,
    pixel_size_nm: float,
    astigmatism_nm: float,
    astig_angle_rad: float,
    phase_offset: float,
    dose_electrons: float,
    readout_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply TEM phase-contrast forward model, return (y, H_ideal).

    Model: CTF modulates phase -> detector intensity = |IFFT(CTF * FFT(x))|^2
    """
    H, W = x_true.shape

    # CTF as "forward model" stored in H_ideal (shifted for display)
    ctf = compute_ctf(
        (H, W), defocus_nm, Cs_mm, wavelength_pm, pixel_size_nm,
        astigmatism_nm, astig_angle_rad, phase_offset,
    )
    H_ideal = np.fft.fftshift(ctf).astype(np.float32)
    # Normalise for storage: map [-1,1] -> [0,1]
    H_ideal_store = (H_ideal + 1.0) / 2.0

    # Weak-phase approximation: exit wave = 1 + i*sigma*V (x_true ~ projected potential)
    # Image = |IFFT(CTF * FFT(psi))|^2 ; psi_q = FT(x_true)
    X_f = np.fft.fft2(x_true.astype(np.float64))
    Y_f = ctf.astype(np.float64) * X_f
    image_complex = np.fft.ifft2(Y_f)
    # Intensity (modulus squared)
    intensity = np.abs(image_complex) ** 2
    intensity = intensity.astype(np.float32)

    # Normalise to [0,1] range before noise
    intensity -= intensity.min()
    intensity /= intensity.max() + 1e-8

    # Poisson noise (electron dose)
    signal_counts = intensity * dose_electrons
    signal_counts = np.clip(signal_counts, 0, None)
    noisy = rng.poisson(signal_counts).astype(np.float32) / dose_electrons

    # Gaussian readout
    noisy = noisy + rng.normal(0, readout_sigma, size=noisy.shape).astype(np.float32)
    y = np.clip(noisy, 0, 1).astype(np.float32)

    return y, H_ideal_store


def tem_baseline_reconstruction(y: np.ndarray, H_ideal_store: np.ndarray) -> np.ndarray:
    """CTF phase-flip correction baseline for TEM.

    Deconvolves by applying sign(CTF) in Fourier domain (Wiener-like).
    """
    H, W = y.shape
    # Recover CTF from stored [0,1] representation
    ctf_shifted = H_ideal_store.astype(np.float64) * 2.0 - 1.0
    ctf = np.fft.ifftshift(ctf_shifted)

    Y_f = np.fft.fft2(y.astype(np.float64))
    # Wiener deconvolution: CTF* / (CTF^2 + eps)
    eps = 0.05
    ctf_conj = np.conj(ctf)  # CTF is real here
    denom = ctf ** 2 + eps
    recon_f = Y_f * ctf_conj / denom
    recon = np.real(np.fft.ifft2(recon_f)).astype(np.float32)
    recon -= recon.min()
    recon /= recon.max() + 1e-8
    return np.clip(recon, 0, 1).astype(np.float32)


# ── Mismatch specs ────────────────────────────────────────────────────────────

SEM_SPEC = {
    "modality": "sem",
    "description": "SEM mismatch parameter ranges for challenge",
    "params": {
        "psf_sigma": {
            "unit": "pixels",
            "nominal": 3.0,
            "range": [2.0, 4.0],
            "mismatch_range": [1.0, 6.0],
            "description": "Gaussian PSF standard deviation",
        },
        "charging_coeff": {
            "unit": "dimensionless",
            "nominal": 0.05,
            "range": [0.0, 0.15],
            "mismatch_range": [0.0, 0.30],
            "description": "Charging artifact amplitude coefficient",
        },
        "scan_distortion_px": {
            "unit": "pixels",
            "nominal": 1.5,
            "range": [0.0, 3.0],
            "mismatch_range": [0.0, 6.0],
            "description": "Peak-to-peak scan-axis sinusoidal distortion amplitude",
        },
        "poisson_scale": {
            "unit": "photon counts at max signal",
            "nominal": 500,
            "range": [200, 1000],
            "description": "Effective photon count scale for Poisson noise",
        },
        "readout_sigma": {
            "unit": "normalised intensity",
            "nominal": 0.01,
            "range": [0.005, 0.025],
            "description": "Gaussian readout noise standard deviation",
        },
    },
}

TEM_SPEC = {
    "modality": "tem",
    "description": "TEM CTF mismatch parameter ranges for challenge",
    "params": {
        "defocus_nm": {
            "unit": "nm",
            "nominal": -500.0,
            "range": [-2000.0, -100.0],
            "mismatch_range": [-4000.0, 200.0],
            "description": "Defocus (negative = underfocus, conventional TEM)",
        },
        "defocus_error_nm": {
            "unit": "nm",
            "nominal": 0.0,
            "mismatch_range": [-300.0, 300.0],
            "description": "Error in estimated defocus (mismatch parameter)",
        },
        "astigmatism_nm": {
            "unit": "nm",
            "nominal": 0.0,
            "range": [0.0, 100.0],
            "mismatch_range": [0.0, 300.0],
            "description": "Two-fold astigmatism amplitude",
        },
        "ctf_phase_offset": {
            "unit": "radians",
            "nominal": 0.0,
            "mismatch_range": [-0.5, 0.5],
            "description": "Additional phase offset in CTF chi (mismatch parameter)",
        },
        "dose_electrons": {
            "unit": "electrons/pixel",
            "nominal": 1000,
            "range": [100, 5000],
            "description": "Total electron dose per pixel",
        },
        "readout_sigma": {
            "unit": "normalised intensity",
            "nominal": 0.005,
            "range": [0.002, 0.015],
            "description": "Detector readout noise standard deviation",
        },
        "Cs_mm": {
            "unit": "mm",
            "nominal": 1.0,
            "range": [0.0, 2.0],
            "description": "Spherical aberration coefficient",
        },
        "wavelength_pm": {
            "unit": "pm",
            "nominal": 2.51,
            "description": "Relativistic electron wavelength at 200 kV",
        },
        "pixel_size_nm": {
            "unit": "nm",
            "nominal": 0.05,
            "range": [0.02, 0.20],
            "description": "Real-space pixel size",
        },
    },
}


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_sem_params(rng: np.random.Generator, tier: str) -> dict:
    """Draw SEM forward model parameters for one sample."""
    # Harder / more diverse for dev and hidden
    if tier == "public":
        psf_sigma = float(rng.uniform(2.0, 4.0))
        charging_coeff = float(rng.uniform(0.0, 0.10))
        scan_distortion = float(rng.uniform(0.0, 2.0))
        poisson_scale = float(rng.integers(300, 800))
        readout_sigma = float(rng.uniform(0.005, 0.015))
    elif tier == "dev":
        psf_sigma = float(rng.uniform(1.5, 5.0))
        charging_coeff = float(rng.uniform(0.0, 0.18))
        scan_distortion = float(rng.uniform(0.0, 4.0))
        poisson_scale = float(rng.integers(150, 1000))
        readout_sigma = float(rng.uniform(0.005, 0.022))
    else:  # hidden
        psf_sigma = float(rng.uniform(1.0, 6.0))
        charging_coeff = float(rng.uniform(0.0, 0.25))
        scan_distortion = float(rng.uniform(0.0, 6.0))
        poisson_scale = float(rng.integers(100, 1200))
        readout_sigma = float(rng.uniform(0.003, 0.028))
    return {
        "psf_sigma": psf_sigma,
        "charging_coeff": charging_coeff,
        "scan_distortion_px": scan_distortion,
        "poisson_scale": poisson_scale,
        "readout_sigma": readout_sigma,
    }


def sample_tem_params(rng: np.random.Generator, tier: str) -> dict:
    """Draw TEM CTF parameters for one sample."""
    wavelength_pm = 2.51   # 200 kV, fixed
    if tier == "public":
        defocus_nm = float(rng.uniform(-1500.0, -200.0))
        astigmatism_nm = float(rng.uniform(0.0, 50.0))
        ctf_phase_offset = float(rng.uniform(-0.2, 0.2))
        dose_electrons = float(rng.integers(500, 2000))
        Cs_mm = float(rng.uniform(0.5, 1.5))
        pixel_size_nm = float(rng.uniform(0.03, 0.10))
        readout_sigma = float(rng.uniform(0.002, 0.008))
    elif tier == "dev":
        defocus_nm = float(rng.uniform(-2500.0, -100.0))
        astigmatism_nm = float(rng.uniform(0.0, 120.0))
        ctf_phase_offset = float(rng.uniform(-0.35, 0.35))
        dose_electrons = float(rng.integers(200, 3000))
        Cs_mm = float(rng.uniform(0.0, 2.0))
        pixel_size_nm = float(rng.uniform(0.02, 0.15))
        readout_sigma = float(rng.uniform(0.002, 0.012))
    else:  # hidden
        defocus_nm = float(rng.uniform(-4000.0, 200.0))
        astigmatism_nm = float(rng.uniform(0.0, 250.0))
        ctf_phase_offset = float(rng.uniform(-0.5, 0.5))
        dose_electrons = float(rng.integers(100, 5000))
        Cs_mm = float(rng.uniform(0.0, 2.0))
        pixel_size_nm = float(rng.uniform(0.02, 0.20))
        readout_sigma = float(rng.uniform(0.001, 0.015))

    astig_angle_rad = float(rng.uniform(0, np.pi))
    return {
        "defocus_nm": defocus_nm,
        "astigmatism_nm": astigmatism_nm,
        "astig_angle_rad": astig_angle_rad,
        "ctf_phase_offset": ctf_phase_offset,
        "dose_electrons": dose_electrons,
        "Cs_mm": Cs_mm,
        "wavelength_pm": wavelength_pm,
        "pixel_size_nm": pixel_size_nm,
        "readout_sigma": readout_sigma,
    }


# ── PNG preview helpers ───────────────────────────────────────────────────────

def save_png(arr: np.ndarray, path: Path) -> None:
    """Save float32 [0,1] array as 8-bit grayscale PNG."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    img_uint8 = (arr_clipped * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(str(path))


# ── Per-sample generation ─────────────────────────────────────────────────────

def generate_sem_sample(
    sample_idx: int,
    tier: str,
    rng: np.random.Generator,
    img_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate one SEM sample; save PNGs; return arrays + params."""
    params = sample_sem_params(rng, tier)

    x_true = make_phantom("sem", rng, SHAPE)

    y, H_ideal = sem_forward(
        x_true,
        psf_sigma=params["psf_sigma"],
        charging_coeff=params["charging_coeff"],
        scan_distortion=params["scan_distortion_px"],
        poisson_scale=params["poisson_scale"],
        readout_sigma=params["readout_sigma"],
        rng=rng,
    )

    recon_bl = sem_baseline_reconstruction(y, H_ideal)

    # Save PNGs
    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon_bl, sdir / "reconstruction_baseline.png")

    return x_true, y, H_ideal, recon_bl, params


def generate_tem_sample(
    sample_idx: int,
    tier: str,
    rng: np.random.Generator,
    img_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate one TEM sample; save PNGs; return arrays + params."""
    params = sample_tem_params(rng, tier)

    x_true = make_phantom("tem", rng, SHAPE)

    y, H_ideal = tem_forward(
        x_true,
        defocus_nm=params["defocus_nm"],
        Cs_mm=params["Cs_mm"],
        wavelength_pm=params["wavelength_pm"],
        pixel_size_nm=params["pixel_size_nm"],
        astigmatism_nm=params["astigmatism_nm"],
        astig_angle_rad=params["astig_angle_rad"],
        phase_offset=params["ctf_phase_offset"],
        dose_electrons=params["dose_electrons"],
        readout_sigma=params["readout_sigma"],
        rng=rng,
    )

    recon_bl = tem_baseline_reconstruction(y, H_ideal)

    # Save PNGs
    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon_bl, sdir / "reconstruction_baseline.png")

    return x_true, y, H_ideal, recon_bl, params


# ── Tier generation ───────────────────────────────────────────────────────────

# Deterministic base seeds (different for SEM and TEM, different per tier)
SEEDS = {
    "sem": {"public": 1000, "dev": 2000, "hidden": 3000},
    "tem": {"public": 4000, "dev": 5000, "hidden": 6000},
}


def generate_tier(modality: str, tier: str) -> None:
    """Generate one (modality, tier) combination and write outputs."""
    n_samples = TIER_SIZES[tier]
    base_seed = SEEDS[modality][tier]

    tier_dir = OUT_BASE / modality / tier
    h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
    img_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating {modality.upper()} | tier={tier} | n={n_samples}")
    print(f"  Output: {tier_dir}")
    print(f"{'='*60}")

    true_spec_records = {}

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = modality
        hf.attrs["tier"] = tier
        hf.attrs["n_samples"] = n_samples
        hf.attrs["shape"] = list(SHAPE)
        hf.attrs["generator"] = "generate_sem_tem_h5.py"
        hf.attrs["date"] = "2026-03-10"

        for i in range(n_samples):
            seed = base_seed + i
            rng = np.random.default_rng(seed)

            if modality == "sem":
                x_true, y, H_ideal, recon_bl, params = generate_sem_sample(
                    i, tier, rng, img_dir
                )
            else:
                x_true, y, H_ideal, recon_bl, params = generate_tem_sample(
                    i, tier, rng, img_dir
                )

            grp_name = f"sample_{i:02d}"
            grp = hf.create_group(grp_name)
            grp.create_dataset("x_true", data=x_true, dtype="float32", compression="gzip")
            grp.create_dataset("y", data=y, dtype="float32", compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32", compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon_bl, dtype="float32", compression="gzip")
            # Store params as attributes
            for k, v in params.items():
                grp.attrs[k] = v
            grp.attrs["seed"] = seed

            true_spec_records[grp_name] = {k: float(v) for k, v in params.items()}
            true_spec_records[grp_name]["seed"] = seed

            # Verification printout
            print(
                f"  [{i:02d}] x_true={x_true.shape} [{x_true.min():.3f},{x_true.max():.3f}]"
                f"  y={y.shape} [{y.min():.3f},{y.max():.3f}]"
                f"  H={H_ideal.shape} [{H_ideal.min():.3f},{H_ideal.max():.3f}]"
                f"  recon=[{recon_bl.min():.3f},{recon_bl.max():.3f}]"
            )

    print(f"  Written: {h5_path}")

    # Write spec.json (mismatch parameter ranges)
    spec = SEM_SPEC if modality == "sem" else TEM_SPEC
    spec_path = tier_dir / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  Written: {spec_path}")

    # Write true_spec.json (actual per-sample values)
    true_spec_path = tier_dir / "true_spec.json"
    with open(true_spec_path, "w") as f:
        json.dump(true_spec_records, f, indent=2)
    print(f"  Written: {true_spec_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for modality in ["sem", "tem"]:
        for tier in ["public", "dev", "hidden"]:
            generate_tier(modality, tier)

    print("\n" + "=" * 60)
    print("  ALL DONE — verifying HDF5 files")
    print("=" * 60)

    for modality in ["sem", "tem"]:
        for tier in ["public", "dev", "hidden"]:
            h5_path = OUT_BASE / modality / tier / f"{modality}_challenge_{tier}.h5"
            with h5py.File(h5_path, "r") as hf:
                keys = sorted(hf.keys())
                print(f"\n{modality.upper()} {tier}: {len(keys)} samples in {h5_path.name}")
                # Check first and last sample
                for k in [keys[0], keys[-1]]:
                    grp = hf[k]
                    for arr_name in ["x_true", "y", "H_ideal", "reconstruction_baseline"]:
                        d = grp[arr_name][:]
                        print(
                            f"    {k}/{arr_name}: shape={d.shape} "
                            f"dtype={d.dtype} range=[{d.min():.4f},{d.max():.4f}]"
                        )


if __name__ == "__main__":
    main()
