"""Generate small example .npy datasets for each imaging category.

Users can download these to try the Dataset Mode workflow.
Each example produces a measurement array + optional sensing matrix
that is compatible with the category's reconstruction algorithm.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cache generated examples in memory (they're small ~few KB each)
_cache: dict[str, dict] = {}

# Pre-serialized .npy bytes cache — filled at module load to avoid blocking the event loop
_npy_cache: dict[str, bytes] = {}


# ── Per-category example generators ──────────────────────────────────────

EXAMPLE_DATASETS: dict[str, dict] = {
    # ── Compressive Imaging ──
    "spc": {
        "display_name": "Single-Pixel Camera (SPC)",
        "category": "compressive",
        "description": "Compressed measurements via Gaussian random sensing matrix",
        "measurement_shape": "(256,)",
        "matrix_shape": "(256, 1024)",
        "has_matrix": True,
        "has_gt": True,
        "prompt_example": "This is single-pixel camera data with a 256x1024 Gaussian sensing matrix and 25% compression ratio",
        "variant_key": "spc_block",
    },
    "cassi": {
        "display_name": "CASSI Hyperspectral",
        "category": "compressive",
        "description": "Coded aperture snapshot spectral imager measurement",
        "measurement_shape": "(64, 91)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is CASSI hyperspectral data with 28 spectral bands and binary coded aperture",
        "variant_key": "sd_cassi",
    },
    "cacti": {
        "display_name": "CACTI Video",
        "category": "compressive",
        "description": "Snapshot compressive video (temporally coded)",
        "measurement_shape": "(64, 64)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is CACTI video compressive sensing data with 8 temporal frames",
        "variant_key": "cacti",
    },
    # ── Medical Imaging ──
    "ct": {
        "display_name": "CT Sinogram",
        "category": "medical",
        "description": "Fan-beam CT sinogram (detectors x angles)",
        "measurement_shape": "(128, 180)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is a CT sinogram from a fan-beam scanner with 180 projections and 128 detector pixels",
        "variant_key": "ct",
    },
    "mri": {
        "display_name": "MRI k-Space",
        "category": "medical",
        "description": "Undersampled k-space data (complex-valued)",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is 2D MRI k-space data with 4x Cartesian undersampling and Gaussian noise",
        "variant_key": "mri",
    },
    # ── Microscopy ──
    "confocal": {
        "display_name": "Confocal Microscopy",
        "category": "microscopy",
        "description": "PSF-blurred fluorescence microscopy image",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is confocal fluorescence microscopy data with Gaussian PSF blur and Poisson noise",
        "variant_key": "confocal_3d",
    },
    "widefield": {
        "display_name": "Widefield Microscopy",
        "category": "microscopy",
        "description": "Low-SNR widefield fluorescence image",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is widefield fluorescence microscopy data with out-of-focus blur",
        "variant_key": "widefield",
    },
    # ── Electron Microscopy ──
    "sem": {
        "display_name": "SEM Image",
        "category": "electron_microscopy",
        "description": "Scanning electron microscope image with CTF effects",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is SEM image data with beam aberration and Poisson shot noise",
        "variant_key": "sem",
    },
    # ── Remote Sensing ──
    "sar": {
        "display_name": "SAR Phase History",
        "category": "remote_sensing",
        "description": "Synthetic aperture radar phase history data",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is SAR phase history data from a stripmap-mode synthetic aperture radar",
        "variant_key": "sar",
    },
    # ── Scanning Probe ──
    "afm": {
        "display_name": "AFM Topography",
        "category": "scanning_probe",
        "description": "Atomic force microscopy surface scan with tip convolution",
        "measurement_shape": "(128, 128)",
        "has_matrix": False,
        "has_gt": True,
        "prompt_example": "This is AFM topography data with tip dilation artifacts and thermal noise",
        "variant_key": "afm",
    },
}


def generate_example(key: str) -> dict:
    """Generate example dataset arrays for a given category key.

    Returns dict with:
        measurement: np.ndarray
        matrix: np.ndarray | None
        ground_truth: np.ndarray | None
        info: dict (metadata about the example)
    """
    if key in _cache:
        return _cache[key]

    info = EXAMPLE_DATASETS.get(key)
    if info is None:
        raise ValueError(f"Unknown example dataset: {key}")

    rng = np.random.default_rng(seed=42)
    gen = _GENERATORS.get(key, _gen_generic)
    result = gen(rng, info)
    result["info"] = info

    _cache[key] = result
    return result


def array_to_npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to .npy format in memory."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ── Generator functions per category ──────────────────────────────────────


def _gen_spc(rng: np.random.Generator, info: dict) -> dict:
    """SPC: Phi * x + noise."""
    n = 1024  # signal dimension (32x32)
    m = 256   # measurements
    x = _phantom_2d(32, rng).ravel().astype(np.float32)
    Phi = rng.standard_normal((m, n)).astype(np.float32) / np.sqrt(m)
    y = (Phi @ x + 0.01 * rng.standard_normal(m)).astype(np.float32)
    return {"measurement": y, "matrix": Phi, "ground_truth": x.reshape(32, 32)}


def _gen_cassi(rng: np.random.Generator, info: dict) -> dict:
    """CASSI: 2D coded measurement of a hyperspectral cube."""
    nx, nL = 64, 28
    cube = np.stack([_phantom_2d(nx, rng) * (0.5 + 0.5 * np.sin(np.pi * l / nL))
                     for l in range(nL)], axis=-1).astype(np.float32)
    mask = rng.integers(0, 2, size=(nx, nx)).astype(np.float32)
    # Simulate dispersion: shift each band and sum
    y = np.zeros((nx, nx + nL - 1), dtype=np.float32)
    for l in range(nL):
        y[:, l:l + nx] += mask * cube[:, :, l]
    y += 0.02 * rng.standard_normal(y.shape).astype(np.float32)
    gt = cube[:, :, nL // 2]  # middle spectral band
    return {"measurement": y, "matrix": None, "ground_truth": gt}


def _gen_cacti(rng: np.random.Generator, info: dict) -> dict:
    """CACTI: temporally coded video compression."""
    nx, nT = 64, 8
    frames = np.stack([_phantom_2d(nx, rng, seed_offset=t) for t in range(nT)],
                      axis=-1).astype(np.float32)
    masks = rng.integers(0, 2, size=(nx, nx, nT)).astype(np.float32)
    y = np.sum(masks * frames, axis=-1)
    y += 0.02 * rng.standard_normal(y.shape).astype(np.float32)
    return {"measurement": y.astype(np.float32), "matrix": None,
            "ground_truth": frames[:, :, 0]}


def _gen_ct(rng: np.random.Generator, info: dict) -> dict:
    """CT: Radon transform sinogram."""
    n = 128
    phantom = _phantom_2d(n, rng)
    n_angles = 180
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    # Simple Radon transform (sum along rotated lines)
    sinogram = np.zeros((n, n_angles), dtype=np.float32)
    for i, angle in enumerate(theta):
        rotated = _rotate_image(phantom, angle)
        sinogram[:, i] = rotated.sum(axis=1) / n
    sinogram += 0.01 * rng.standard_normal(sinogram.shape).astype(np.float32)
    return {"measurement": sinogram, "matrix": None, "ground_truth": phantom}


def _gen_mri(rng: np.random.Generator, info: dict) -> dict:
    """MRI: undersampled k-space."""
    n = 128
    phantom = _phantom_2d(n, rng)
    kspace = np.fft.fftshift(np.fft.fft2(phantom))
    # 4x undersampling mask (keep center + random lines)
    mask = np.zeros((n, n), dtype=np.float32)
    mask[n // 2 - 8:n // 2 + 8, :] = 1  # center fully sampled
    rand_lines = rng.choice(n, size=n // 4, replace=False)
    mask[rand_lines, :] = 1
    y = (kspace * mask).astype(np.complex64)
    # Save as real/imag stacked
    y_ri = np.stack([y.real, y.imag], axis=-1).astype(np.float32)
    return {"measurement": y_ri, "matrix": None, "ground_truth": phantom}


def _gen_microscopy(rng: np.random.Generator, info: dict) -> dict:
    """Microscopy: PSF-blurred + Poisson noise."""
    n = 128
    phantom = _phantom_2d(n, rng)
    # Gaussian PSF
    sigma = 2.5
    ax = np.arange(n) - n // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    # Convolve
    from numpy.fft import fft2, ifft2
    blurred = np.real(ifft2(fft2(phantom) * fft2(psf))).clip(0, 1)
    # Poisson noise
    photon_count = 1000
    noisy = rng.poisson(blurred * photon_count).astype(np.float32) / photon_count
    return {"measurement": noisy, "matrix": None, "ground_truth": phantom}


def _gen_electron(rng: np.random.Generator, info: dict) -> dict:
    """Electron microscopy: CTF modulation."""
    n = 128
    phantom = _phantom_2d(n, rng)
    # CTF in Fourier domain
    ky = np.fft.fftfreq(n)
    kx = np.fft.fftfreq(n)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    ctf = np.sin(-np.pi * 2.0 * k2)
    # Apply CTF
    img_f = np.fft.fft2(phantom)
    modulated = np.real(np.fft.ifft2(img_f * ctf))
    noisy = modulated + 0.05 * rng.standard_normal((n, n))
    return {"measurement": noisy.astype(np.float32), "matrix": None,
            "ground_truth": phantom}


def _gen_sar(rng: np.random.Generator, info: dict) -> dict:
    """SAR: phase history data."""
    n = 128
    phantom = _phantom_2d(n, rng)
    # Phase history = FFT of scene with phase errors
    phase_err = 0.1 * rng.standard_normal((n, n))
    kdata = np.fft.fft2(phantom) * np.exp(1j * phase_err)
    # Save as real/imag stacked
    y_ri = np.stack([kdata.real, kdata.imag], axis=-1).astype(np.float32)
    return {"measurement": y_ri, "matrix": None, "ground_truth": phantom}


def _gen_afm(rng: np.random.Generator, info: dict) -> dict:
    """AFM: tip-dilated surface topography."""
    n = 128
    phantom = _phantom_2d(n, rng)
    # Tip dilation (morphological dilation with small kernel)
    from scipy.ndimage import grey_dilation
    tip = np.ones((5, 5))
    dilated = grey_dilation(phantom, footprint=tip)
    noisy = dilated + 0.02 * rng.standard_normal((n, n))
    return {"measurement": noisy.astype(np.float32), "matrix": None,
            "ground_truth": phantom}


def _gen_generic(rng: np.random.Generator, info: dict) -> dict:
    """Fallback: noisy measurement of a phantom."""
    n = 128
    phantom = _phantom_2d(n, rng)
    noisy = phantom + 0.05 * rng.standard_normal((n, n))
    return {"measurement": noisy.astype(np.float32), "matrix": None,
            "ground_truth": phantom}


# Generator dispatch
_GENERATORS = {
    "spc": _gen_spc,
    "cassi": _gen_cassi,
    "cacti": _gen_cacti,
    "ct": _gen_ct,
    "mri": _gen_mri,
    "confocal": _gen_microscopy,
    "widefield": _gen_microscopy,
    "sem": _gen_electron,
    "sar": _gen_sar,
    "afm": _gen_afm,
}


# ── Phantom generators ───────────────────────────────────────────────────


def _phantom_2d(n: int, rng: np.random.Generator, seed_offset: int = 0) -> np.ndarray:
    """Generate a Shepp-Logan-like 2D phantom with geometric shapes."""
    img = np.zeros((n, n), dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:n, 0:n].astype(np.float32) / n

    # Background ellipse
    cx, cy = 0.5, 0.5
    a, b = 0.4, 0.35
    mask = ((x_grid - cx) / a) ** 2 + ((y_grid - cy) / b) ** 2 <= 1
    img[mask] = 0.3

    # Add random ellipses
    local_rng = np.random.default_rng(rng.integers(0, 2**31) + seed_offset)
    n_shapes = local_rng.integers(3, 8)
    for _ in range(n_shapes):
        cx = local_rng.uniform(0.2, 0.8)
        cy = local_rng.uniform(0.2, 0.8)
        a = local_rng.uniform(0.05, 0.15)
        b = local_rng.uniform(0.05, 0.15)
        val = local_rng.uniform(0.4, 1.0)
        mask = ((x_grid - cx) / a) ** 2 + ((y_grid - cy) / b) ** 2 <= 1
        img[mask] = val

    return img


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle (degrees) using nearest-neighbor interpolation."""
    angle_rad = np.deg2rad(angle_deg)
    n = img.shape[0]
    center = n / 2.0
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    y_out, x_out = np.mgrid[0:n, 0:n].astype(np.float64)
    x_shifted = x_out - center
    y_shifted = y_out - center

    x_src = cos_a * x_shifted + sin_a * y_shifted + center
    y_src = -sin_a * x_shifted + cos_a * y_shifted + center

    x_src = np.round(x_src).astype(int)
    y_src = np.round(y_src).astype(int)

    valid = (x_src >= 0) & (x_src < n) & (y_src >= 0) & (y_src < n)
    result = np.zeros_like(img)
    result[valid] = img[y_src[valid], x_src[valid]]
    return result


# ── Pre-cached .npy bytes for instant downloads ─────────────────────────


def get_npy_bytes(key: str, role: str) -> bytes:
    """Return pre-serialized .npy bytes for a given example + role.

    This is instant (no CPU work) because all data is pre-generated at startup.
    """
    cache_key = f"{key}:{role}"
    if cache_key in _npy_cache:
        return _npy_cache[cache_key]
    # Fallback: generate on demand (shouldn't happen after warmup)
    example = generate_example(key)
    arr = example.get(role)
    if arr is None:
        raise ValueError(f"No {role} for {key}")
    data = array_to_npy_bytes(arr)
    _npy_cache[cache_key] = data
    return data


def warmup_all():
    """Pre-generate all example datasets and serialize to .npy bytes.

    Call once at app startup (in a background thread) so that
    download_example_dataset() never blocks the async event loop.
    """
    logger.info("Pre-generating %d example datasets...", len(EXAMPLE_DATASETS))
    for key, info in EXAMPLE_DATASETS.items():
        example = generate_example(key)
        for role in ("measurement", "matrix", "ground_truth"):
            arr = example.get(role)
            if arr is not None:
                _npy_cache[f"{key}:{role}"] = array_to_npy_bytes(arr)
    logger.info("Example datasets ready (%d .npy files cached)", len(_npy_cache))
