#!/usr/bin/env python3
"""
Generate benchmark datasets for batch 10 modalities:
  sonar, spinning_disk, srs, stem, stm, streak_camera,
  structured_light, talbot_lau, terahertz, three_photon

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Seeds: public  1400+i*17
       dev     7400+i*17
       hidden  9450+i*17
"""
from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available — skipping PNG previews")

ROOT = pathlib.Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model")
DATASETS_ROOT = ROOT / "datasets" / "benchmark"

TIERS = {
    "public": 12,
    "dev": 20,
    "hidden": 20,
}

SEED_BASES = {
    "public": 1400,
    "dev": 7400,
    "hidden": 9450,
}
SEED_STEP = 17


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _save_png(arr: np.ndarray, path: pathlib.Path) -> None:
    if not HAS_PIL:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        disp = arr.astype(np.float32)
        if disp.ndim == 3:
            # For 3D arrays: use first 2D slice or the first channel
            if disp.shape[0] <= disp.shape[-1]:
                disp = disp[0]  # first frame/channel along axis 0
            else:
                disp = disp[..., 0]
        img = np.clip(_norm01(disp) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img, "L").save(str(path))
    except Exception as e:
        print(f"  PNG save failed for {path}: {e}")


def make_dirs(modality: str, tier: str) -> pathlib.Path:
    d = DATASETS_ROOT / modality / tier
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d


def write_spec(tier_dir: pathlib.Path, modality: str, tier: str, n_samples: int,
               extra: dict | None = None) -> None:
    spec = {
        "modality": modality,
        "tier": tier,
        "n_samples": n_samples,
        "measurement_key": "y",
        "groundtruth_key": "x_true",
    }
    if extra:
        spec.update(extra)
    (tier_dir / "spec.json").write_text(json.dumps(spec, indent=2))


def write_true_spec(tier_dir: pathlib.Path, true_params: dict) -> None:
    (tier_dir / "true_spec.json").write_text(json.dumps(true_params, indent=2))


def write_h5(tier_dir: pathlib.Path, modality: str, tier: str, samples: list) -> pathlib.Path:
    fname = tier_dir / f"{modality}_challenge_{tier}.h5"
    with h5py.File(fname, "w") as f:
        for i, s in enumerate(samples):
            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=s["x_true"].astype(np.float32))
            grp.create_dataset("y", data=s["y"].astype(np.float32))
            grp.create_dataset("H_ideal", data=s["H_ideal"].astype(np.float32))
            grp.create_dataset("reconstruction_baseline",
                               data=s["reconstruction_baseline"].astype(np.float32))
    return fname


def make_blob_phantom(rng: np.random.Generator, size: int = 128, n_blobs: int = 8) -> np.ndarray:
    """Random smooth phantom via superposition of Gaussian blobs."""
    canvas = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.ogrid[:size, :size]
    n = int(rng.integers(n_blobs // 2, n_blobs + 1))
    for _ in range(n):
        cy = rng.integers(10, size - 10)
        cx = rng.integers(10, size - 10)
        sigma = rng.uniform(4, 22)
        amp = rng.uniform(0.3, 1.0)
        canvas += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return _norm01(canvas)


# ---------------------------------------------------------------------------
# Physics generators
# ---------------------------------------------------------------------------

# ── SONAR ──────────────────────────────────────────────────────────────────
def gen_sonar(rng: np.random.Generator):
    """
    x_true: (128,128) underwater object reflectivity map.
    y: (90,128) sonar echo projections (delay-and-sum sinogram-like).
    H_ideal: array of projection angles [float32].
    reconstruction_baseline: delay-and-sum back-projection.
    """
    size = 128
    n_angles = 90
    x_true = make_blob_phantom(rng, size=size)

    # Add sparse point scatterers
    n_scatterers = int(rng.integers(3, 10))
    for _ in range(n_scatterers):
        sy = int(rng.integers(5, size - 5))
        sx = int(rng.integers(5, size - 5))
        x_true[sy, sx] = min(1.0, x_true[sy, sx] + rng.uniform(0.3, 0.8))
    x_true = _norm01(x_true)

    # Sonar projections: beam sweeps at angles, project along perpendicular
    angles_deg = np.linspace(0, 179, n_angles)
    angles_rad = np.deg2rad(angles_deg)
    projections = np.zeros((n_angles, size), dtype=np.float32)
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    for k, ang in enumerate(angles_rad):
        # project along angle direction
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        # t-coordinate for each pixel: perpendicular to beam direction
        t = (xx - cx) * cos_a + (yy - cy) * sin_a  # shape (size,size)
        t_idx = np.round(t + size // 2).astype(int)
        t_idx = np.clip(t_idx, 0, size - 1)
        proj = np.zeros(size, dtype=np.float64)
        np.add.at(proj, t_idx.ravel(), x_true.ravel())
        # Normalize by number of contributions
        counts = np.zeros(size, dtype=np.float64)
        np.add.at(counts, t_idx.ravel(), 1.0)
        counts = np.where(counts == 0, 1.0, counts)
        projections[k] = (proj / counts).astype(np.float32)

    # Add noise
    noise_std = float(rng.uniform(0.01, 0.04))
    y = projections + rng.normal(0, noise_std, projections.shape).astype(np.float32)
    y = y.astype(np.float32)

    H_ideal = angles_deg.astype(np.float32)  # (90,)

    # Delay-and-sum back-projection
    recon = np.zeros((size, size), dtype=np.float64)
    for k, ang in enumerate(angles_rad):
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        t = (xx - cx) * cos_a + (yy - cy) * sin_a
        t_idx = np.round(t + size // 2).astype(int)
        t_idx = np.clip(t_idx, 0, size - 1)
        recon += y[k][t_idx]
    recon /= n_angles
    reconstruction_baseline = _norm01(recon)

    true_params = {
        "n_angles": n_angles,
        "angle_range_deg": [0.0, 179.0],
        "noise_std": noise_std,
        "n_scatterers": n_scatterers,
        "model": "Sonar delay-and-sum projection + back-projection",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── SPINNING DISK ─────────────────────────────────────────────────────────
def gen_spinning_disk(rng: np.random.Generator):
    """
    x_true: (128,128) confocal fluorescence image.
    y: x_true convolved with spinning-disk PSF (Gaussian) + Poisson noise.
    H_ideal: PSF sigma [float32 scalar array].
    reconstruction_baseline: Richardson-Lucy deconvolution (5 iters).
    """
    size = 128
    x_true = make_blob_phantom(rng, size=size)

    sigma_psf = float(rng.uniform(1.0, 3.5))
    blurred = gaussian_filter(x_true, sigma=sigma_psf)

    # Poisson noise scaled by photon count
    scale = float(rng.uniform(80, 300))
    y = rng.poisson(blurred * scale).astype(np.float32) / scale
    y = y.astype(np.float32)

    H_ideal = np.array([sigma_psf], dtype=np.float32)

    # Richardson-Lucy deconvolution 5 iters
    est = y.copy() + 1e-8
    for _ in range(5):
        ratio = y / (gaussian_filter(est, sigma_psf) + 1e-8)
        est = est * gaussian_filter(ratio, sigma_psf)
        est = np.clip(est, 0, None)
    reconstruction_baseline = est.astype(np.float32)

    true_params = {
        "psf_sigma_px": sigma_psf,
        "photon_scale": scale,
        "noise_type": "Poisson",
        "model": "Spinning-disk Gaussian PSF + Poisson noise; RL deconv 5 iters",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── SRS ───────────────────────────────────────────────────────────────────
def gen_srs(rng: np.random.Generator):
    """
    x_true: (128,128,4) stimulated Raman scattering image at 4 wavenumbers.
    y: (128,128,4) measured with shot noise.
    H_ideal: (4,) wavenumbers array [float32].
    reconstruction_baseline: y / y.max() (normalized).
    """
    size = 128
    n_wn = 4  # wavenumbers

    # Build 4-channel image; each channel is a smooth phantom
    channels = []
    for ch in range(n_wn):
        base = make_blob_phantom(rng, size=size)
        # Each wavenumber has different relative intensities
        weight = float(rng.uniform(0.3, 1.0))
        channels.append(base * weight)
    x_true = np.stack(channels, axis=-1).astype(np.float32)  # (128,128,4)

    # Shot noise: Poisson
    scale = float(rng.uniform(100, 500))
    y = rng.poisson(x_true * scale).astype(np.float32) / scale
    y = y.astype(np.float32)

    # Wavenumber axis: typical SRS window around CH2 stretch
    wavenumbers = np.array([2800.0, 2850.0, 2900.0, 2950.0], dtype=np.float32)
    H_ideal = wavenumbers  # (4,)

    # Reconstruction: normalize
    max_val = float(y.max()) + 1e-8
    reconstruction_baseline = (y / max_val).astype(np.float32)

    true_params = {
        "wavenumbers_cm-1": wavenumbers.tolist(),
        "photon_scale": scale,
        "noise_type": "Poisson shot noise",
        "model": "SRS 4-wavenumber image with shot noise; normalized reconstruction",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── STEM ──────────────────────────────────────────────────────────────────
def gen_stem(rng: np.random.Generator):
    """
    x_true: (128,128) STEM-HAADF atomic image (periodic lattice + sparse atoms).
    y: x_true + Gaussian noise (sigma=0.03).
    H_ideal: (1,) probe sigma [float32].
    reconstruction_baseline: Gaussian smoothing.
    """
    size = 128
    # Periodic lattice
    period = int(rng.integers(8, 18))
    yy, xx = np.mgrid[:size, :size]
    lattice = (np.sin(2 * np.pi * yy / period) * np.sin(2 * np.pi * xx / period))
    lattice = _norm01(lattice)

    # Sparse atom peaks
    n_atoms = int(rng.integers(10, 30))
    for _ in range(n_atoms):
        ay = int(rng.integers(2, size - 2))
        ax = int(rng.integers(2, size - 2))
        amp = float(rng.uniform(0.5, 1.5))
        sigma_atom = float(rng.uniform(0.8, 2.0))
        blob = amp * np.exp(-((yy - ay) ** 2 + (xx - ax) ** 2) / (2 * sigma_atom ** 2))
        lattice = lattice + blob.astype(np.float32)
    x_true = _norm01(lattice)

    # HAADF: intensity ~ Z^2; approximate with slight nonlinearity
    x_true_haadf = (x_true ** 1.7).astype(np.float32)
    x_true_haadf = _norm01(x_true_haadf)

    noise_sigma = 0.03
    y = (x_true_haadf + rng.normal(0, noise_sigma, x_true_haadf.shape)).astype(np.float32)

    probe_sigma = float(rng.uniform(0.8, 1.5))
    H_ideal = np.array([probe_sigma], dtype=np.float32)

    # Baseline: light Gaussian smoothing
    reconstruction_baseline = gaussian_filter(y, sigma=0.8).astype(np.float32)

    true_params = {
        "lattice_period_px": period,
        "n_atoms": n_atoms,
        "noise_sigma": noise_sigma,
        "probe_sigma_px": probe_sigma,
        "model": "STEM-HAADF periodic lattice + atoms + Gaussian noise; Gaussian smoothing baseline",
    }
    return dict(x_true=x_true_haadf, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── STM ───────────────────────────────────────────────────────────────────
def gen_stm(rng: np.random.Generator):
    """
    x_true: (128,128) STM tunneling current map (surface topography).
    y: x_true + 1/f noise overlay + horizontal line artifacts.
    H_ideal: (1,) 1/f noise amplitude [float32].
    reconstruction_baseline: subtract row means (line artifact removal).
    """
    size = 128

    # Surface topography: smooth with periodic features
    period = int(rng.integers(6, 20))
    yy, xx = np.mgrid[:size, :size]
    surface = (0.5 * np.sin(2 * np.pi * xx / period) +
               0.3 * np.sin(2 * np.pi * yy / (period * 1.3) + 0.5))
    # Add random atomic features
    n_features = int(rng.integers(5, 20))
    for _ in range(n_features):
        fy = int(rng.integers(2, size - 2))
        fx = int(rng.integers(2, size - 2))
        amp = float(rng.uniform(0.2, 0.8))
        sig = float(rng.uniform(1.0, 4.0))
        surface += amp * np.exp(-((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * sig ** 2))
    x_true = _norm01(surface)

    # 1/f noise
    freqs = np.fft.fftfreq(size)
    fy2d, fx2d = np.meshgrid(freqs, freqs, indexing='ij')
    f_magnitude = np.sqrt(fy2d ** 2 + fx2d ** 2)
    f_magnitude[0, 0] = 1.0  # avoid division by zero
    noise_spectrum = (rng.standard_normal((size, size)) +
                      1j * rng.standard_normal((size, size))) / f_magnitude
    noise_1f = np.real(np.fft.ifft2(noise_spectrum)).astype(np.float32)
    noise_1f = noise_1f / (noise_1f.std() + 1e-8)
    noise_amp = float(rng.uniform(0.03, 0.12))
    noise_1f *= noise_amp

    # Line artifacts: add random offset per row
    line_amp = float(rng.uniform(0.02, 0.08))
    line_offsets = rng.uniform(-line_amp, line_amp, size).astype(np.float32)
    line_artifact = np.tile(line_offsets[:, np.newaxis], (1, size))

    y = (x_true + noise_1f + line_artifact).astype(np.float32)

    H_ideal = np.array([noise_amp, line_amp], dtype=np.float32)

    # Baseline: subtract row means to remove line artifacts
    reconstruction_baseline = y - y.mean(axis=1, keepdims=True)
    reconstruction_baseline = _norm01(reconstruction_baseline)

    true_params = {
        "lattice_period_px": period,
        "n_surface_features": n_features,
        "noise_1f_amplitude": noise_amp,
        "line_artifact_amplitude": line_amp,
        "model": "STM 1/f noise + line artifacts; row-mean subtraction baseline",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── STREAK CAMERA ─────────────────────────────────────────────────────────
def gen_streak_camera(rng: np.random.Generator):
    """
    x_true: (128,64) time-resolved spectrum (space x time).
    y: x_true + readout noise + Gaussian blur in time.
    H_ideal: (1,) time blur sigma [float32].
    reconstruction_baseline: Wiener deconvolution (1D in time).
    """
    n_space = 128
    n_time = 64

    # Time-resolved spectrum: few temporal pulses at different spatial positions
    x_true = np.zeros((n_space, n_time), dtype=np.float32)
    n_pulses = int(rng.integers(2, 6))
    for _ in range(n_pulses):
        s_center = int(rng.integers(5, n_space - 5))
        t_center = int(rng.integers(5, n_time - 5))
        s_sigma = float(rng.uniform(3, 15))
        t_sigma = float(rng.uniform(1, 5))
        amp = float(rng.uniform(0.3, 1.0))
        ss, tt = np.ogrid[:n_space, :n_time]
        pulse = amp * np.exp(
            -((ss - s_center) ** 2 / (2 * s_sigma ** 2) +
              (tt - t_center) ** 2 / (2 * t_sigma ** 2))
        )
        x_true += pulse.astype(np.float32)
    x_true = _norm01(x_true)

    # Time-axis blur (streak smearing in time)
    t_blur_sigma = float(rng.uniform(0.5, 2.0))
    from scipy.ndimage import uniform_filter1d
    y_blurred = gaussian_filter(x_true, sigma=(0, t_blur_sigma))  # blur only along time axis

    # Readout noise
    readout_std = float(rng.uniform(0.01, 0.04))
    y = (y_blurred + rng.normal(0, readout_std, y_blurred.shape)).astype(np.float32)

    H_ideal = np.array([t_blur_sigma], dtype=np.float32)

    # Wiener deconvolution (1D in time per spatial row)
    from numpy.fft import fft, ifft
    snr = 20.0
    recon = np.zeros_like(y)
    # Build 1D Gaussian kernel in time
    t_coords = np.arange(n_time)
    kernel_1d = np.exp(-t_coords ** 2 / (2 * t_blur_sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    H_f = np.fft.rfft(kernel_1d, n=n_time)
    H_conj = np.conj(H_f)
    denom = H_conj * H_f + (1.0 / snr)
    Wiener_f = H_conj / denom
    for row in range(n_space):
        Y_f = np.fft.rfft(y[row], n=n_time)
        X_hat = Wiener_f * Y_f
        recon[row] = np.real(np.fft.irfft(X_hat, n=n_time))
    reconstruction_baseline = _norm01(recon.astype(np.float32))

    true_params = {
        "n_space_pixels": n_space,
        "n_time_bins": n_time,
        "n_pulses": n_pulses,
        "time_blur_sigma": t_blur_sigma,
        "readout_noise_std": readout_std,
        "model": "Streak camera: Gaussian time blur + readout noise; Wiener deconv baseline",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── STRUCTURED LIGHT ─────────────────────────────────────────────────────
def gen_structured_light(rng: np.random.Generator):
    """
    x_true: (128,128) 3D surface depth map.
    y: (4,128,128) four phase-shifted fringe patterns.
    H_ideal: (4,) phase shifts [float32].
    reconstruction_baseline: 4-step phase-shift algorithm output.
    """
    size = 128

    # 3D depth map: smooth surface with bumps
    depth = make_blob_phantom(rng, size=size) * 0.5  # depth range 0..0.5
    # Add tilted plane
    yy, xx = np.mgrid[:size, :size]
    tilt_x = float(rng.uniform(-0.002, 0.002))
    tilt_y = float(rng.uniform(-0.002, 0.002))
    depth = depth + tilt_x * xx + tilt_y * yy
    depth = _norm01(depth)
    x_true = depth.astype(np.float32)

    # Fringe frequency
    f_fringe = float(rng.uniform(0.08, 0.15))  # cycles per pixel
    phase_shifts = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32)

    # Modulation amplitude and DC
    A = float(rng.uniform(0.3, 0.6))  # fringe amplitude
    B = float(rng.uniform(0.4, 0.7))  # DC background

    # Object phase: proportional to depth
    phi_obj = x_true * 2 * np.pi  # phase in [0, 2pi]

    # 4 fringe patterns
    fringe_patterns = []
    noise_std = float(rng.uniform(0.005, 0.02))
    for delta in phase_shifts:
        fringe = B + A * np.cos(2 * np.pi * f_fringe * xx + phi_obj + delta)
        fringe = fringe + rng.normal(0, noise_std, fringe.shape).astype(np.float32)
        fringe_patterns.append(fringe.astype(np.float32))
    y = np.stack(fringe_patterns, axis=0).astype(np.float32)  # (4,128,128)

    H_ideal = phase_shifts  # (4,)

    # 4-step phase shift reconstruction: phi = atan2(I4-I2, I1-I3)
    I1, I2, I3, I4 = y[0], y[1], y[2], y[3]
    numerator = I4 - I2
    denominator = I1 - I3
    phase_map = np.arctan2(numerator, denominator).astype(np.float32)
    # Normalize to [0,1] for reconstruction
    reconstruction_baseline = _norm01(phase_map)

    true_params = {
        "fringe_frequency_cpp": f_fringe,
        "n_phase_steps": 4,
        "phase_shifts_rad": phase_shifts.tolist(),
        "fringe_amplitude": A,
        "dc_background": B,
        "noise_std": noise_std,
        "model": "Structured light 4-step phase shift; arctan2 reconstruction",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── TALBOT-LAU ────────────────────────────────────────────────────────────
def gen_talbot_lau(rng: np.random.Generator):
    """
    x_true: (128,128) phase grating Talbot image (periodic + object).
    y: x_true convolved with Talbot kernel (cosine pattern) + noise.
    H_ideal: (2,) [grating_period, Talbot_distance] [float32].
    reconstruction_baseline: demodulate by dividing by reference fringe.
    """
    size = 128
    yy, xx = np.mgrid[:size, :size]

    # Object: smooth phase map
    obj_phase = make_blob_phantom(rng, size=size)

    # Grating parameters
    grating_period = float(rng.uniform(6, 14))  # pixels
    visibility = float(rng.uniform(0.3, 0.7))

    # Reference fringe pattern (Talbot self-image)
    ref_fringe = 1.0 + visibility * np.cos(2 * np.pi * xx / grating_period)

    # Phase contrast image: modulation of fringe by object
    phi_shift = obj_phase * np.pi * float(rng.uniform(0.3, 1.0))
    talbot_image = 1.0 + visibility * np.cos(2 * np.pi * xx / grating_period + phi_shift)
    x_true = _norm01(talbot_image)

    # Measurement: Talbot image + Poisson noise
    scale = float(rng.uniform(100, 400))
    y_count = rng.poisson(x_true * scale).astype(np.float32) / scale
    noise_std = float(rng.uniform(0.005, 0.02))
    y = (y_count + rng.normal(0, noise_std, x_true.shape)).astype(np.float32)

    talbot_distance = float(rng.uniform(20, 60))  # mm (metadata only)
    H_ideal = np.array([grating_period, talbot_distance], dtype=np.float32)

    # Reconstruction: divide by reference fringe to demodulate
    ref_norm = _norm01(ref_fringe) + 0.1  # avoid div by zero
    reconstruction_baseline = _norm01(y / ref_norm)

    true_params = {
        "grating_period_px": grating_period,
        "talbot_distance_mm": talbot_distance,
        "fringe_visibility": visibility,
        "photon_scale": scale,
        "noise_std": noise_std,
        "model": "Talbot-Lau fringe + Poisson noise; demodulation baseline",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── TERAHERTZ ─────────────────────────────────────────────────────────────
def gen_terahertz(rng: np.random.Generator):
    """
    x_true: (128,128) THz absorption coefficient map.
    y: (128,128) THz transmission image (exp(-x_true)) + noise.
    H_ideal: (1,) THz wavelength [mm, float32].
    reconstruction_baseline: -log(y) absorption recovery.
    """
    size = 128

    # Absorption coefficient map: sparse features (0..2 range)
    x_true = make_blob_phantom(rng, size=size) * float(rng.uniform(0.5, 2.0))
    x_true = x_true.astype(np.float32)

    # Transmission: Beer-Lambert law
    transmission = np.exp(-x_true)

    # THz detector noise
    noise_std = float(rng.uniform(0.005, 0.03))
    y = (transmission + rng.normal(0, noise_std, transmission.shape)).astype(np.float32)
    y = np.clip(y, 0.01, 1.5)  # physical bounds: keep positive, allow >1 for noise

    # THz wavelength (typical: 0.1-3 THz → lambda ~0.1-3 mm)
    thz_freq_thz = float(rng.uniform(0.3, 2.5))
    lambda_mm = 299.792 / (thz_freq_thz * 1000)  # c/f in mm

    H_ideal = np.array([lambda_mm], dtype=np.float32)

    # Reconstruction: absorption = -log(T)
    reconstruction_baseline = (-np.log(np.clip(y, 1e-3, None))).astype(np.float32)
    reconstruction_baseline = np.clip(reconstruction_baseline, 0, None)

    true_params = {
        "absorption_coeff_max": float(x_true.max()),
        "thz_frequency_thz": thz_freq_thz,
        "wavelength_mm": lambda_mm,
        "noise_std": noise_std,
        "model": "THz Beer-Lambert transmission + Gaussian noise; -log(T) baseline",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ── THREE-PHOTON ──────────────────────────────────────────────────────────
def gen_three_photon(rng: np.random.Generator):
    """
    x_true: (128,128) 3-photon excitation microscopy deep tissue image.
    y: x_true^3 + Poisson noise (3-photon nonlinear response).
    H_ideal: (1,) excitation order [float32 = 3.0].
    reconstruction_baseline: y^(1/3) cube-root linearization.
    """
    size = 128
    x_true = make_blob_phantom(rng, size=size)

    # 3-photon signal: cubic excitation
    signal_3pm = x_true ** 3  # range 0..1

    # Poisson noise: scale to photon counts
    scale = float(rng.uniform(200, 800))
    y_counts = rng.poisson(signal_3pm * scale).astype(np.float32) / scale
    # Additional read noise
    read_noise_std = float(rng.uniform(0.002, 0.01))
    y = (y_counts + rng.normal(0, read_noise_std, y_counts.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    H_ideal = np.array([3.0], dtype=np.float32)

    # Reconstruction: y^(1/3)
    reconstruction_baseline = (np.power(y + 1e-8, 1.0 / 3.0)).astype(np.float32)
    reconstruction_baseline = _norm01(reconstruction_baseline)

    true_params = {
        "excitation_order": 3,
        "photon_scale": scale,
        "readout_noise_std": read_noise_std,
        "model": "3PM cubic excitation + Poisson noise; cube-root linearization baseline",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    "sonar": gen_sonar,
    "spinning_disk": gen_spinning_disk,
    "srs": gen_srs,
    "stem": gen_stem,
    "stm": gen_stm,
    "streak_camera": gen_streak_camera,
    "structured_light": gen_structured_light,
    "talbot_lau": gen_talbot_lau,
    "terahertz": gen_terahertz,
    "three_photon": gen_three_photon,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_modality(modality: str, gen_fn) -> None:
    print(f"\n=== Generating: {modality} ===")
    for tier, n_samples in TIERS.items():
        tier_dir = make_dirs(modality, tier)
        seed_base = SEED_BASES[tier]
        samples = []
        true_params_all = {}

        for i in range(n_samples):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            sample_data, true_params = gen_fn(rng)
            samples.append(sample_data)
            true_params_all[f"sample_{i:02d}"] = true_params

            # Per-sample image previews in images/sample_NN/
            sample_img_dir = tier_dir / "images" / f"sample_{i:02d}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)

            _save_png(sample_data["x_true"], sample_img_dir / "x_true.png")
            _save_png(sample_data["y"], sample_img_dir / "y.png")
            _save_png(sample_data["H_ideal"], sample_img_dir / "H_ideal.png")
            _save_png(sample_data["reconstruction_baseline"],
                      sample_img_dir / "reconstruction_baseline.png")

        h5_path = write_h5(tier_dir, modality, tier, samples)
        write_spec(tier_dir, modality, tier, n_samples)
        write_true_spec(tier_dir, true_params_all)

        print(f"  [{tier:6s}] {n_samples:2d} samples -> {h5_path.name}")

    print(f"  Done: {modality}")


def verify_outputs() -> bool:
    print("\n=== Verification ===")
    all_ok = True
    for modality in GENERATORS:
        for tier, n_samples in TIERS.items():
            tier_dir = DATASETS_ROOT / modality / tier
            h5_file = tier_dir / f"{modality}_challenge_{tier}.h5"
            spec_file = tier_dir / "spec.json"
            true_spec_file = tier_dir / "true_spec.json"
            images_dir = tier_dir / "images"

            missing = []
            if not h5_file.exists():
                missing.append("h5")
            if not spec_file.exists():
                missing.append("spec.json")
            if not true_spec_file.exists():
                missing.append("true_spec.json")
            if not images_dir.exists():
                missing.append("images/")

            if missing:
                print(f"  FAIL {modality}/{tier}: missing {missing}")
                all_ok = False
                continue

            with h5py.File(h5_file, "r") as f:
                n_groups = len(f.keys())
                s0_keys = list(f["sample_00"].keys()) if "sample_00" in f else []

            expected_keys = {"x_true", "y", "H_ideal", "reconstruction_baseline"}
            if n_groups != n_samples:
                print(f"  WARN {modality}/{tier}: expected {n_samples} groups, got {n_groups}")
                all_ok = False
            elif not expected_keys.issubset(set(s0_keys)):
                print(f"  WARN {modality}/{tier}: missing keys {expected_keys - set(s0_keys)}")
                all_ok = False
            else:
                # Report shapes
                with h5py.File(h5_file, "r") as f:
                    x_shape = f["sample_00"]["x_true"].shape
                    y_shape = f["sample_00"]["y"].shape
                print(f"  OK   {modality}/{tier}: {n_groups} samples "
                      f"x_true={x_shape} y={y_shape}")

    return all_ok


def main():
    print(f"Output root: {DATASETS_ROOT}")
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    for modality, gen_fn in GENERATORS.items():
        generate_modality(modality, gen_fn)

    ok = verify_outputs()
    print("\n" + ("Batch 10 dataset generation COMPLETE." if ok
                  else "Batch 10 generation finished with WARNINGS."))


if __name__ == "__main__":
    main()
