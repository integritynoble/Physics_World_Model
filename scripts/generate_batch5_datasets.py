#!/usr/bin/env python3
"""
Generate benchmark datasets for batch 5 modalities:
  flash_lidar, flim, fluoroscopy, fwi, gpr, gravitational_wave,
  hdr_imaging, impedance_tomo, integral, ism

Each modality: 3 tiers (public=12, dev=20, hidden=20 samples)
Each tier: {modality}_challenge_{tier}.h5, spec.json, true_spec.json, images/

Seeds: public = 5000 + i*17, dev = 8700 + i*17, hidden = 9850 + i*17
"""

import io
import json
import os
import sys

import h5py
import numpy as np

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

try:
    from scipy import signal, ndimage
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
except ImportError:
    raise ImportError("pip install scipy")

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model"
BENCH_DIR = os.path.join(ROOT, "datasets", "benchmark")

TIER_SIZES = {"public": 12, "dev": 20, "hidden": 20}
TIER_SEEDS = {"public": 5000, "dev": 8700, "hidden": 9850}
SEED_STEP = 17

VERSION = "1.0"


# ── Helper utilities ──────────────────────────────────────────────────────────

def save_png(arr2d: np.ndarray, path: str):
    """Save 2D float32 array as normalised 8-bit PNG."""
    a = arr2d.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    img = Image.fromarray((a * 255).astype(np.uint8))
    img.save(path)


def make_dirs(out_dir: str, tier: str):
    tier_dir = os.path.join(out_dir, tier)
    img_dir = os.path.join(tier_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return tier_dir, img_dir


def random_blobs(rng, size=128, n_blobs=5, min_r=5, max_r=20) -> np.ndarray:
    """Random Gaussian blobs on a 2D field."""
    field = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_blobs):
        cy = rng.integers(max_r, size - max_r)
        cx = rng.integers(max_r, size - max_r)
        r = rng.uniform(min_r, max_r)
        amp = rng.uniform(0.3, 1.0)
        field += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    return (field / (field.max() + 1e-8)).astype(np.float32)


def layered_medium(rng, size=128, n_layers=4) -> np.ndarray:
    """Horizontal layers with random values and noise."""
    field = np.zeros((size, size), dtype=np.float32)
    boundaries = np.sort(rng.integers(10, size - 10, size=n_layers - 1))
    vals = rng.uniform(0.2, 1.0, size=n_layers).astype(np.float32)
    prev = 0
    for k, b in enumerate(boundaries):
        field[prev:b, :] = vals[k]
        prev = b
    field[prev:, :] = vals[-1]
    field += rng.normal(0, 0.03, field.shape).astype(np.float32)
    return np.clip(field, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FLASH LIDAR
# ═══════════════════════════════════════════════════════════════════════════════

def gen_flash_lidar_sample(rng, i: int):
    """
    x_true: (128,128) depth map in [0,10] m
    y:      (128,128) ToF histogram peak with Poisson noise (uint16-like float32)
    H_ideal: scalar float32 scalar stored as (1,) array — pulse_width in ns
    reconstruction_baseline: y normalised to [0,10]
    """
    SIZE = 128
    x_true = random_blobs(rng, SIZE, n_blobs=rng.integers(3, 8)) * 10.0  # metres

    pulse_width_ns = float(rng.uniform(0.5, 3.0))

    # ToF peak intensity: I ~ exp(-depth/lambda) * photon_count
    photon_scale = rng.uniform(200, 1000)
    intensity = photon_scale * np.exp(-x_true / 5.0)
    y = rng.poisson(np.clip(intensity, 0, None)).astype(np.float32)

    H_ideal = np.array([pulse_width_ns], dtype=np.float32)

    # Reconstruction: normalise y to [0,10]
    y_max = y.max() + 1e-8
    recon = (y / y_max) * 10.0
    recon = recon.astype(np.float32)

    spec_entry = {
        "pulse_width_ns": pulse_width_ns,
        "photon_scale": float(photon_scale),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true.astype(np.float32), y, H_ideal, recon, spec_entry


def generate_flash_lidar(out_dir: str):
    modality = "flash_lidar"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "pulse_width_ns": {"min": 0.5, "max": 3.0, "unit": "ns"},
        "depth_range_m": {"min": 0.0, "max": 10.0, "unit": "m"},
        "photon_scale": {"min": 200, "max": 1000, "unit": "photons"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128,
                "forward_model": "ToF peak intensity = I0*exp(-depth/5) + Poisson noise",
                "baseline_method": "y normalised to depth range [0,10]m",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_flash_lidar_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                # Preview
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y, os.path.join(img_dir, f"sample_{i:02d}_y.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FLIM
# ═══════════════════════════════════════════════════════════════════════════════

def gen_flim_sample(rng, i: int):
    """
    x_true: (128,128) fluorescence lifetime map in [0.5, 4] ns
    y:      (128,128,16) time-gated images (16 time bins)
    H_ideal: (16,) time bin centres in ns
    reconstruction_baseline: (128,128) average weighted lifetime from y
    """
    SIZE = 128
    N_GATES = 16
    TAU_MIN, TAU_MAX = 0.5, 4.0  # ns

    # Lifetime map: blobs with different lifetimes
    tau_map = 0.5 + random_blobs(rng, SIZE, n_blobs=rng.integers(3, 7)) * 3.5
    tau_map = tau_map.astype(np.float32)

    # Time bins: 0 to 12 ns
    t_bins = np.linspace(0, 12.0, N_GATES + 1)
    t_centres = 0.5 * (t_bins[:-1] + t_bins[1:])  # (16,)
    H_ideal = t_centres.astype(np.float32)

    # Time-gated intensity: I(t) = I0 * exp(-t/tau)
    photons = rng.uniform(100, 1000)
    y = np.zeros((SIZE, SIZE, N_GATES), dtype=np.float32)
    for g in range(N_GATES):
        intensity = photons * np.exp(-t_centres[g] / tau_map)
        y[:, :, g] = rng.poisson(np.clip(intensity, 0, None)).astype(np.float32)

    # Reconstruction: weighted average lifetime
    t_arr = t_centres[np.newaxis, np.newaxis, :]  # (1,1,16)
    y_sum = y.sum(axis=2) + 1e-8
    recon = (y * t_arr).sum(axis=2) / y_sum
    recon = np.clip(recon, TAU_MIN, TAU_MAX).astype(np.float32)

    spec_entry = {
        "photons_per_pixel": float(photons),
        "n_time_gates": N_GATES,
        "time_range_ns": 12.0,
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return tau_map, y, H_ideal, recon, spec_entry


def generate_flim(out_dir: str):
    modality = "flim"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "lifetime_range_ns": {"min": 0.5, "max": 4.0, "unit": "ns"},
        "n_time_gates": {"value": 16},
        "time_range_ns": {"value": 12.0, "unit": "ns"},
        "photons_per_pixel": {"min": 100, "max": 1000, "unit": "photons"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128, "n_time_gates": 16,
                "forward_model": "I(t) = I0*exp(-t/tau) + Poisson; 16 time gates over 12ns",
                "baseline_method": "intensity-weighted average time = estimated lifetime",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_flim_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y[:, :, 0], os.path.join(img_dir, f"sample_{i:02d}_y_gate0.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FLUOROSCOPY
# ═══════════════════════════════════════════════════════════════════════════════

def gen_fluoroscopy_sample(rng, i: int):
    """
    x_true: (128,128) X-ray linear attenuation map [0.1, 2.0] cm^-1
    y:      (128,128) Beer-Lambert + Poisson noisy log-projected image
    H_ideal: (128,128) noiseless log projection
    reconstruction_baseline: direct log-linearisation of y
    """
    SIZE = 128
    # Attenuation map: tissue-like blobs
    base = rng.uniform(0.05, 0.15)
    x_true = base + random_blobs(rng, SIZE, n_blobs=rng.integers(3, 8)) * 1.9
    x_true = x_true.astype(np.float32)

    I0 = rng.uniform(5000, 20000)  # incident photon count
    # Beer-Lambert: I = I0 * exp(-mu * L), L=1cm per pixel thickness
    I_noiseless = I0 * np.exp(-x_true)
    H_ideal = np.log(I0 / (I_noiseless + 1e-8)).astype(np.float32)  # noiseless log

    # Add Poisson noise
    I_noisy = rng.poisson(np.clip(I_noiseless, 0, None)).astype(np.float32)
    I_noisy = np.clip(I_noisy, 1, None)
    y = np.log(I0 / I_noisy).astype(np.float32)  # measured log projection

    # Reconstruction: direct log-linearisation
    recon = y.copy()

    spec_entry = {
        "I0_photons": float(I0),
        "mean_attenuation": float(x_true.mean()),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, y, H_ideal, recon, spec_entry


def generate_fluoroscopy(out_dir: str):
    modality = "fluoroscopy"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "attenuation_range_cm-1": {"min": 0.1, "max": 2.0, "unit": "cm^-1"},
        "I0_photons": {"min": 5000, "max": 20000, "unit": "photons"},
        "pixel_thickness_cm": {"value": 1.0, "unit": "cm"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128,
                "forward_model": "I = I0*exp(-mu) + Poisson; y = log(I0/I_noisy)",
                "baseline_method": "direct log-linearisation: recon = y",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_fluoroscopy_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y, os.path.join(img_dir, f"sample_{i:02d}_y.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FWI (Full Waveform Inversion)
# ═══════════════════════════════════════════════════════════════════════════════

def ricker_wavelet(freq_hz: float, dt: float, n: int) -> np.ndarray:
    """Ricker (Mexican hat) wavelet."""
    t = np.arange(n) * dt - 1.0 / freq_hz
    w = (1 - 2 * (np.pi * freq_hz * t) ** 2) * np.exp(-(np.pi * freq_hz * t) ** 2)
    return w.astype(np.float32)


def gen_fwi_sample(rng, i: int):
    """
    x_true: (128,128) seismic velocity model [1500, 4500] m/s
    y:      (64,128) seismic wavefield traces (convolve x_true rows with Ricker)
    H_ideal: (64,) dominant frequency values (Hz)
    reconstruction_baseline: cross-correlation migration (FFT-based)
    """
    SIZE = 128
    N_TRACES = 64

    # Velocity model: layered with perturbations
    v_min, v_max = 1500.0, 4500.0
    v_model = layered_medium(rng, SIZE) * (v_max - v_min) + v_min
    v_model = v_model.astype(np.float32)

    # Ricker wavelet parameters
    freq_hz = float(rng.uniform(10, 50))  # dominant frequency
    dt = 0.004  # seconds
    n_t = SIZE
    wavelet = ricker_wavelet(freq_hz, dt, n_t)

    # Forward model: convolve each row of velocity model with Ricker wavelet
    # Sample N_TRACES rows uniformly
    row_indices = np.linspace(0, SIZE - 1, N_TRACES, dtype=int)
    traces = np.zeros((N_TRACES, SIZE), dtype=np.float32)
    for k, row in enumerate(row_indices):
        # Normalised reflectivity from velocity
        refl = v_model[row, :] / v_max
        conv = np.convolve(refl, wavelet, mode="same")
        noise_level = float(rng.uniform(0.01, 0.05))
        noise = rng.normal(0, noise_level, conv.shape).astype(np.float32)
        traces[k, :] = (conv + noise).astype(np.float32)

    H_ideal = np.full(N_TRACES, freq_hz, dtype=np.float32)

    # Reconstruction: simple cross-correlation migration (IFFT of spectrum)
    Y = np.fft.fft2(traces)
    recon_traces = np.real(np.fft.ifft2(np.conj(Y) * Y)).astype(np.float32)
    # Map back to (128,128) via interpolation
    recon = np.zeros((SIZE, SIZE), dtype=np.float32)
    for k, row in enumerate(row_indices):
        recon[row, :] = recon_traces[k, :]
    # Fill gaps by nearest-neighbour
    for r in range(SIZE):
        nearest = row_indices[np.argmin(np.abs(row_indices - r))]
        k = list(row_indices).index(nearest)
        recon[r, :] = recon_traces[k, :]

    # Normalise recon to velocity range
    r_min, r_max = recon.min(), recon.max()
    if r_max > r_min:
        recon = (recon - r_min) / (r_max - r_min) * (v_max - v_min) + v_min
    recon = recon.astype(np.float32)

    spec_entry = {
        "dominant_freq_hz": freq_hz,
        "dt_s": dt,
        "n_traces": N_TRACES,
        "v_min_ms": v_min,
        "v_max_ms": v_max,
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return v_model, traces, H_ideal, recon, spec_entry


def generate_fwi(out_dir: str):
    modality = "fwi"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "velocity_range_ms": {"min": 1500, "max": 4500, "unit": "m/s"},
        "dominant_freq_hz": {"min": 10, "max": 50, "unit": "Hz"},
        "dt_s": {"value": 0.004, "unit": "s"},
        "n_traces": {"value": 64},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128,
                "forward_model": "seismic traces = conv(v_model_rows, Ricker_wavelet) + noise",
                "baseline_method": "cross-correlation migration via FFT",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_fwi_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y, os.path.join(img_dir, f"sample_{i:02d}_y.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GPR (Ground Penetrating Radar)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_gpr_sample(rng, i: int):
    """
    x_true: (128,128) subsurface dielectric permittivity map [1.0, 10.0]
    y:      (128,128) GPR B-scan with hyperbolic reflections
    H_ideal: (128,) depth-to-time conversion factor per column
    reconstruction_baseline: Stolt migration (FFT-based f-k migration)
    """
    SIZE = 128

    # Subsurface model: layered + embedded objects
    x_true = layered_medium(rng, SIZE, n_layers=rng.integers(3, 7))
    x_true = (x_true * 9.0 + 1.0).astype(np.float32)  # scale to [1, 10]

    # Add some buried point scatterers (hyperbola sources)
    n_scatterers = rng.integers(2, 6)
    scat_depths = rng.integers(10, SIZE - 20, size=n_scatterers)
    scat_x = rng.integers(10, SIZE - 10, size=n_scatterers)
    scat_amp = rng.uniform(0.5, 2.0, size=n_scatterers)

    # Velocity in medium (proportional to 1/sqrt(eps))
    v0 = 3e8  # m/s
    eps_mean = float(x_true.mean())
    velocity = v0 / np.sqrt(eps_mean)

    # GPR B-scan: sum of hyperbolic arrivals
    dt = 0.5e-9  # 0.5 ns time step
    dx = 0.05    # 5 cm spatial step
    b_scan = np.zeros((SIZE, SIZE), dtype=np.float32)

    # Ricker wavelet for GPR
    freq_hz = float(rng.uniform(50e6, 500e6))  # 50-500 MHz
    wavelet = ricker_wavelet(freq_hz * 1e-9, dt * 1e9, SIZE)  # work in ns

    for s in range(n_scatterers):
        d = scat_depths[s] * dx
        xs = scat_x[s] * dx
        for col in range(SIZE):
            x_col = col * dx
            dist = 2 * np.sqrt(d ** 2 + (x_col - xs) ** 2) / velocity
            t_idx = int(dist / dt)
            if 0 <= t_idx < SIZE:
                b_scan[t_idx, col] += scat_amp[s]

    # Convolve each column with Ricker wavelet
    for col in range(SIZE):
        b_scan[:, col] = np.convolve(b_scan[:, col], wavelet, mode="same")

    noise_std = float(rng.uniform(0.01, 0.05))
    b_scan += rng.normal(0, noise_std, b_scan.shape).astype(np.float32)
    b_scan = b_scan.astype(np.float32)

    H_ideal = np.full(SIZE, float(dt / dx), dtype=np.float32)

    # Stolt migration: f-k domain phase shift
    # Simplified: 2D FFT, apply phase correction, IFFT
    Y = fft2(b_scan)
    fx = fftshift(np.fft.fftfreq(SIZE, d=dx))
    ft = fftshift(np.fft.fftfreq(SIZE, d=dt))
    FX, FT = np.meshgrid(fx, ft)
    # Phase shift: migrate to depth
    phase = np.exp(1j * 2 * np.pi * np.sqrt(np.maximum((FT / velocity) ** 2 - FX ** 2, 0)))
    recon = np.real(ifft2(ifftshift(fftshift(Y) * phase))).astype(np.float32)

    # Normalise to [1, 10]
    r_min, r_max = recon.min(), recon.max()
    if r_max > r_min:
        recon = (recon - r_min) / (r_max - r_min) * 9.0 + 1.0
    recon = recon.astype(np.float32)

    spec_entry = {
        "dominant_freq_MHz": float(freq_hz / 1e6),
        "velocity_ms": float(velocity),
        "n_scatterers": int(n_scatterers),
        "eps_mean": float(eps_mean),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, b_scan, H_ideal, recon, spec_entry


def generate_gpr(out_dir: str):
    modality = "gpr"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "permittivity_range": {"min": 1.0, "max": 10.0, "unit": "dimensionless"},
        "center_freq_MHz": {"min": 50, "max": 500, "unit": "MHz"},
        "spatial_step_cm": {"value": 5.0, "unit": "cm"},
        "time_step_ns": {"value": 0.5, "unit": "ns"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128,
                "forward_model": "GPR B-scan = sum of hyperbolic arrivals conv Ricker + noise",
                "baseline_method": "Stolt f-k migration (FFT phase shift)",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_gpr_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y, os.path.join(img_dir, f"sample_{i:02d}_y.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRAVITATIONAL WAVE
# ═══════════════════════════════════════════════════════════════════════════════

def gen_gravitational_wave_sample(rng, i: int):
    """
    x_true: (1,4096) strain waveform (chirp signal, CBC-like)
    y:      (1,4096) strain + AWGN noise
    H_ideal: (1,) SNR value
    reconstruction_baseline: matched filter output (y itself as placeholder)
    """
    N = 4096
    dt = 1.0 / 4096.0  # 1 second at 4096 Hz
    t = np.arange(N) * dt

    # Compact binary coalescence chirp: f(t) = f0 * (1 - t/T_coal)^(-3/8)
    f0 = float(rng.uniform(20.0, 50.0))   # start frequency Hz
    T_coal = float(rng.uniform(0.5, 0.9))  # coalescence time (within window)
    amp = float(rng.uniform(1e-21, 1e-20))

    # Chirp phase: Phi(t) = -2*(pi*f0*T_coal/5) * [(1-t/T_coal)^(5/8) - 1]
    t_merge = T_coal * N * dt
    eps = 1e-6
    tau = np.clip(t_merge - t, eps, None)
    # Instantaneous frequency increases as merger approaches
    f_inst = f0 * (tau / t_merge) ** (-3.0 / 8.0)
    f_inst = np.clip(f_inst, 0, 2000)  # cap at Nyquist
    phase = 2 * np.pi * np.cumsum(f_inst) * dt
    envelope = np.exp(-0.5 * ((t - T_coal) / 0.05) ** 2)  # Gaussian taper at merger
    h = (amp * envelope * np.cos(phase)).astype(np.float32)
    h = h.reshape(1, N)

    snr = float(rng.uniform(8.0, 30.0))
    noise_sigma = float(amp / snr)
    noise = rng.normal(0, noise_sigma, (1, N)).astype(np.float32)
    y = (h + noise).astype(np.float32)

    H_ideal = np.array([snr], dtype=np.float32)

    # Reconstruction: matched filter = cross-correlate y with template h
    # For simplicity, matched filter output = y (the measurement itself)
    recon = y.copy()

    spec_entry = {
        "f0_hz": f0,
        "T_coal_s": T_coal,
        "amplitude": float(amp),
        "snr": snr,
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return h, y, H_ideal, recon, spec_entry


def generate_gravitational_wave(out_dir: str):
    modality = "gravitational_wave"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "start_freq_hz": {"min": 20.0, "max": 50.0, "unit": "Hz"},
        "coalescence_time_s": {"min": 0.5, "max": 0.9, "unit": "s"},
        "snr": {"min": 8.0, "max": 30.0, "unit": "dimensionless"},
        "sample_rate_hz": {"value": 4096, "unit": "Hz"},
        "duration_s": {"value": 1.0, "unit": "s"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "n_samples_waveform": 4096,
                "forward_model": "y = CBC chirp waveform + AWGN(sigma=amp/SNR)",
                "baseline_method": "matched filter (y passed through as baseline)",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_gravitational_wave_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                # Preview: 1D waveform as image (transpose)
                wf_img = np.tile(x_true[0:1, :], (64, 1))
                save_png(wf_img, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                wf_img_y = np.tile(y[0:1, :], (64, 1))
                save_png(wf_img_y, os.path.join(img_dir, f"sample_{i:02d}_y.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HDR IMAGING
# ═══════════════════════════════════════════════════════════════════════════════

def gen_hdr_imaging_sample(rng, i: int):
    """
    x_true: (128,128) HDR radiance map (log-normal, range ~[0.01, 100])
    y:      (3,128,128) 3 LDR exposures (EV -2, 0, +2)
    H_ideal: (3,) exposure values in EV
    reconstruction_baseline: (128,128) exposure fusion (weighted average)
    """
    SIZE = 128

    # HDR radiance map: log-normal spatial distribution
    log_radiance = rng.normal(0, 1.5, (SIZE, SIZE)).astype(np.float32)
    log_radiance += random_blobs(rng, SIZE, n_blobs=rng.integers(3, 7)) * 3 - 1.5
    x_true = np.exp(log_radiance).astype(np.float32)  # range ~[0.003, 90]

    # Camera model: y = clip(radiance * exposure_time, 0, 1)
    ev_stops = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    exposure_times = 2.0 ** ev_stops  # [0.25, 1.0, 4.0]

    y = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    for k, t in enumerate(exposure_times):
        ldr = x_true * t / (x_true.max() + 1e-8)  # normalise first
        # Apply simple camera response (gamma)
        ldr = np.clip(ldr, 0, 1) ** (1.0 / 2.2)
        # Add sensor noise
        noise_std = float(rng.uniform(0.005, 0.02))
        ldr = ldr + rng.normal(0, noise_std, ldr.shape).astype(np.float32)
        y[k] = np.clip(ldr, 0, 1).astype(np.float32)

    H_ideal = ev_stops.copy()

    # Reconstruction: exposure fusion — weight by well-exposed regions
    fused = np.zeros((SIZE, SIZE), dtype=np.float32)
    weight_sum = np.zeros((SIZE, SIZE), dtype=np.float32)
    for k in range(3):
        # Weight by proximity to mid-tone (0.5)
        w = np.exp(-4 * (y[k] - 0.5) ** 2).astype(np.float32)
        # Inverse-gamma to linear
        linear = y[k] ** 2.2
        fused += w * linear
        weight_sum += w
    recon = (fused / (weight_sum + 1e-8)).astype(np.float32)
    # Scale to match x_true range
    recon = recon * x_true.max()

    spec_entry = {
        "ev_stops": list(ev_stops.astype(float)),
        "radiance_max": float(x_true.max()),
        "radiance_min": float(x_true.min()),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, y, H_ideal, recon, spec_entry


def generate_hdr_imaging(out_dir: str):
    modality = "hdr_imaging"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "ev_stops": {"values": [-2.0, 0.0, 2.0], "unit": "EV"},
        "radiance_range": {"min": 0.01, "max": 100.0, "unit": "dimensionless"},
        "gamma": {"value": 2.2},
        "noise_std": {"min": 0.005, "max": 0.02},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128, "n_exposures": 3,
                "forward_model": "LDR_k = clip(x_true * t_k / max, 0,1)^(1/2.2) + noise",
                "baseline_method": "exposure fusion: weighted average by mid-tone proximity",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_hdr_imaging_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(np.log1p(x_true), os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y[1], os.path.join(img_dir, f"sample_{i:02d}_y_ev0.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. IMPEDANCE TOMOGRAPHY (EIT)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_impedance_tomo_sample(rng, i: int):
    """
    x_true: (64,64) conductivity map [0.1, 1.0] S/m
    y:      (32,) boundary voltage measurements (random projection)
    H_ideal: (32, 4096) measurement matrix H
    reconstruction_baseline: (64,64) backprojection: reshape(H^T @ y)
    """
    SIZE = 64
    N_MEAS = 32
    N_PIXELS = SIZE * SIZE  # 4096

    # Conductivity map: background + inclusions
    sigma = np.ones((SIZE, SIZE), dtype=np.float32) * 0.3  # background
    n_inclusions = rng.integers(2, 6)
    for _ in range(n_inclusions):
        cy = rng.integers(5, SIZE - 5)
        cx = rng.integers(5, SIZE - 5)
        r = rng.integers(3, 12)
        val = float(rng.uniform(0.5, 1.0))
        yy, xx = np.ogrid[:SIZE, :SIZE]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        sigma[mask] = val
    x_true = np.clip(sigma, 0.1, 1.0).astype(np.float32)

    # Measurement matrix: simulate electrode patterns (adjacent drive patterns)
    # Use random projections as simplified EIT forward model
    H_seed = int(rng.integers(0, 100000))
    H_rng = np.random.default_rng(H_seed)
    # Boundary electrodes: 16 electrodes on circle
    n_elec = 16
    theta = np.linspace(0, 2 * np.pi, n_elec, endpoint=False)
    elec_y = (SIZE // 2 + (SIZE // 2 - 2) * np.sin(theta)).astype(int)
    elec_x = (SIZE // 2 + (SIZE // 2 - 2) * np.cos(theta)).astype(int)

    # Measurement matrix rows: sensitivity patterns
    H = np.zeros((N_MEAS, N_PIXELS), dtype=np.float32)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for m in range(N_MEAS):
        # Drive pair
        e1 = m % n_elec
        e2 = (m + 1) % n_elec
        # Simplified sensitivity: 1/r^2 from each electrode
        d1 = ((yy - elec_y[e1]) ** 2 + (xx - elec_x[e1]) ** 2 + 1.0) ** (-0.5)
        d2 = ((yy - elec_y[e2]) ** 2 + (xx - elec_x[e2]) ** 2 + 1.0) ** (-0.5)
        sens = (d1 - d2).astype(np.float32)
        H[m, :] = sens.flatten()

    # Forward: y = H @ x_true_flat
    x_flat = x_true.flatten()
    y_clean = H @ x_flat
    noise_std = float(rng.uniform(0.001, 0.01) * np.abs(y_clean).max())
    y = (y_clean + rng.normal(0, noise_std, N_MEAS)).astype(np.float32)

    H_ideal = H  # (32, 4096) — but store as stored matrix

    # Reconstruction: backprojection H^T @ y reshaped
    recon = (H.T @ y).reshape(SIZE, SIZE).astype(np.float32)
    # Normalise to conductivity range
    r_min, r_max = recon.min(), recon.max()
    if r_max > r_min:
        recon = (recon - r_min) / (r_max - r_min) * 0.9 + 0.1
    recon = recon.astype(np.float32)

    spec_entry = {
        "n_electrodes": n_elec,
        "n_measurements": N_MEAS,
        "noise_std_fraction": float(noise_std / (np.abs(y_clean).max() + 1e-8)),
        "n_inclusions": int(n_inclusions),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, y, H_ideal, recon, spec_entry


def generate_impedance_tomo(out_dir: str):
    modality = "impedance_tomo"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "conductivity_range_Sm": {"min": 0.1, "max": 1.0, "unit": "S/m"},
        "n_electrodes": {"value": 16},
        "n_measurements": {"value": 32},
        "image_size": {"value": 64},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 64,
                "forward_model": "y = H @ sigma_flat; H = electrode sensitivity matrix (32x4096)",
                "baseline_method": "backprojection: reshape(H^T @ y, 64, 64)",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_impedance_tomo_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTEGRAL (Integral Imaging / Light Field)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_integral_sample(rng, i: int):
    """
    x_true: (128,128) light field focal stack slice (in-focus scene)
    y:      (4,128,128) 4 refocused views at different depths
    H_ideal: (4,) refocus depths
    reconstruction_baseline: (128,128) average of 4 views
    """
    SIZE = 128
    N_VIEWS = 4

    # Ground truth: sharp scene with depth variation
    x_true = random_blobs(rng, SIZE, n_blobs=rng.integers(4, 9))
    x_true = x_true.astype(np.float32)

    # Depth map for simulating blur
    depth_map = layered_medium(rng, SIZE, n_layers=rng.integers(2, 5))

    # Refocus depths
    refocus_depths = np.linspace(0.0, 1.0, N_VIEWS, dtype=np.float32)
    H_ideal = refocus_depths.copy()

    y = np.zeros((N_VIEWS, SIZE, SIZE), dtype=np.float32)
    for k, d in enumerate(refocus_depths):
        # Defocus blur: sigma proportional to |depth_map - d|
        blur_sigma = np.abs(depth_map - d) * 5.0  # max blur = 5 pixels
        # Apply spatially-varying blur (approximated as average sigma blur)
        avg_sigma = float(blur_sigma.mean())
        if avg_sigma > 0.1:
            blurred = ndimage.gaussian_filter(x_true, sigma=avg_sigma)
        else:
            blurred = x_true.copy()
        noise_std = float(rng.uniform(0.005, 0.02))
        view = blurred + rng.normal(0, noise_std, blurred.shape).astype(np.float32)
        y[k] = np.clip(view, 0, 1).astype(np.float32)

    # Reconstruction: average of views
    recon = y.mean(axis=0).astype(np.float32)

    spec_entry = {
        "n_views": N_VIEWS,
        "refocus_depths": list(refocus_depths.astype(float)),
        "n_blobs": int(rng.integers(4, 9)),
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, y, H_ideal, recon, spec_entry


def generate_integral(out_dir: str):
    modality = "integral"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "n_views": {"value": 4},
        "refocus_depths": {"min": 0.0, "max": 1.0, "unit": "normalised"},
        "max_defocus_sigma_px": {"value": 5.0, "unit": "pixels"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128, "n_views": 4,
                "forward_model": "refocused_view_k = gaussian_blur(x, sigma~|depth-d_k|*5) + noise",
                "baseline_method": "average of 4 refocused views",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_integral_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y[0], os.path.join(img_dir, f"sample_{i:02d}_y_view0.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ISM (Image Scanning Microscopy)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_ism_sample(rng, i: int):
    """
    x_true: (128,128) ISM super-resolved image
    y:      (128,128,9) ISM raw frames (9 detector pixels in 3x3 array)
    H_ideal: (9,2) detector pixel offsets (dx, dy) in pixels
    reconstruction_baseline: (128,128) reassigned sum of detector frames
    """
    SIZE = 128
    N_DET = 9  # 3x3 detector array

    # Ground truth: fine-detail fluorescent structure
    x_true = random_blobs(rng, SIZE, n_blobs=rng.integers(5, 12), min_r=2, max_r=10)
    x_true = x_true.astype(np.float32)

    # PSF parameters
    psf_sigma = float(rng.uniform(1.5, 3.0))  # Airy disk approximation

    # 3x3 detector array offsets: -(psf_sigma) to +(psf_sigma) in x, y
    d_spacing = float(rng.uniform(0.5, 1.0))  # detector spacing in units of PSF sigma
    det_offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            det_offsets.append([dy * d_spacing * psf_sigma,
                                 dx * d_spacing * psf_sigma])
    det_offsets = np.array(det_offsets, dtype=np.float32)  # (9, 2)
    H_ideal = det_offsets.copy()

    # Forward model: each detector pixel collects shifted, blurred image
    y = np.zeros((SIZE, SIZE, N_DET), dtype=np.float32)
    photons = float(rng.uniform(100, 500))
    for d in range(N_DET):
        dy_shift = det_offsets[d, 0]
        dx_shift = det_offsets[d, 1]
        # Blur with PSF
        blurred = ndimage.gaussian_filter(x_true, sigma=psf_sigma)
        # Shift by detector offset
        shifted = ndimage.shift(blurred, [dy_shift, dx_shift], mode="reflect")
        # Scale and add Poisson noise
        frame = rng.poisson(np.clip(shifted * photons, 0, None)).astype(np.float32)
        y[:, :, d] = frame

    # Reconstruction: pixel reassignment sum
    # Shift each frame by half its offset (reassignment) then sum
    recon = np.zeros((SIZE, SIZE), dtype=np.float32)
    for d in range(N_DET):
        dy_shift = det_offsets[d, 0]
        dx_shift = det_offsets[d, 1]
        # Reassign by shifting back by half
        reassigned = ndimage.shift(y[:, :, d], [-dy_shift / 2, -dx_shift / 2], mode="reflect")
        recon += reassigned
    recon = recon.astype(np.float32)

    spec_entry = {
        "psf_sigma_px": psf_sigma,
        "n_detectors": N_DET,
        "detector_spacing_sigma": d_spacing,
        "photons_per_pixel": photons,
        "phantom_seed": int(rng.integers(0, 100000)),
    }
    return x_true, y, H_ideal, recon, spec_entry


def generate_ism(out_dir: str):
    modality = "ism"
    print(f"\n=== {modality} ===")
    spec_ranges = {
        "psf_sigma_px": {"min": 1.5, "max": 3.0, "unit": "pixels"},
        "n_detectors": {"value": 9, "layout": "3x3"},
        "detector_spacing_sigma": {"min": 0.5, "max": 1.0, "unit": "sigma"},
        "photons_per_pixel": {"min": 100, "max": 500, "unit": "photons"},
    }

    for tier, n in TIER_SIZES.items():
        tier_dir, img_dir = make_dirs(out_dir, tier)
        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "modality": modality, "version": VERSION, "tier": tier,
                "n_samples": n, "image_size": 128, "n_detectors": 9,
                "forward_model": "y_d = Poisson(shift(blur(x, psf), offset_d) * photons)",
                "baseline_method": "pixel reassignment: sum of half-shift-corrected frames",
            })
            for i in range(n):
                seed = TIER_SEEDS[tier] + i * SEED_STEP
                rng = np.random.default_rng(seed)
                x_true, y, H_ideal, recon, spec_entry = gen_ism_sample(rng, i)
                grp = hf.create_group(f"sample_{i:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
                grp.create_dataset("reconstruction_baseline", data=recon, compression="gzip")
                true_spec[f"sample_{i:02d}"] = spec_entry
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_png(y[:, :, 4], os.path.join(img_dir, f"sample_{i:02d}_y_center.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon.png"))

        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec_ranges, f, indent=2)
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)
        print(f"  {tier}: {n} samples → {h5_path}")
    print(f"  {modality} done.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    generators = [
        ("flash_lidar",       generate_flash_lidar),
        ("flim",              generate_flim),
        ("fluoroscopy",       generate_fluoroscopy),
        ("fwi",               generate_fwi),
        ("gpr",               generate_gpr),
        ("gravitational_wave", generate_gravitational_wave),
        ("hdr_imaging",       generate_hdr_imaging),
        ("impedance_tomo",    generate_impedance_tomo),
        ("integral",          generate_integral),
        ("ism",               generate_ism),
    ]

    results = {}
    for modality, gen_fn in generators:
        out_dir = os.path.join(BENCH_DIR, modality)
        try:
            gen_fn(out_dir)
            results[modality] = "OK"
        except Exception as e:
            import traceback
            print(f"  ERROR in {modality}: {e}")
            traceback.print_exc()
            results[modality] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for modality, status in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  [{icon}] {modality}: {status}")

    all_ok = all(v == "OK" for v in results.values())
    print(f"\nAll modalities OK: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
