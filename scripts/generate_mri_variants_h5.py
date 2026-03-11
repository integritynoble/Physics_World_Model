#!/usr/bin/env python3
"""
Generate benchmark HDF5 datasets for 6 MRI variant modalities:
  asl_mri, mrs, mra, swi, mr_elastography, mr_fingerprinting

Each modality → 3 tiers (public=12, dev=20, hidden=20 samples).
HDF5: {modality}_challenge_{tier}.h5
Groups: sample_00, sample_01, ...
  Datasets: x_true, y, H_ideal, reconstruction_baseline  (all 256x256 float32)
  Attributes: spec (JSON), true_spec (JSON)

Physics: k-space undersampling via 2D FFT, Shepp-Logan phantoms as base.
Deps: numpy, scipy, h5py, PIL (Pillow)
"""

import os
import json
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = (
    "D:/onedrive/startup/program/physics_world_model/"
    "PWM5/Physics_World_Model/datasets/benchmark"
)
IMG_SIZE = 256
TIERS = {"public": 12, "dev": 20, "hidden": 20}

# Reproducible RNG (seeded per modality+tier+sample for determinism)
GLOBAL_RNG = np.random.default_rng(2025_03_10)


# ── Phantom generators ────────────────────────────────────────────────────────

def shepp_logan_256(rng, variation=0.03):
    """
    Generate a 256x256 Shepp-Logan phantom with small random variations.
    Returns float32 in [0, 1].
    """
    # Hand-coded 10-ellipse Shepp-Logan (normalized coords [-1,1])
    ellipses = [
        # (a, b, x0, y0, angle_deg, intensity)
        (0.69,  0.92,  0.00,  0.00,   0,  2.00),
        (0.6624, 0.8740, 0.00, -0.0184,  0, -0.98),
        (0.11,  0.31,  0.22,  0.00, -18,  -0.02),
        (0.16,  0.41, -0.22,  0.00,  18,  -0.02),
        (0.21,  0.25,  0.00,  0.35,   0,   0.01),
        (0.046, 0.046, 0.00,  0.10,   0,   0.01),
        (0.046, 0.046, 0.00, -0.10,   0,   0.01),
        (0.046, 0.023,-0.08, -0.605,  0,   0.01),
        (0.023, 0.023, 0.00, -0.606,  0,   0.01),
        (0.023, 0.046, 0.06, -0.605,  0,   0.01),
    ]
    N = IMG_SIZE
    lin = np.linspace(-1, 1, N)
    x_grid, y_grid = np.meshgrid(lin, -lin)
    img = np.zeros((N, N), dtype=np.float64)

    for (a, b, x0, y0, angle_deg, intensity) in ellipses:
        theta = np.deg2rad(angle_deg)
        ct, st = np.cos(theta), np.sin(theta)
        xr = ct * (x_grid - x0) + st * (y_grid - y0)
        yr = -st * (x_grid - x0) + ct * (y_grid - y0)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        img[mask] += intensity

    # Add small random perturbation per sample
    img += rng.normal(0, variation, img.shape)
    img = np.clip(img, 0, None)
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def vessel_phantom_256(rng):
    """
    Generate a 256x256 vascular phantom with random curved tube-like vessels.
    Returns float32 in [0, 1].
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    n_vessels = rng.integers(4, 10)
    for _ in range(n_vessels):
        # Random Bezier-like curve via control points
        n_pts = rng.integers(3, 7)
        cx = rng.uniform(20, IMG_SIZE - 20, n_pts)
        cy = rng.uniform(20, IMG_SIZE - 20, n_pts)
        # Interpolate along curve
        n_interp = 400
        t = np.linspace(0, 1, n_interp)
        # De Casteljau / linear chain (approximate)
        pts_x = np.interp(t, np.linspace(0, 1, n_pts), cx)
        pts_y = np.interp(t, np.linspace(0, 1, n_pts), cy)
        radius = rng.uniform(1.5, 4.5)
        intensity = rng.uniform(0.7, 1.0)
        for px, py in zip(pts_x, pts_y):
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < IMG_SIZE and 0 <= yi < IMG_SIZE:
                img[yi, xi] = intensity

    # Gaussian blur makes tubes
    sigma = rng.uniform(1.2, 2.5)
    img = gaussian_filter(img, sigma=sigma)
    # Normalise
    if img.max() > 1e-8:
        img /= img.max()
    return img.astype(np.float32)


# ── k-space utilities ─────────────────────────────────────────────────────────

def fft2c(img):
    """Centred 2D FFT (img → k-space), complex64."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img))
    ).astype(np.complex64)


def ifft2c(kspace):
    """Centred 2D IFFT (k-space → img), complex64."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace))
    ).astype(np.complex64)


def random_undersample_mask(shape, acceleration, rng, center_fraction=0.08):
    """
    1D variable-density undersampling mask (phase-encode direction).
    Returns (H, W) float32 mask with values 0 or 1.
    """
    H, W = shape
    n_center = max(1, int(round(W * center_fraction)))
    n_keep = max(n_center, W // acceleration)

    mask_1d = np.zeros(W, dtype=np.float32)
    center_start = (W - n_center) // 2
    mask_1d[center_start: center_start + n_center] = 1.0

    outer = np.where(mask_1d == 0)[0]
    n_outer = max(0, n_keep - n_center)
    chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False)
    mask_1d[chosen] = 1.0

    return np.tile(mask_1d, (H, 1))  # (H, W) float32


def golden_angle_mask(shape, n_spokes, rng):
    """
    Radial golden-angle undersampling mask (binary, float32).
    Approximated as random radial lines in 2D k-space.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    golden = 111.246  # degrees
    max_r = min(H, W) // 2
    for k in range(n_spokes):
        angle = np.deg2rad(k * golden % 180)
        for r in range(-max_r, max_r + 1):
            yi = int(round(cy + r * np.sin(angle)))
            xi = int(round(cx + r * np.cos(angle)))
            if 0 <= yi < H and 0 <= xi < W:
                mask[yi, xi] = 1.0
    return mask


def add_complex_noise(kspace, sigma_factor, rng):
    """Add Gaussian complex noise. sigma = sigma_factor * max|kspace|."""
    scale = sigma_factor * (np.abs(kspace).max() + 1e-10)
    noise = rng.normal(0, scale, kspace.shape) + 1j * rng.normal(0, scale, kspace.shape)
    return (kspace + noise.astype(np.complex64)).astype(np.complex64)


# ── HDF5 writer helpers ───────────────────────────────────────────────────────

def abs_float32(arr):
    """|complex| → float32, or pass-through if already real."""
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float32)
    return arr.astype(np.float32)


def write_sample(grp, x_true, y_complex, H_ideal_mask, spec_dict, true_spec_dict):
    """
    Write one sample group to an open HDF5 file.
    All spatial arrays stored as float32 256x256.
    y stored as (256,256,2) float32 [real, imag].
    H_ideal stored as (256,256) float32 mask.
    reconstruction_baseline = |IFFT(y)|.
    """
    # reconstruction_baseline
    y_kspace = y_complex.astype(np.complex64)
    baseline = abs_float32(ifft2c(y_kspace))

    # y as real/imag stacked → but spec says (256,256) complex64
    # We store as (256,256,2) float32 consistent with the rest of the framework
    y_ri = np.stack([y_kspace.real, y_kspace.imag], axis=-1).astype(np.float32)
    H_float = H_ideal_mask.astype(np.float32)

    grp.create_dataset("x_true",                  data=x_true.astype(np.float32),  compression="gzip")
    grp.create_dataset("y",                        data=y_ri,                        compression="gzip")
    grp.create_dataset("H_ideal",                  data=H_float,                     compression="gzip")
    grp.create_dataset("reconstruction_baseline",  data=baseline,                    compression="gzip")
    grp.attrs["spec"]      = json.dumps(spec_dict)
    grp.attrs["true_spec"] = json.dumps(true_spec_dict)


def open_h5(out_dir, modality, tier):
    """Create output directory and open HDF5 for writing."""
    tier_dir = os.path.join(out_dir, modality, tier)
    os.makedirs(tier_dir, exist_ok=True)
    fpath = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
    hf = h5py.File(fpath, "w")
    hf.attrs["modality"] = modality
    hf.attrs["tier"]     = tier
    hf.attrs["version"]  = "1.0"
    hf.attrs["img_size"] = IMG_SIZE
    hf.attrs["generator"] = "generate_mri_variants_h5.py"
    return hf, fpath


# ── PNG preview helpers ───────────────────────────────────────────────────────

def save_png(arr_f32, path):
    """Save float32 [0,1] array as 8-bit grayscale PNG."""
    arr_clipped = np.clip(arr_f32, 0, 1)
    pil_img = Image.fromarray((arr_clipped * 255).astype(np.uint8))
    pil_img.save(path)


def make_preview(out_dir, modality, tier, sample_idx, x_true, y_kspace, baseline):
    img_dir = os.path.join(out_dir, modality, "images")
    os.makedirs(img_dir, exist_ok=True)
    prefix = f"{modality}_{tier}_s{sample_idx:02d}"
    save_png(x_true, os.path.join(img_dir, f"{prefix}_xtrue.png"))
    # log-magnitude of k-space
    logmag = np.log1p(np.abs(y_kspace))
    logmag /= logmag.max() + 1e-8
    save_png(logmag, os.path.join(img_dir, f"{prefix}_kspace.png"))
    save_png(baseline, os.path.join(img_dir, f"{prefix}_baseline.png"))


# ── Modality generators ───────────────────────────────────────────────────────

# ---------- 1. ASL MRI -------------------------------------------------------
def generate_asl_mri_sample(rng, tier, sample_idx):
    """
    CBF map → BOLD-modulated → FFT → 4x undersample → noise.
    """
    # Spec params (true = what was actually used)
    TI        = float(rng.uniform(1.0, 2.5))
    T1_blood  = float(rng.uniform(1.6, 1.9))
    lab_eff   = float(rng.uniform(0.70, 0.95))

    # x_true: CBF map from Shepp-Logan
    x_true = shepp_logan_256(rng, variation=0.04)  # [0,1]

    # BOLD-like modulation
    x_eff = x_true * (1.0 - 0.3 * np.exp(-TI / 1.2))
    x_eff = np.clip(x_eff, 0, 1).astype(np.complex64)

    # Full k-space
    kspace_full = fft2c(x_eff)

    # Undersampling mask (4x acceleration)
    acceleration = 4
    if tier == "hidden":
        acceleration = int(rng.choice([4, 6, 8]))
    mask = random_undersample_mask((IMG_SIZE, IMG_SIZE), acceleration, rng, center_fraction=0.08)

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "TI_s":              round(TI, 4),
        "T1_blood_s":        round(T1_blood, 4),
        "labeling_efficiency": round(lab_eff, 4),
        "acceleration":      acceleration,
    }
    true_spec = dict(spec)  # same for non-hidden tiers; hidden adds mismatch

    return x_true, kspace_noisy, mask, spec, true_spec


# ---------- 2. MRS -----------------------------------------------------------
def generate_mrs_sample(rng, tier, sample_idx):
    """
    Metabolite concentration map → k-space → undersample.
    """
    TE_ms          = float(rng.uniform(10, 288))
    spectral_width = float(rng.uniform(1000, 4000))
    n_avg          = int(rng.integers(1, 129))

    # x_true: metabolite map (Shepp-Logan variant)
    x_true = shepp_logan_256(rng, variation=0.05)

    kspace_full = fft2c(x_true.astype(np.complex64))

    acceleration = 2
    if tier == "hidden":
        acceleration = int(rng.choice([2, 3, 4]))
    mask = random_undersample_mask((IMG_SIZE, IMG_SIZE), acceleration, rng, center_fraction=0.10)

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "TE_ms":           round(TE_ms, 2),
        "spectral_width_hz": round(spectral_width, 1),
        "n_averages":      n_avg,
        "acceleration":    acceleration,
    }
    true_spec = dict(spec)

    return x_true, kspace_noisy, mask, spec, true_spec


# ---------- 3. MRA -----------------------------------------------------------
def generate_mra_sample(rng, tier, sample_idx):
    """
    Vessel phantom → FFT → 2x undersample → noise.
    """
    flip_angle = float(rng.uniform(10, 60))
    TR_ms      = float(rng.uniform(5, 30))
    TE_ms      = float(rng.uniform(1, 7))

    # x_true: vessel phantom
    x_true = vessel_phantom_256(rng)

    kspace_full = fft2c(x_true.astype(np.complex64))

    acceleration = 2
    if tier == "hidden":
        acceleration = int(rng.choice([2, 3, 4]))
    mask = random_undersample_mask((IMG_SIZE, IMG_SIZE), acceleration, rng, center_fraction=0.12)

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "flip_angle_deg": round(flip_angle, 2),
        "TR_ms":          round(TR_ms, 2),
        "TE_ms":          round(TE_ms, 3),
        "acceleration":   acceleration,
    }
    true_spec = dict(spec)

    return x_true, kspace_noisy, mask, spec, true_spec


# ---------- 4. SWI -----------------------------------------------------------
def generate_swi_sample(rng, tier, sample_idx):
    """
    Susceptibility map (magnitude+phase combined) → FFT → 2x undersample.
    """
    TE_ms           = float(rng.uniform(20, 40))
    B0_Tesla        = float(rng.choice([1.5, 3.0, 7.0]))
    hpf_sigma       = float(rng.uniform(3, 20))

    # Base anatomy from Shepp-Logan
    mag = shepp_logan_256(rng, variation=0.04)

    # Susceptibility phase: simulate vein-like structures with negative susceptibility
    phase_base = shepp_logan_256(rng, variation=0.02)
    # High-pass filter (background removal) by subtracting Gaussian blurred version
    phase_smooth = gaussian_filter(phase_base, sigma=hpf_sigma)
    phase_hpf = phase_base - phase_smooth
    # Normalise phase to [-pi, pi] range scaled
    ptp_val = float(phase_hpf.max() - phase_hpf.min())
    if ptp_val > 1e-8:
        phase_hpf = phase_hpf / ptp_val * 2 * np.pi
    phase_hpf = phase_hpf.astype(np.float32)

    # SWI combined image: magnitude * phase-mask (phase-mask from negative phase voxels)
    phase_mask = np.where(phase_hpf < 0, (phase_hpf / np.pi + 1.0), 1.0).astype(np.float32)
    x_true = (mag * phase_mask)
    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    kspace_full = fft2c(x_true.astype(np.complex64))

    acceleration = 2
    if tier == "hidden":
        acceleration = int(rng.choice([2, 3]))
        TE_ms = float(rng.uniform(20, 40))
        B0_Tesla = float(rng.choice([1.5, 3.0, 7.0]))
    mask = random_undersample_mask((IMG_SIZE, IMG_SIZE), acceleration, rng, center_fraction=0.10)

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "TE_ms":               round(TE_ms, 2),
        "B0_Tesla":            round(B0_Tesla, 1),
        "phase_mask_hpf_sigma": round(hpf_sigma, 2),
        "acceleration":        acceleration,
    }
    true_spec = dict(spec)

    return x_true, kspace_noisy, mask, spec, true_spec


# ---------- 5. MR Elastography -----------------------------------------------
def generate_mr_elastography_sample(rng, tier, sample_idx):
    """
    Shear stiffness map (normalised kPa) → FFT of displacement field → undersample.
    Displacement field: stiffness map modulated by sinusoidal shear wave.
    """
    actuation_freq = float(rng.uniform(50, 100))
    TR_ms          = float(rng.uniform(20, 100))
    meg_grad       = float(rng.uniform(10, 50))   # mT/m

    # x_true: shear stiffness map (normalised, [0,1])
    # Harder tissues = brighter; soft tissues = darker
    stiffness_base = shepp_logan_256(rng, variation=0.05)
    # Scale to simulate kPa range [0.5, 8 kPa] → normalise
    stiffness_kPa = 0.5 + stiffness_base * 7.5
    x_true = (stiffness_kPa - 0.5) / 7.5  # back to [0,1]
    x_true = x_true.astype(np.float32)

    # Displacement field: shear wave at actuation_freq
    lin = np.linspace(0, 2 * np.pi, IMG_SIZE)
    wave_x, wave_y = np.meshgrid(lin, lin)
    k_wave = 2 * np.pi * actuation_freq / (1000 * 3.0)  # approx spatial freq
    displacement = x_true * np.sin(k_wave * (wave_x + wave_y)).astype(np.float32)
    displacement = displacement.astype(np.complex64)

    kspace_full = fft2c(displacement)

    acceleration = 3
    if tier == "hidden":
        acceleration = int(rng.choice([3, 4, 5]))
    mask = random_undersample_mask((IMG_SIZE, IMG_SIZE), acceleration, rng, center_fraction=0.08)

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "actuation_freq_hz":     round(actuation_freq, 1),
        "TR_ms":                 round(TR_ms, 2),
        "motion_encode_grad_mT_per_m": round(meg_grad, 2),
        "acceleration":          acceleration,
    }
    true_spec = dict(spec)

    return x_true, kspace_noisy, mask, spec, true_spec


# ---------- 6. MR Fingerprinting --------------------------------------------
def generate_mr_fingerprinting_sample(rng, tier, sample_idx):
    """
    T1 map → k-space sequence with golden-angle undersampling.
    """
    n_TR        = int(rng.integers(500, 3001))
    fa_var_deg  = float(rng.uniform(5, 90))
    TR_var_ms   = float(rng.uniform(5, 20))

    # x_true: T1 map normalised [0,1] (T1 range 0.3-3.0s)
    T1_base = shepp_logan_256(rng, variation=0.04)
    T1_s    = 0.3 + T1_base * 2.7  # [0.3, 3.0] s
    x_true  = (T1_s - 0.3) / 2.7   # normalise to [0,1]
    x_true  = x_true.astype(np.float32)

    kspace_full = fft2c(x_true.astype(np.complex64))

    # Golden-angle radial spokes — n_spokes scales with n_TR
    # For 256x256 we approximate with a manageable number of spokes
    n_spokes = max(32, min(n_TR // 10, 128))
    mask = golden_angle_mask((IMG_SIZE, IMG_SIZE), n_spokes, rng)
    # Ensure mask is binary float32
    mask = (mask > 0).astype(np.float32)

    if tier == "hidden":
        # vary flip angle pattern
        fa_var_deg = float(rng.uniform(5, 90))

    kspace_under = (kspace_full * mask).astype(np.complex64)
    kspace_noisy = add_complex_noise(kspace_under, 0.01, rng)

    spec = {
        "n_TR":                 n_TR,
        "flip_angle_variation_deg": round(fa_var_deg, 2),
        "TR_variation_ms":      round(TR_var_ms, 3),
        "n_spokes":             n_spokes,
    }
    true_spec = dict(spec)

    return x_true, kspace_noisy, mask, spec, true_spec


# ── Master generator mapping ──────────────────────────────────────────────────

MODALITY_GENERATORS = {
    "asl_mri":          generate_asl_mri_sample,
    "mrs":              generate_mrs_sample,
    "mra":              generate_mra_sample,
    "swi":              generate_swi_sample,
    "mr_elastography":  generate_mr_elastography_sample,
    "mr_fingerprinting": generate_mr_fingerprinting_sample,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all():
    out_dir = ROOT
    preview_modalities = list(MODALITY_GENERATORS.keys())  # save PNGs for all

    total_files = 0
    total_bytes = 0

    for modality, gen_fn in MODALITY_GENERATORS.items():
        print(f"\n{'='*60}")
        print(f"  Modality: {modality}")
        print(f"{'='*60}")

        for tier, n_samples in TIERS.items():
            # Seed per (modality, tier) for reproducibility
            seed = abs(hash(modality + tier)) % (2**31)
            rng = np.random.default_rng(seed)

            hf, fpath = open_h5(out_dir, modality, tier)
            print(f"  [{tier}] {n_samples} samples -> {fpath}")

            for i in range(n_samples):
                x_true, kspace_noisy, mask, spec, true_spec = gen_fn(rng, tier, i)

                grp = hf.create_group(f"sample_{i:02d}")
                write_sample(grp, x_true, kspace_noisy, mask, spec, true_spec)

                # Save PNG previews for first 3 samples of public tier
                if tier == "public" and i < 3:
                    baseline = abs_float32(ifft2c(kspace_noisy))
                    make_preview(out_dir, modality, tier, i, x_true, kspace_noisy, baseline)

            hf.close()
            sz = os.path.getsize(fpath)
            total_bytes += sz
            total_files += 1
            print(f"    Saved ({sz/1e6:.2f} MB)")

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {total_files} HDF5 files, {total_bytes/1e6:.1f} MB total")
    print(f"  Root: {out_dir}".encode("ascii", "replace").decode("ascii"))
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_all()
