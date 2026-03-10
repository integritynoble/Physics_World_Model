#!/usr/bin/env python3
"""Generate fMRI benchmark datasets (public / dev / hidden).

Forward model:  y = U_Omega * F * (x_baseline + BOLD_activation) + n
  - x_baseline: anatomical brain image (T2*-weighted)
  - BOLD_activation: small signal changes (1-5% of baseline) in activation regions
  - F: 2D Discrete Fourier Transform
  - U_Omega: Cartesian undersampling mask (random phase-encode lines for
             temporal acceleration, centre always fully sampled)
  - n: complex Gaussian noise

Each "sample" represents one time frame of a BOLD fMRI acquisition.

Phantoms are procedural brain slice images with:
  - Gray/white matter contrast (T2*-weighted)
  - CSF-filled ventricles
  - BOLD activation blobs (Gaussian, 5-15 px sigma) in motor/visual cortex
  - Different activation patterns per sample

Mismatch parameters:
  - acceleration_factor: 2-6x undersampling
  - noise_sigma: complex Gaussian noise std
  - field_inhomogeneity: B0 inhomogeneity amplitude (Hz)
  - motion_artifact_amplitude: inter-frame rigid motion (pixels)

CPU baseline: zero-filled IFFT => expected ~20-26 dB PSNR.

Tier seeds: public=0, dev=10000, hidden=20000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 256
OUT_DIR = Path(__file__).resolve().parent

TIER_CONFIGS = {
    "public":  {"n_samples": 12, "seed_offset": 0},
    "dev":     {"n_samples": 20, "seed_offset": 10000},
    "hidden":  {"n_samples": 20, "seed_offset": 20000},
}

# ---------------------------------------------------------------------------
# Mismatch parameter ranges per tier
# ---------------------------------------------------------------------------
SPEC_PUBLIC = {
    "acceleration_factor": {"min": 2.0, "max": 4.0, "unit": "x"},
    "noise_sigma":         {"min": 0.005, "max": 0.015, "unit": ""},
    "field_inhomogeneity": {"min": 0.0, "max": 15.0, "unit": "Hz"},
    "motion_artifact_amplitude": {"min": 0.0, "max": 1.0, "unit": "pixels"},
}

SPEC_DEV = {
    "acceleration_factor": {"min": 2.5, "max": 5.0, "unit": "x"},
    "noise_sigma":         {"min": 0.008, "max": 0.025, "unit": ""},
    "field_inhomogeneity": {"min": 0.0, "max": 30.0, "unit": "Hz"},
    "motion_artifact_amplitude": {"min": 0.0, "max": 2.0, "unit": "pixels"},
}

SPEC_HIDDEN = {
    "acceleration_factor": {"min": 3.0, "max": 6.0, "unit": "x"},
    "noise_sigma":         {"min": 0.010, "max": 0.040, "unit": ""},
    "field_inhomogeneity": {"min": 5.0, "max": 50.0, "unit": "Hz"},
    "motion_artifact_amplitude": {"min": 0.5, "max": 3.0, "unit": "pixels"},
}

TIER_SPECS = {
    "public": SPEC_PUBLIC,
    "dev": SPEC_DEV,
    "hidden": SPEC_HIDDEN,
}


# ============================================================================
# Brain phantom generation
# ============================================================================

def _ellipse_mask(H: int, W: int, cy: float, cx: float,
                  ry: float, rx: float, angle_deg: float = 0.0) -> np.ndarray:
    """Binary mask for a filled, optionally rotated ellipse."""
    yy, xx = np.mgrid[:H, :W]
    yy = (yy - cy).astype(np.float64)
    xx = (xx - cx).astype(np.float64)
    if abs(angle_deg) > 0.01:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        yy_r = cos_t * yy + sin_t * xx
        xx_r = -sin_t * yy + cos_t * xx
        yy, xx = yy_r, xx_r
    return ((yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0).astype(np.float64)


def generate_brain_phantom(rng: np.random.RandomState,
                           H: int = 256, W: int = 256) -> np.ndarray:
    """Generate a T2*-weighted brain slice phantom (no BOLD yet).

    Returns a float64 image in [0, 1] with:
      - skull outline (bright ring)
      - gray matter (mid-intensity cortex)
      - white matter (lower intensity interior)
      - lateral ventricles (CSF, bright)
      - smooth tissue texture
    """
    cx, cy = W / 2, H / 2
    img = np.zeros((H, W), dtype=np.float64)

    # --- skull ellipse (outer boundary) ---
    ry_skull = H * rng.uniform(0.42, 0.46)
    rx_skull = W * rng.uniform(0.38, 0.42)
    angle_skull = rng.uniform(-3, 3)
    skull = _ellipse_mask(H, W, cy, cx, ry_skull, rx_skull, angle_skull)

    # Inner brain (slightly smaller)
    ry_brain = ry_skull - rng.uniform(6, 10)
    rx_brain = rx_skull - rng.uniform(6, 10)
    brain = _ellipse_mask(H, W, cy, cx, ry_brain, rx_brain, angle_skull)

    # Skull ring: bright on T2* (~0.15)
    skull_ring = skull.astype(np.float64) - brain.astype(np.float64)
    skull_ring = np.clip(skull_ring, 0, 1)
    img += skull_ring * 0.15

    # --- Gray matter (cortex) ---
    # Gray matter fills the brain region at intensity ~0.65
    img += brain * 0.65

    # --- White matter (interior ellipse, lower intensity ~0.45) ---
    ry_wm = ry_brain * rng.uniform(0.55, 0.65)
    rx_wm = rx_brain * rng.uniform(0.55, 0.65)
    wm_cy = cy + rng.uniform(-3, 3)
    wm_cx = cx + rng.uniform(-2, 2)
    wm = _ellipse_mask(H, W, wm_cy, wm_cx, ry_wm, rx_wm, angle_skull)
    # Replace gray with white matter intensity
    img[wm > 0.5] = 0.45

    # --- Lateral ventricles (CSF, T2*-bright ~0.90) ---
    # Left ventricle
    vent_ry = rng.uniform(18, 28)
    vent_rx = rng.uniform(6, 10)
    v_angle = rng.uniform(-15, 15)
    vent_L = _ellipse_mask(H, W,
                           cy + rng.uniform(-5, 5),
                           cx - rng.uniform(12, 22),
                           vent_ry, vent_rx, v_angle)
    # Right ventricle (mirror)
    vent_R = _ellipse_mask(H, W,
                           cy + rng.uniform(-5, 5),
                           cx + rng.uniform(12, 22),
                           vent_ry, vent_rx, -v_angle)
    ventricles = np.clip(vent_L + vent_R, 0, 1)
    img[ventricles > 0.5] = 0.90

    # --- Third ventricle (small, midline) ---
    v3 = _ellipse_mask(H, W, cy + rng.uniform(-2, 4), cx,
                       rng.uniform(6, 10), rng.uniform(2, 4), 0)
    img[v3 > 0.5] = 0.88

    # --- Caudate nuclei (gray matter islands in white matter) ---
    for side in [-1, 1]:
        cn_cy = cy + rng.uniform(-8, 0)
        cn_cx = cx + side * rng.uniform(14, 22)
        cn = _ellipse_mask(H, W, cn_cy, cn_cx,
                           rng.uniform(8, 14), rng.uniform(5, 8),
                           rng.uniform(-10, 10))
        img[cn > 0.5] = 0.60

    # --- Thalamus (pair of ellipses) ---
    for side in [-1, 1]:
        th_cy = cy + rng.uniform(2, 10)
        th_cx = cx + side * rng.uniform(6, 12)
        th = _ellipse_mask(H, W, th_cy, th_cx,
                           rng.uniform(10, 16), rng.uniform(8, 12),
                           rng.uniform(-5, 5))
        img[th > 0.5] = 0.55

    # --- Smooth tissue texture ---
    texture = rng.randn(H, W) * 0.02
    texture = gaussian_filter(texture, sigma=4.0)
    img += texture * brain  # only inside brain

    # Ensure brain boundary
    img *= skull
    img = np.clip(img, 0, 1)

    return img


def generate_bold_activation(rng: np.random.RandomState,
                             brain_mask: np.ndarray,
                             H: int = 256, W: int = 256,
                             n_blobs: int | None = None) -> np.ndarray:
    """Generate BOLD activation map (fractional signal change).

    Returns activation map in [0, max_pct] where max_pct ~ 1-5%.
    Blobs are placed in typical fMRI activation regions:
      - Motor cortex (posterior, near midline)
      - Visual cortex (occipital, posterior)
      - Auditory cortex (lateral temporal)
      - Prefrontal cortex (anterior)
    """
    cx, cy = W / 2, H / 2

    if n_blobs is None:
        n_blobs = rng.randint(2, 7)

    activation = np.zeros((H, W), dtype=np.float64)

    # Candidate activation regions (relative to center, in pixels)
    # (dy, dx, description)
    region_centers = [
        # Motor cortex (top, near midline)
        (-rng.uniform(55, 80), rng.uniform(-15, 15)),
        (-rng.uniform(55, 80), rng.uniform(-40, -15)),
        (-rng.uniform(55, 80), rng.uniform(15, 40)),
        # Visual cortex (bottom-posterior)
        (rng.uniform(55, 85), rng.uniform(-25, 25)),
        (rng.uniform(60, 85), rng.uniform(-45, -10)),
        (rng.uniform(60, 85), rng.uniform(10, 45)),
        # Auditory / lateral temporal
        (rng.uniform(-10, 15), -rng.uniform(55, 80)),
        (rng.uniform(-10, 15), rng.uniform(55, 80)),
        # Prefrontal
        (-rng.uniform(30, 55), rng.uniform(-40, 40)),
        # Supplementary motor area
        (-rng.uniform(40, 60), rng.uniform(-10, 10)),
        # Parietal
        (-rng.uniform(20, 45), rng.uniform(-50, -20)),
        (-rng.uniform(20, 45), rng.uniform(20, 50)),
    ]

    # Pick n_blobs from the candidate regions
    chosen = rng.choice(len(region_centers), size=min(n_blobs, len(region_centers)),
                        replace=False)

    yy, xx = np.mgrid[:H, :W]

    for idx in chosen:
        dy, dx = region_centers[idx]
        blob_cy = cy + dy
        blob_cx = cx + dx
        sigma = rng.uniform(5, 15)
        amplitude = rng.uniform(0.01, 0.05)  # 1-5% signal change

        blob = amplitude * np.exp(
            -((yy - blob_cy) ** 2 + (xx - blob_cx) ** 2) / (2 * sigma ** 2)
        )
        activation += blob

    # Mask to brain region only
    activation *= brain_mask

    return activation


# ============================================================================
# Forward model components
# ============================================================================

def image_to_kspace(img: np.ndarray) -> np.ndarray:
    """2D FFT: image domain -> k-space (centered)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def kspace_to_image(ksp: np.ndarray) -> np.ndarray:
    """2D IFFT: k-space -> image domain (centered)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp)))


def generate_cartesian_mask(H: int, W: int, acceleration: float,
                            center_fraction: float = 0.08,
                            seed: int = 42) -> np.ndarray:
    """Generate Cartesian undersampling mask (phase-encode direction = columns).

    Keeps a fully-sampled centre band and randomly selects additional
    phase-encode lines to reach the target acceleration factor.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)

    # Always keep centre k-space lines
    n_center = max(1, int(W * center_fraction))
    c = W // 2
    half = n_center // 2
    mask[:, c - half:c + half] = 1.0

    # Total lines to sample
    n_total = max(n_center, int(W / acceleration))
    n_random = n_total - n_center

    available = [i for i in range(W)
                 if i < c - half or i >= c + half]
    if n_random > 0 and len(available) > 0:
        chosen = rng.choice(available,
                            size=min(n_random, len(available)),
                            replace=False)
        mask[:, chosen] = 1.0

    return mask


def apply_field_inhomogeneity(img: np.ndarray,
                              inhomogeneity_hz: float,
                              rng: np.random.RandomState,
                              te_ms: float = 30.0) -> np.ndarray:
    """Apply B0 field inhomogeneity as a spatially varying phase modulation.

    Simulates EPI geometric distortion from off-resonance effects.
    """
    if abs(inhomogeneity_hz) < 0.1:
        return img.astype(np.complex128)

    H, W = img.shape[-2:]
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(yy, xx, indexing="ij")

    # Smooth, spatially varying B0 map (Hz)
    # Combine linear gradients + quadratic terms
    a = rng.uniform(-0.4, 0.4)
    b = rng.uniform(-0.3, 0.3)
    c = rng.uniform(-0.2, 0.2)
    d = rng.uniform(-0.15, 0.15)
    b0_map = inhomogeneity_hz * (a * Y + b * X + c * Y * X + d * (Y ** 2 - X ** 2))

    # Phase accumulation: phi = 2*pi*f*TE
    te_s = te_ms / 1000.0
    phase = 2 * np.pi * b0_map * te_s
    phase_map = np.exp(1j * phase)

    return img.astype(np.complex128) * phase_map


def apply_motion_artifact(kspace: np.ndarray,
                          amplitude_px: float,
                          rng: np.random.RandomState) -> np.ndarray:
    """Simulate inter-frame rigid-body motion as phase ramp in k-space.

    A translational shift of (dy, dx) pixels introduces a linear phase
    ramp: k_shifted = k * exp(-i 2 pi (ky*dy/H + kx*dx/W)).
    """
    if amplitude_px < 0.01:
        return kspace

    H, W = kspace.shape
    dy = rng.uniform(-amplitude_px, amplitude_px)
    dx = rng.uniform(-amplitude_px, amplitude_px)

    ky = np.fft.fftfreq(H).reshape(-1, 1)
    kx = np.fft.fftfreq(W).reshape(1, -1)

    phase_ramp = np.exp(-1j * 2 * np.pi * (ky * dy + kx * dx))

    # Apply shifted k-space with ifftshift/fftshift to handle centred convention
    kspace_uncentered = np.fft.ifftshift(kspace)
    kspace_shifted = kspace_uncentered * phase_ramp
    return np.fft.fftshift(kspace_shifted)


def forward_model(x_true: np.ndarray,
                  acceleration: float,
                  noise_sigma: float,
                  inhomogeneity_hz: float,
                  motion_px: float,
                  seed: int) -> dict:
    """Full fMRI forward model.

    y = U_Omega * F * (x_true with phase from B0) + n

    Returns dict with y (undersampled k-space), mask, x_true_complex.
    """
    rng = np.random.RandomState(seed)
    H, W = x_true.shape

    # Apply field inhomogeneity (phase modulation)
    x_complex = apply_field_inhomogeneity(x_true, inhomogeneity_hz, rng)

    # Forward: image -> k-space
    kspace_full = image_to_kspace(x_complex)

    # Apply motion artifact
    kspace_full = apply_motion_artifact(kspace_full, motion_px, rng)

    # Generate undersampling mask
    mask = generate_cartesian_mask(H, W, acceleration,
                                   center_fraction=0.08, seed=seed)

    # Undersample
    kspace_under = kspace_full * mask

    # Add complex Gaussian noise (only where mask is nonzero)
    noise = (rng.randn(H, W) + 1j * rng.randn(H, W)) * (noise_sigma / np.sqrt(2))
    kspace_under = kspace_under + noise * mask

    return {
        "y": kspace_under.astype(np.complex64),
        "H_ideal": mask,
        "kspace_full": kspace_full.astype(np.complex64),
        "x_complex": x_complex.astype(np.complex64),
    }


# ============================================================================
# Reconstruction (CPU baseline)
# ============================================================================

def reconstruct_zero_filled(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-filled IFFT reconstruction.

    Standard baseline for accelerated MRI: insert zeros for missing k-space
    lines and apply inverse FFT. Returns magnitude image.
    """
    img = kspace_to_image(y)
    return np.abs(img).astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Simple global SSIM (no windowing, no skimage dependency)."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# ============================================================================
# Image I/O helpers
# ============================================================================

def _to_uint8(arr: np.ndarray, plow: float = 1, phigh: float = 99) -> np.ndarray:
    """Normalise a real-valued array to uint8 using percentile scaling."""
    arr = np.abs(arr) if np.iscomplexobj(arr) else arr.copy()
    pos = arr[arr > 0]
    if pos.size > 0:
        vmin, vmax = np.percentile(pos, [plow, phigh])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return (normed * 255).astype(np.uint8)


def save_image(arr: np.ndarray, path: str, **kw):
    Image.fromarray(_to_uint8(arr, **kw)).save(path)


def save_kspace_image(ksp: np.ndarray, path: str):
    """Save log-magnitude k-space as grayscale PNG."""
    mag = np.abs(ksp)
    log_mag = np.log1p(mag)
    vmin, vmax = log_mag.min(), log_mag.max()
    if vmax <= vmin:
        vmax = vmin + 1
    normed = np.clip((log_mag - vmin) / (vmax - vmin), 0, 1)
    Image.fromarray((normed * 255).astype(np.uint8)).save(path)


def save_mask_image(mask: np.ndarray, path: str):
    """Save undersampling mask as binary PNG."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


# ============================================================================
# Tier generation
# ============================================================================

def generate_tier(tier_name: str,
                  spec: dict,
                  n_samples: int,
                  base_seed: int):
    """Generate all samples for one tier and write HDF5 + images."""
    tier_dir = OUT_DIR / tier_name
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    rng_tier = np.random.RandomState(base_seed)
    h5_path = tier_dir / f"fmri_challenge_{tier_name}.h5"

    true_specs = {}
    all_psnr = []
    all_ssim = []

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["description"] = (
            f"fMRI {tier_name} tier -- BOLD brain phantom with Cartesian "
            f"k-space undersampling. Forward: y = U*F*(x_base + BOLD) + n"
        )
        hf.attrs["variant"] = "fmri"
        hf.attrs["tier"] = tier_name
        hf.attrs["image_size"] = IMG_SIZE
        hf.attrs["spec_ranges"] = json.dumps(spec)

        for si in range(n_samples):
            sample_name = f"sample_{si:02d}"
            grp = hf.create_group(sample_name)

            # --- Generate phantom ---
            phantom_seed = base_seed + si * 137
            rng_phantom = np.random.RandomState(phantom_seed)
            x_baseline = generate_brain_phantom(rng_phantom, IMG_SIZE, IMG_SIZE)
            brain_mask = (x_baseline > 0.05).astype(np.float64)

            # Generate BOLD activation
            activation = generate_bold_activation(rng_phantom, brain_mask,
                                                  IMG_SIZE, IMG_SIZE)

            # x_true = baseline + BOLD activation (signal change is multiplicative)
            x_true = x_baseline * (1.0 + activation)
            x_true = np.clip(x_true, 0, 1).astype(np.float64)

            # --- Sample mismatch parameters ---
            accel = rng_tier.uniform(spec["acceleration_factor"]["min"],
                                     spec["acceleration_factor"]["max"])
            noise_sigma = rng_tier.uniform(spec["noise_sigma"]["min"],
                                           spec["noise_sigma"]["max"])
            inhom = rng_tier.uniform(spec["field_inhomogeneity"]["min"],
                                     spec["field_inhomogeneity"]["max"])
            motion = rng_tier.uniform(
                spec["motion_artifact_amplitude"]["min"],
                spec["motion_artifact_amplitude"]["max"])

            fwd_seed = base_seed + si * 311 + 42

            # --- Forward model ---
            result = forward_model(x_true, accel, noise_sigma, inhom,
                                   motion, fwd_seed)

            # --- Reconstruction ---
            recon = reconstruct_zero_filled(result["y"], result["H_ideal"])

            # Normalise for metrics
            gt_f = x_true.astype(np.float32)
            gt_max = gt_f.max()
            if gt_max > 1e-8:
                gt_norm = gt_f / gt_max
                rec_norm = np.clip(recon / gt_max, 0, None)
            else:
                gt_norm = gt_f
                rec_norm = recon

            psnr = compute_psnr(gt_norm, rec_norm)
            ssim = compute_ssim(gt_norm, rec_norm)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            # --- Write HDF5 ---
            grp.create_dataset("x_true", data=gt_f, compression="gzip")
            grp.create_dataset("y", data=result["y"], compression="gzip")
            grp.create_dataset("H_ideal", data=result["H_ideal"],
                               compression="gzip")

            sample_spec = {
                "acceleration_factor": float(accel),
                "noise_sigma": float(noise_sigma),
                "field_inhomogeneity": float(inhom),
                "motion_artifact_amplitude": float(motion),
            }
            true_specs[sample_name] = sample_spec
            grp.attrs["true_spec"] = json.dumps(sample_spec)
            grp.attrs["metadata"] = json.dumps({
                "shape": [IMG_SIZE, IMG_SIZE],
                "phantom_seed": phantom_seed,
                "forward_seed": fwd_seed,
                "baseline_psnr_dB": round(psnr, 2),
                "baseline_ssim": round(ssim, 4),
                "n_activation_blobs": int((activation > 0.001).sum() > 0),
            })

            # --- Per-sample images ---
            sample_img_dir = img_dir / sample_name
            sample_img_dir.mkdir(parents=True, exist_ok=True)

            save_image(gt_f, str(sample_img_dir / "gt.png"))
            save_kspace_image(result["y"], str(sample_img_dir / "measurement.png"))
            save_image(recon, str(sample_img_dir / "recon.png"))
            save_mask_image(result["H_ideal"],
                            str(sample_img_dir / "mask.png"))

            # Activation overlay
            act_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            gt_uint8 = _to_uint8(gt_f)
            act_vis[:, :, 0] = gt_uint8
            act_vis[:, :, 1] = gt_uint8
            act_vis[:, :, 2] = gt_uint8
            # Overlay activation in red
            act_norm = activation / (activation.max() + 1e-12)
            hot_mask = act_norm > 0.1
            act_vis[hot_mask, 0] = np.clip(
                act_vis[hot_mask, 0].astype(np.float32) +
                act_norm[hot_mask] * 200, 0, 255).astype(np.uint8)
            act_vis[hot_mask, 1] = (act_vis[hot_mask, 1] * 0.5).astype(np.uint8)
            act_vis[hot_mask, 2] = (act_vis[hot_mask, 2] * 0.3).astype(np.uint8)
            Image.fromarray(act_vis).save(str(sample_img_dir / "activation.png"))

            print(f"    {sample_name}: accel={accel:.1f}x  noise={noise_sigma:.4f}  "
                  f"B0={inhom:.1f}Hz  motion={motion:.2f}px  "
                  f"PSNR={psnr:.2f} dB  SSIM={ssim:.3f}")

    # --- Tier-level spec files ---
    with open(tier_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    with open(tier_dir / "true_spec.json", "w") as f:
        json.dump(true_specs, f, indent=2)

    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)
    print(f"  {tier_name} tier: {n_samples} samples  "
          f"mean PSNR={mean_psnr:.2f} dB  mean SSIM={mean_ssim:.3f}")
    print(f"  HDF5 -> {h5_path}")

    return h5_path, mean_psnr, mean_ssim


# ============================================================================
# Gallery image generation
# ============================================================================

def generate_gallery_images(n_scenes: int = 4):
    """Generate gallery images for the platform (scene_00 .. scene_03).

    Each scene: gt.png, measurement_I.png, measurement_II.png,
                recon_I.png, recon_II.png
    """
    gallery_root = (Path(__file__).resolve().parents[3] /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "fmri")

    print(f"\nGenerating {n_scenes} gallery scenes -> {gallery_root}")

    for scene_idx in range(n_scenes):
        scene_dir = gallery_root / f"scene_{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        seed = 99000 + scene_idx * 777
        rng = np.random.RandomState(seed)

        # Generate phantom
        x_baseline = generate_brain_phantom(rng, IMG_SIZE, IMG_SIZE)
        brain_mask = (x_baseline > 0.05).astype(np.float64)
        activation = generate_bold_activation(rng, brain_mask,
                                              IMG_SIZE, IMG_SIZE)
        x_true = np.clip(x_baseline * (1.0 + activation), 0, 1)

        # gt.png
        save_image(x_true.astype(np.float32), str(scene_dir / "gt.png"))

        # measurement_I: undersampled k-space (low accel)
        res1 = forward_model(x_true, acceleration=3.0, noise_sigma=0.010,
                             inhomogeneity_hz=5.0, motion_px=0.5,
                             seed=seed + 1)
        save_kspace_image(res1["y"], str(scene_dir / "measurement_I.png"))

        # measurement_II: undersampled k-space (high accel)
        res2 = forward_model(x_true, acceleration=5.0, noise_sigma=0.025,
                             inhomogeneity_hz=20.0, motion_px=1.5,
                             seed=seed + 2)
        save_kspace_image(res2["y"], str(scene_dir / "measurement_II.png"))

        # recon_I: zero-filled from low accel
        rec1 = reconstruct_zero_filled(res1["y"], res1["H_ideal"])
        save_image(rec1, str(scene_dir / "recon_I.png"))

        # recon_II: zero-filled from high accel
        rec2 = reconstruct_zero_filled(res2["y"], res2["H_ideal"])
        save_image(rec2, str(scene_dir / "recon_II.png"))

        # recon_III: activation overlay on gt (showing BOLD contrast)
        act_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        gt8 = _to_uint8(x_true.astype(np.float32))
        act_vis[:, :, 0] = gt8
        act_vis[:, :, 1] = gt8
        act_vis[:, :, 2] = gt8
        act_n = activation / (activation.max() + 1e-12)
        hot = act_n > 0.1
        act_vis[hot, 0] = np.clip(
            act_vis[hot, 0].astype(np.float32) + act_n[hot] * 200,
            0, 255).astype(np.uint8)
        act_vis[hot, 1] = (act_vis[hot, 1] * 0.5).astype(np.uint8)
        act_vis[hot, 2] = (act_vis[hot, 2] * 0.3).astype(np.uint8)
        Image.fromarray(act_vis).save(str(scene_dir / "recon_III.png"))

        print(f"    scene_{scene_idx:02d}: written {scene_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("fMRI Benchmark Dataset Generator")
    print("  Forward: y = U_Omega * F * (x_baseline + BOLD) + noise")
    print("  Baseline recon: zero-filled IFFT")
    print("=" * 65)

    results = {}
    for tier_name in ["public", "dev", "hidden"]:
        cfg = TIER_CONFIGS[tier_name]
        spec = TIER_SPECS[tier_name]
        print(f"\n--- {tier_name.upper()} Tier ({cfg['n_samples']} samples) ---")
        h5_path, mean_psnr, mean_ssim = generate_tier(
            tier_name, spec, cfg["n_samples"], cfg["seed_offset"])
        results[tier_name] = {
            "h5_path": str(h5_path),
            "mean_psnr": round(mean_psnr, 2),
            "mean_ssim": round(mean_ssim, 4),
        }

    # Gallery images
    generate_gallery_images(n_scenes=4)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for tier, r in results.items():
        print(f"  {tier:8s}: PSNR={r['mean_psnr']:6.2f} dB  "
              f"SSIM={r['mean_ssim']:.4f}  -> {r['h5_path']}")

    # Write summary JSON
    with open(OUT_DIR / "generation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutput directory: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
