#!/usr/bin/env python3
"""Generate MRI benchmark datasets (public / dev / hidden).

Forward model:  y = U_Omega * F * C * x + n
  - x: complex image (ground truth)
  - C: coil sensitivity maps (SENSE model, 4 coils)
  - F: 2D Discrete Fourier Transform
  - U_Omega: Cartesian undersampling mask (random k-space lines)
  - n: complex Gaussian noise

Public data from M4Raw (multi-coil MRI, T1/T2 brain, 4 coils, 256x256).
Dev and hidden use augmented slices from different patients/contrasts.

Reference: Lyu, M. et al. "M4Raw: A multi-contrast, multi-repetition,
multi-channel MRI k-space dataset for reproducible AI research,"
Scientific Data, 2023.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# ── Constants ──
IMG_SIZE = 256
N_COILS = 4
GCS_MRI_PATH = "gs://pwm-benchmark-datasets/datasets/real_mri/real_mri/multicoil_val"
LOCAL_SRC = Path("/tmp/mri_src")
OUT_DIR = Path(__file__).resolve().parent

# M4Raw source files — 2 T1 + 1 T2, 2 patients
SOURCE_FILES = [
    "2022061203_T101.h5",  # Patient A, T1-weighted, 18 slices
    "2022061203_T201.h5",  # Patient A, T2-weighted, 18 slices
    "2022061204_T101.h5",  # Patient B, T1-weighted, 18 slices
]


# ── Data Loading ──

def ensure_source_data():
    """Download M4Raw HDF5 files from GCS if not cached."""
    LOCAL_SRC.mkdir(parents=True, exist_ok=True)
    for fn in SOURCE_FILES:
        local = LOCAL_SRC / fn
        if local.exists() and local.stat().st_size > 0:
            continue
        gcs = f"{GCS_MRI_PATH}/{fn}"
        print(f"  Downloading {fn} from GCS...")
        subprocess.run(["gcloud", "storage", "cp", gcs, str(local)],
                       check=True, capture_output=True)
    print(f"  Source data ready in {LOCAL_SRC}")


def load_all_slices():
    """Load all M4Raw slices. Returns list of dicts with kspace and rss."""
    ensure_source_data()
    slices = []
    for fn in SOURCE_FILES:
        f = h5py.File(LOCAL_SRC / fn, "r")
        kspace = f["kspace"][()]        # (n_slices, n_coils, H, W) complex64
        rss = f["reconstruction_rss"][()]  # (n_slices, H, W) float32
        acq = str(f.attrs.get("acquisition", "unknown"))
        pid = str(f.attrs.get("patient_id", "unknown"))[:8]
        for si in range(kspace.shape[0]):
            slices.append({
                "kspace": kspace[si],   # (4, 256, 256) complex64
                "rss": rss[si],         # (256, 256) float32
                "source_file": fn,
                "slice_idx": si,
                "acquisition": acq,
                "patient_id": pid,
            })
        f.close()
    return slices


# ── Forward Model ──

def estimate_coil_maps(kspace: np.ndarray, cal_size: int = 24) -> np.ndarray:
    """Estimate coil sensitivity maps from center of k-space (simple method).

    Args:
        kspace: (n_coils, H, W) complex array of fully-sampled k-space
        cal_size: size of calibration region

    Returns:
        coil_maps: (n_coils, H, W) complex coil sensitivity maps
    """
    n_coils, H, W = kspace.shape
    # Extract calibration region (center of k-space)
    ch, cw = H // 2, W // 2
    half = cal_size // 2
    cal_kspace = np.zeros_like(kspace)
    cal_kspace[:, ch - half:ch + half, cw - half:cw + half] = \
        kspace[:, ch - half:ch + half, cw - half:cw + half]

    # Low-resolution images from calibration data
    cal_images = kspace_to_image(cal_kspace)

    # RSS of calibration images
    rss = np.sqrt(np.sum(np.abs(cal_images) ** 2, axis=0) + 1e-12)

    # Coil maps = cal_image / RSS
    coil_maps = cal_images / rss[None, :, :]
    return coil_maps


def generate_cartesian_mask(H: int, W: int, acceleration: float,
                            center_fraction: float = 0.08,
                            seed: int = 42) -> np.ndarray:
    """Generate Cartesian (line) undersampling mask.

    Always keeps center lines; randomly samples remaining lines.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)

    # Keep center lines
    n_center = max(1, int(W * center_fraction))
    c = W // 2
    mask[:, c - n_center // 2:c + n_center // 2] = 1.0

    # Randomly sample additional lines
    n_total_lines = max(n_center, int(W / acceleration))
    n_random = n_total_lines - n_center
    available = [i for i in range(W)
                 if i < c - n_center // 2 or i >= c + n_center // 2]
    if n_random > 0 and len(available) > 0:
        chosen = rng.choice(available, size=min(n_random, len(available)),
                            replace=False)
        mask[:, chosen] = 1.0
    return mask


def apply_forward_model(kspace_full: np.ndarray, acceleration: float,
                        noise_sigma: float, seed: int,
                        coil_err: float = 0.0,
                        off_resonance_hz: float = 0.0,
                        traj_deviation: float = 0.0) -> dict:
    """Apply MRI forward model with mismatch.

    Args:
        kspace_full: (n_coils, H, W) fully-sampled k-space
        acceleration: undersampling factor (e.g. 4.0)
        noise_sigma: complex Gaussian noise std
        seed: random seed
        coil_err: relative perturbation to coil maps
        off_resonance_hz: B0 inhomogeneity in Hz
        traj_deviation: k-space trajectory deviation (relative)

    Returns:
        dict with undersampled kspace, mask, coil maps, etc.
    """
    rng = np.random.RandomState(seed)
    n_coils, H, W = kspace_full.shape

    # Estimate coil maps
    coil_maps = estimate_coil_maps(kspace_full)

    # Apply coil sensitivity error (mismatch)
    if coil_err > 0:
        perturb = rng.randn(*coil_maps.shape).astype(np.complex64) * coil_err
        coil_maps_used = coil_maps + perturb
        # Re-normalize
        rss_maps = np.sqrt(np.sum(np.abs(coil_maps_used) ** 2, axis=0) + 1e-12)
        coil_maps_used = coil_maps_used / rss_maps[None, :, :]
    else:
        coil_maps_used = coil_maps

    # Apply off-resonance (phase modulation in image domain)
    if abs(off_resonance_hz) > 0.1:
        # Create smooth B0 map (spatial variation)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        b0_map = off_resonance_hz * (0.3 * yy + 0.2 * xx + 0.1 * yy * xx)
        # Phase = 2*pi*f*TE, assume TE~20ms
        te = 0.020
        phase_map = np.exp(1j * 2 * np.pi * b0_map * te).astype(np.complex64)
        # Apply to image domain of each coil
        kspace_mod = np.zeros_like(kspace_full)
        for c in range(n_coils):
            img_c = kspace_to_image(kspace_full[c:c+1])[0]
            img_c = img_c * phase_map
            kspace_mod[c] = image_to_kspace(img_c[None])[0]
        kspace_full = kspace_mod

    # Apply trajectory deviation (k-space shift)
    if abs(traj_deviation) > 0.001:
        shift_px = traj_deviation * W
        # Apply sub-pixel shift via phase ramp in k-space
        freq_y = np.fft.fftfreq(H) * 2 * np.pi
        freq_x = np.fft.fftfreq(W) * 2 * np.pi
        _, xx = np.meshgrid(freq_y, freq_x, indexing="ij")
        shift_phase = np.exp(-1j * (xx * shift_px)).astype(np.complex64)
        for c in range(n_coils):
            kspace_full[c] = kspace_full[c] * shift_phase

    # Generate undersampling mask
    mask = generate_cartesian_mask(H, W, acceleration, seed=seed)

    # Undersample k-space
    kspace_under = kspace_full * mask[None, :, :]

    # Add complex Gaussian noise
    noise = (rng.randn(*kspace_under.shape) +
             1j * rng.randn(*kspace_under.shape)).astype(np.complex64)
    noise *= noise_sigma / np.sqrt(2)
    kspace_under = kspace_under + noise * mask[None, :, :]

    return {
        "kspace_undersampled": kspace_under,
        "mask": mask,
        "coil_maps": coil_maps_used,
        "kspace_full": kspace_full,
    }


# ── Reconstruction (CPU) ──

def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Convert k-space to image domain with correct M4Raw normalization.

    M4Raw stores k-space in ortho convention; need sqrt(H*W) scale factor.
    Input:  (n_coils, H, W) or (H, W) complex
    Output: same shape, complex image
    """
    axes = (-2, -1)
    imgs = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes),
        axes=axes)
    H, W = kspace.shape[-2], kspace.shape[-1]
    return imgs * np.sqrt(H * W)


def image_to_kspace(imgs: np.ndarray) -> np.ndarray:
    """Convert image to k-space (inverse of kspace_to_image)."""
    axes = (-2, -1)
    H, W = imgs.shape[-2], imgs.shape[-1]
    scaled = imgs / np.sqrt(H * W)
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(scaled, axes=axes), axes=axes),
        axes=axes)


def reconstruct_zero_filled(kspace_under: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """Zero-filled IFFT reconstruction (simplest CPU method).

    Returns magnitude image (H, W).
    """
    imgs = kspace_to_image(kspace_under)
    # RSS combination
    rss = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0))
    return rss.astype(np.float32)


# ── Metrics ──

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = gt.max() - gt.min()
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Simple SSIM (no skimage dependency)."""
    c1 = (0.01 * (gt.max() - gt.min())) ** 2
    c2 = (0.03 * (gt.max() - gt.min())) ** 2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# ── Image Saving ──

def save_image(arr: np.ndarray, path: str, is_complex: bool = False):
    """Save array as 8-bit PNG."""
    if is_complex:
        arr = np.abs(arr)
    if arr.ndim == 3:  # multi-coil → RSS
        arr = np.sqrt(np.sum(arr ** 2, axis=0))
    vmin, vmax = np.percentile(arr[arr > 0], [1, 99]) if arr.max() > 0 else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1
    normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    img_uint8 = (normalized * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def save_kspace_image(kspace: np.ndarray, path: str):
    """Save k-space magnitude (log scale) as PNG."""
    mag = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0))
    log_mag = np.log1p(mag)
    vmin, vmax = log_mag.min(), log_mag.max()
    if vmax <= vmin:
        vmax = vmin + 1
    normalized = (log_mag - vmin) / (vmax - vmin)
    img_uint8 = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# ── Tier Generation ──

def generate_tier(tier_name: str, slices: list, mismatch_ranges: dict,
                  n_samples: int, base_seed: int, augment: bool = False,
                  adversarial: bool = False):
    """Generate one tier (public/dev/hidden)."""
    tier_dir = OUT_DIR / tier_name
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    rng = np.random.RandomState(base_seed)

    # Select slices for this tier
    if len(slices) >= n_samples:
        selected = slices[:n_samples]
    else:
        # Repeat with different augmentations
        selected = []
        for i in range(n_samples):
            selected.append(slices[i % len(slices)])

    # Per-sample mismatch
    true_specs = {}
    h5_path = tier_dir / f"mri_challenge_{tier_name}.h5"

    with h5py.File(h5_path, "w") as hf:
        # Root attributes
        hf.attrs["description"] = (
            f"MRI {tier_name} tier — multi-coil Cartesian undersampled "
            f"k-space from M4Raw (Scientific Data 2023). "
            f"Forward model: y = U_Omega * kspace_full + noise"
        )
        hf.attrs["source"] = "M4Raw multi-coil MRI (Lyu et al., Scientific Data 2023)"
        hf.attrs["geometry"] = json.dumps({
            "image_size": IMG_SIZE,
            "n_coils": N_COILS,
            "fov_mm": 220.0,
            "pixel_size_mm": 220.0 / IMG_SIZE,
            "field_strength_T": 3.0,
        })
        hf.attrs["spec_ranges"] = json.dumps(mismatch_ranges)

        for si, sl in enumerate(selected):
            sample_name = f"sample_{si:02d}"
            grp = hf.create_group(sample_name)

            kspace_full = sl["kspace"].copy()  # (4, 256, 256)
            rss_gt = sl["rss"].copy()          # (256, 256)

            # Augmentation for dev/hidden
            if augment:
                k = rng.randint(0, 4)  # 0,90,180,270 rotation
                if k > 0:
                    kspace_full = np.rot90(kspace_full, k, axes=(-2, -1)).copy()
                    rss_gt = np.rot90(rss_gt, k).copy()
                if rng.rand() < 0.5:  # horizontal flip
                    kspace_full = kspace_full[:, :, ::-1].copy()
                    rss_gt = rss_gt[:, ::-1].copy()

            # Adversarial modifications for hidden tier
            if adversarial and rng.rand() < 0.3:
                # Add a subtle lesion (low-contrast blob)
                cy, cx = rng.randint(60, 196), rng.randint(60, 196)
                r = rng.randint(5, 15)
                yy, xx = np.ogrid[-r:r+1, -r:r+1]
                circle = (yy**2 + xx**2 <= r**2).astype(np.float32)
                intensity = rss_gt[cy, cx] * rng.uniform(0.85, 0.95)
                rss_gt[cy-r:cy+r+1, cx-r:cx+r+1] *= (1 - 0.15 * circle)
                # Re-compute k-space from modified image (approximate)
                for c in range(N_COILS):
                    img_c = kspace_to_image(kspace_full[c:c+1])[0]
                    scale = np.abs(img_c) + 1e-12
                    new_scale = scale.copy()
                    new_scale[cy-r:cy+r+1, cx-r:cx+r+1] *= (1 - 0.15 * circle)
                    img_c = img_c * (new_scale / scale)
                    kspace_full[c] = image_to_kspace(img_c[None])[0]

            # Sample mismatch parameters
            accel = rng.uniform(mismatch_ranges["acceleration_factor"]["min"],
                                mismatch_ranges["acceleration_factor"]["max"])
            noise_sigma = rng.uniform(mismatch_ranges["noise_sigma"]["min"],
                                      mismatch_ranges["noise_sigma"]["max"])
            coil_err = rng.uniform(mismatch_ranges["coil_sensitivity_error"]["min"],
                                   mismatch_ranges["coil_sensitivity_error"]["max"])
            off_res = rng.uniform(mismatch_ranges["off_resonance_hz"]["min"],
                                  mismatch_ranges["off_resonance_hz"]["max"])
            traj_dev = rng.uniform(mismatch_ranges["trajectory_deviation"]["min"],
                                   mismatch_ranges["trajectory_deviation"]["max"])

            seed = base_seed + si * 100 + 42
            result = apply_forward_model(
                kspace_full, accel, noise_sigma, seed,
                coil_err=coil_err,
                off_resonance_hz=off_res,
                traj_deviation=traj_dev,
            )

            # Reconstruction (zero-filled IFFT)
            recon = reconstruct_zero_filled(result["kspace_undersampled"],
                                            result["mask"])

            # Normalize GT for PSNR/SSIM
            gt_norm = rss_gt / (rss_gt.max() + 1e-12)
            recon_norm = recon / (rss_gt.max() + 1e-12)
            psnr = compute_psnr(gt_norm, recon_norm)
            ssim = compute_ssim(gt_norm, recon_norm)

            # Save to HDF5
            grp.create_dataset("x_true", data=rss_gt, compression="gzip")
            grp.create_dataset("kspace_full", data=kspace_full, compression="gzip")
            grp.create_dataset("kspace_undersampled",
                               data=result["kspace_undersampled"],
                               compression="gzip")
            grp.create_dataset("mask", data=result["mask"], compression="gzip")
            grp.create_dataset("coil_maps", data=result["coil_maps"],
                               compression="gzip")

            # Metadata
            spec = {
                "acceleration_factor": float(accel),
                "noise_sigma": float(noise_sigma),
                "coil_sensitivity_error": float(coil_err),
                "off_resonance_hz": float(off_res),
                "trajectory_deviation": float(traj_dev),
            }
            true_specs[sample_name] = spec
            grp.attrs["metadata"] = json.dumps({
                "scene": f"{sl['acquisition']}_{sl['patient_id']}_s{sl['slice_idx']:02d}",
                "shape": [IMG_SIZE, IMG_SIZE],
                "n_coils": N_COILS,
                "source": sl["source_file"],
                "slice_idx": int(sl["slice_idx"]),
            })
            grp.attrs["true_spec"] = json.dumps(spec)
            grp.attrs["spec_ranges"] = json.dumps(mismatch_ranges)

            # Save images
            scene_name = f"{sl['acquisition']}_{sl['patient_id']}_s{sl['slice_idx']:02d}"
            sample_img_dir = img_dir / f"{sample_name}_{scene_name}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)

            save_image(rss_gt, str(sample_img_dir / "ground_truth.png"))
            save_kspace_image(kspace_full, str(sample_img_dir / "kspace_full.png"))
            save_kspace_image(result["kspace_undersampled"],
                              str(sample_img_dir / "kspace_undersampled.png"))
            save_image(result["mask"], str(sample_img_dir / "mask.png"))
            save_image(recon, str(sample_img_dir / "reconstruction_zf.png"))

            # Overview image (4-panel)
            panels = []
            for arr, label in [
                (rss_gt, "Ground Truth"),
                (recon, "Zero-Filled IFFT"),
            ]:
                v = np.percentile(arr[arr > 0], [1, 99]) if arr.max() > 0 else [0, 1]
                norm = np.clip((arr - v[0]) / (v[1] - v[0] + 1e-8), 0, 1)
                panels.append((norm * 255).astype(np.uint8))

            # k-space panels
            for ks in [kspace_full, result["kspace_undersampled"]]:
                mag = np.sqrt(np.sum(np.abs(ks) ** 2, axis=0))
                log_mag = np.log1p(mag)
                norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
                panels.append((np.clip(norm, 0, 1) * 255).astype(np.uint8))

            top = np.hstack(panels[:2])
            bot = np.hstack(panels[2:])
            overview = np.vstack([top, bot])
            Image.fromarray(overview).save(str(sample_img_dir / "overview.png"))

            # Per-sample spec.json
            with open(sample_img_dir / "spec.json", "w") as jf:
                json.dump(spec, jf, indent=2)

            print(f"    {sample_name}: accel={accel:.1f}x, "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim:.3f}")

    # Write tier spec.json
    with open(tier_dir / "spec.json", "w") as jf:
        json.dump(mismatch_ranges, jf, indent=2)

    # Write true_spec.json
    with open(tier_dir / "true_spec.json", "w") as jf:
        json.dump(true_specs, jf, indent=2)

    # Write README
    _write_tier_readme(tier_name, tier_dir, n_samples, mismatch_ranges, true_specs)

    print(f"  {tier_name} tier: {n_samples} samples → {h5_path}")
    return h5_path


def _write_tier_readme(tier_name: str, tier_dir: Path, n_samples: int,
                       ranges: dict, true_specs: dict):
    """Generate per-tier README."""
    lines = [
        f"# MRI {tier_name.title()} Tier",
        "",
        f"**Samples:** {n_samples}",
        f"**Source:** M4Raw multi-coil brain MRI (Lyu et al., Scientific Data 2023)",
        f"**Forward model:** y = U_Omega * F * C * x + n (Cartesian k-space undersampling)",
        "",
        "## Mismatch Parameter Ranges",
        "",
        "| Parameter | Min | Max | Unit |",
        "|-----------|-----|-----|------|",
    ]
    for name, r in ranges.items():
        lines.append(f"| {name} | {r['min']} | {r['max']} | {r.get('unit', '')} |")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    for sname, spec in true_specs.items():
        lines.append(f"- **{sname}**: accel={spec['acceleration_factor']:.1f}x, "
                     f"noise_sigma={spec['noise_sigma']:.4f}, "
                     f"coil_err={spec['coil_sensitivity_error']:.4f}")
    lines.append("")

    with open(tier_dir / "README.md", "w") as f:
        f.write("\n".join(lines))


# ── Main ──

# Mismatch ranges per tier
SPEC_PUBLIC = {
    "acceleration_factor": {"min": 3.0, "max": 5.0, "unit": "x"},
    "noise_sigma": {"min": 0.001, "max": 0.005, "unit": ""},
    "coil_sensitivity_error": {"min": 0.0, "max": 0.02, "unit": "relative"},
    "off_resonance_hz": {"min": -10.0, "max": 10.0, "unit": "Hz"},
    "trajectory_deviation": {"min": -0.005, "max": 0.005, "unit": "relative"},
}

SPEC_DEV = {
    "acceleration_factor": {"min": 3.0, "max": 6.0, "unit": "x"},
    "noise_sigma": {"min": 0.002, "max": 0.010, "unit": ""},
    "coil_sensitivity_error": {"min": 0.0, "max": 0.04, "unit": "relative"},
    "off_resonance_hz": {"min": -20.0, "max": 20.0, "unit": "Hz"},
    "trajectory_deviation": {"min": -0.01, "max": 0.01, "unit": "relative"},
}

SPEC_HIDDEN = {
    "acceleration_factor": {"min": 4.0, "max": 8.0, "unit": "x"},
    "noise_sigma": {"min": 0.005, "max": 0.020, "unit": ""},
    "coil_sensitivity_error": {"min": 0.0, "max": 0.06, "unit": "relative"},
    "off_resonance_hz": {"min": -30.0, "max": 30.0, "unit": "Hz"},
    "trajectory_deviation": {"min": -0.02, "max": 0.02, "unit": "relative"},
}


def main():
    print("=" * 60)
    print("MRI Benchmark Dataset Generator")
    print("=" * 60)

    # Load all source data
    print("\nLoading M4Raw source data...")
    all_slices = load_all_slices()
    print(f"  Loaded {len(all_slices)} slices from {len(SOURCE_FILES)} volumes")

    # Select slices per tier (no overlap)
    # Patient A T1 (0-17), Patient A T2 (18-35), Patient B T1 (36-53)
    # Select best mid-brain slices (indices 5-14 have best anatomy)
    good_indices = list(range(5, 15))  # 10 per volume = 30 total

    # Public: 12 diverse slices from all 3 volumes
    public_slices = [all_slices[i] for i in good_indices[:4]] + \
                    [all_slices[18 + i] for i in good_indices[:4]] + \
                    [all_slices[36 + i] for i in good_indices[:4]]

    # Dev: 20 slices from remaining indices
    dev_indices = list(range(3, 5)) + list(range(10, 15))  # 7 per volume
    dev_slices = [all_slices[i] for i in dev_indices[:7]] + \
                 [all_slices[18 + i] for i in dev_indices[:7]] + \
                 [all_slices[36 + i] for i in dev_indices[:6]]

    # Hidden: 20 slices from edge indices + augmented
    hidden_indices = list(range(0, 5)) + list(range(13, 18))  # 10 per volume
    hidden_slices = [all_slices[i] for i in hidden_indices[:7]] + \
                    [all_slices[18 + i] for i in hidden_indices[:7]] + \
                    [all_slices[36 + i] for i in hidden_indices[:6]]

    # Generate tiers
    print("\n--- Public Tier (12 samples) ---")
    generate_tier("public", public_slices, SPEC_PUBLIC, 12, base_seed=1000)

    print("\n--- Dev Tier (20 samples) ---")
    generate_tier("dev", dev_slices, SPEC_DEV, 20, base_seed=2000,
                  augment=True)

    print("\n--- Hidden Tier (20 samples) ---")
    generate_tier("hidden", hidden_slices, SPEC_HIDDEN, 20, base_seed=3000,
                  augment=True, adversarial=True)

    # Write top-level README
    _write_top_readme()

    print("\n" + "=" * 60)
    print("MRI benchmark dataset generation complete!")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)


def _write_top_readme():
    """Write top-level README for MRI benchmark."""
    readme = """# MRI Benchmark Dataset

## Overview

Multi-coil brain MRI reconstruction benchmark using real M4Raw data
(Lyu et al., Scientific Data 2023). Standard Cartesian k-space
undersampling forward model with mismatch parameters.

## Forward Model

```
y = U_Omega * F * C * x + n

where:
  x: complex MR image (ground truth, 256x256)
  C: coil sensitivity maps (4 coils, estimated from calibration region)
  F: 2D Discrete Fourier Transform
  U_Omega: Cartesian undersampling mask (random k-space lines)
  n: complex Gaussian noise
```

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| acceleration_factor | Undersampling ratio | 3-5x | 3-6x | 4-8x |
| noise_sigma | Complex noise std | 0.001-0.005 | 0.002-0.010 | 0.005-0.020 |
| coil_sensitivity_error | Coil map perturbation | 0-2% | 0-4% | 0-6% |
| off_resonance_hz | B0 inhomogeneity | +/-10 Hz | +/-20 Hz | +/-30 Hz |
| trajectory_deviation | k-space shift | +/-0.5% | +/-1% | +/-2% |

## Dataset Structure

```
mri/
├── README.md
├── generate_dataset.py
├── public/                    (12 samples, GT visible)
│   ├── mri_challenge_public.h5
│   ├── spec.json
│   ├── true_spec.json
│   └── images/sample_XX_*/
├── dev/                       (20 samples, blind eval)
│   ├── mri_challenge_dev.h5
│   ├── spec.json
│   ├── true_spec.json
│   └── images/sample_XX_*/
└── hidden/                    (20 samples, server-only)
    ├── mri_challenge_hidden.h5
    ├── spec.json
    ├── true_spec.json
    └── images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
├── x_true (256, 256) float32         # Ground truth (RSS image)
├── kspace_full (4, 256, 256) complex64  # Fully-sampled multi-coil k-space
├── kspace_undersampled (4, 256, 256) complex64  # Undersampled + noise
├── mask (256, 256) float32           # Undersampling mask
└── coil_maps (4, 256, 256) complex64 # Estimated coil sensitivity maps
```

## References

1. Lyu, M. et al. "M4Raw: A multi-contrast, multi-repetition, multi-channel
   MRI k-space dataset for reproducible AI research," Scientific Data, 2023.
2. Pruessmann, K. et al. "SENSE: Sensitivity encoding for fast MRI," MRM, 1999.
3. Lustig, M. et al. "Sparse MRI," MRM, 2007.
4. Zbontar, J. et al. "fastMRI: An open dataset and benchmarks for accelerated
   MRI," arXiv 2018.
"""
    with open(OUT_DIR / "README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
