"""Build multi-coil MRI benchmark dataset (PWM-style).

Forward model (parallel imaging with 4-knob mismatch):
    y_c = mask · k_traj_error( F( S_c_true · warp(x · exp(i·2π·B0·TE·b0)) ) ) + n_c

where:
  S_c_true = S_c_nominal · (1 + ε_c)          [coil sensitivity mismatch]
  warp      via smooth displacement field       [gradient nonlinearity mismatch]
  b0_map    smooth B0 inhomogeneity field       [field inhomogeneity mismatch]
  k-ramp    per-line phase ramp                 [k-trajectory mismatch]

Nominal operator (what algorithms see):
  H_nominal:  y_c = mask · F( S_c_nominal · x )

Dataset parameters (matches fastMRI multi-coil knee):
  Shape   = 320 × 320
  Coils   = 15
  Accel   = 4×  (variable-density Cartesian, center fraction 0.08)
  TE      = 25 ms

Public tier — real fastMRI multi-coil knee k-space
--------------------------------------------------
Set the environment variable FASTMRI_ROOT to the directory containing
fastMRI multi-coil knee H5 files (the *_multicoil_train or *_multicoil_val
subdirectories from the official download).

  export FASTMRI_ROOT=/path/to/knee_multicoil_train

Files are named fileNNNNNN.h5 and contain:
  kspace : (n_slices, n_coils, kH, kW)  complex64

Download instructions:
  1. Register at https://fastmri.med.nyu.edu/
  2. Download "Knee MRI multi-coil raw data" (multicoil_train + multicoil_val)
  3. Set FASTMRI_ROOT to the directory containing the .h5 files

If FASTMRI_ROOT is not set or contains no files, the public tier is built
from synthetic Shepp-Logan phantoms as a placeholder (clearly labelled).

Run from the mri/ directory:
    python build_dataset.py

Creates:
  public/   mri_challenge_public.h5  + images/  (11 samples, mild mismatch)
  dev/      mri_challenge_dev.h5     + images/  (20 knee-like, mild mismatch)
  hidden/   mri_challenge_hidden.h5  + images/  (20 adversarial, severe mismatch)
"""

from __future__ import annotations

import json
import os
import sys
import glob

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_scenes import generate_mri_gt


# ── Constants ──────────────────────────────────────────────────────────────────

SHAPE    = (320, 320)   # matches fastMRI knee multi-coil target
N_COILS  = 15           # matches fastMRI knee coil count
ACCEL    = 4
ACS_FRAC = 0.08         # 8% auto-calibration signal (ACS) lines
TE_S     = 0.025        # 25 ms echo time

SPEC_RANGES = {
    "public": {
        "B0_inhomog_hz":         {"min":  5.0,  "max": 15.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min": 0.001, "max": 0.003, "unit": "frac"},
        "coil_sensitivity_frac": {"min": 0.01,  "max": 0.03,  "unit": "frac"},
        "k_trajectory_frac":     {"min": 0.001, "max": 0.003, "unit": "frac"},
        "noise_sigma":           {"min": 0.01,  "max": 0.02,  "unit": "rel"},
    },
    "dev": {
        "B0_inhomog_hz":         {"min":  5.0,  "max": 20.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min": 0.001, "max": 0.005, "unit": "frac"},
        "coil_sensitivity_frac": {"min": 0.01,  "max": 0.05,  "unit": "frac"},
        "k_trajectory_frac":     {"min": 0.001, "max": 0.005, "unit": "frac"},
        "noise_sigma":           {"min": 0.01,  "max": 0.03,  "unit": "rel"},
    },
    "hidden": {
        "B0_inhomog_hz":         {"min": 20.0,  "max": 60.0,  "unit": "Hz"},
        "gradient_nonlin_frac":  {"min": 0.005, "max": 0.02,  "unit": "frac"},
        "coil_sensitivity_frac": {"min": 0.05,  "max": 0.15,  "unit": "frac"},
        "k_trajectory_frac":     {"min": 0.005, "max": 0.02,  "unit": "frac"},
        "noise_sigma":           {"min": 0.03,  "max": 0.06,  "unit": "rel"},
    },
}

# Deterministic subset of fastMRI volumes for public tier (11 samples).
# Sorted by filename; we pick one middle slice from each chosen volume.
PUBLIC_N_SAMPLES = 11


# ── fastMRI loader ─────────────────────────────────────────────────────────────

def _ifft2c(k):
    """Centred 2D IFFT."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def _fft2c(x):
    """Centred 2D FFT."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def load_fastmri_slice(h5_path: str, slice_idx: int | None = None,
                       target_shape: tuple = SHAPE) -> tuple:
    """Load one slice from a fastMRI multi-coil knee H5 file.

    Applies the standard fastMRI preprocessing pipeline:
      1. Centred IFFT of full k-space  → (n_coils, kH, kW) complex image
      2. Centre-crop readout dimension  → (n_coils, 320, W_crop)
      3. Centre-crop phase-encode       → (n_coils, 320, 320)
      4. RSS to get ground-truth image
      5. Re-FFT for clean benchmark k-space

    Parameters
    ----------
    h5_path     : path to fastMRI multi-coil .h5 file
    slice_idx   : which slice to load (None → middle slice)
    target_shape: (H, W) of output image

    Returns
    -------
    kspace_tgt : (n_coils, H, W)  complex64  — full k-space at target resolution
    coil_maps  : (n_coils, H, W)  complex64  — ACS-estimated sensitivity maps
    x_true     : (H, W)          float32     — RSS ground truth in [0, 1]
    n_coils    : int
    """
    H_tgt, W_tgt = target_shape

    with h5py.File(h5_path, "r") as hf:
        kspace_all = hf["kspace"][:]     # (n_slices, n_coils, kH, kW) complex64

    n_slices, n_coils_raw, kH, kW = kspace_all.shape
    if slice_idx is None:
        slice_idx = n_slices // 2
    kspace = kspace_all[slice_idx].astype(np.complex64)  # (C, kH, kW)

    # IFFT → image domain per coil
    imgs = _ifft2c(kspace)   # (C, kH, kW)

    # Centre-crop to target size (fastMRI knee: kH=640 → crop to 320 in readout)
    h_start = (kH - H_tgt) // 2
    w_start = (kW - W_tgt) // 2
    w_end   = w_start + W_tgt
    # If kW < W_tgt, pad symmetrically
    if kW < W_tgt:
        pad_w = (W_tgt - kW + 1) // 2
        imgs = np.pad(imgs, ((0,0), (0,0), (pad_w, pad_w)), mode="constant")
        w_start, w_end = 0, W_tgt
    imgs_crop = imgs[:, h_start:h_start + H_tgt, w_start:w_end]   # (C, H_tgt, W_tgt)

    # RSS ground truth
    rss = np.sqrt(np.sum(np.abs(imgs_crop) ** 2, axis=0)).astype(np.float32)
    scale = float(rss.max()) + 1e-8
    x_true = (rss / scale).astype(np.float32)

    # Re-FFT to get clean target-resolution k-space
    kspace_tgt = _fft2c(imgs_crop).astype(np.complex64)
    kspace_tgt /= scale   # normalise to match x_true

    # Estimate coil sensitivity maps from ACS (low-frequency k-space centre)
    coil_maps = _estimate_coil_maps_acs(kspace_tgt, acs_frac=ACS_FRAC)

    return kspace_tgt, coil_maps, x_true, n_coils_raw


def _estimate_coil_maps_acs(kspace: np.ndarray, acs_frac: float = 0.08) -> np.ndarray:
    """Smooth per-coil sensitivity maps estimated from the ACS region.

    Method: low-pass filter each coil image (keep only ACS k-space), then
    divide by RSS to get S_c(r) = coil_image_c(r) / RSS(r).

    Parameters
    ----------
    kspace   : (C, H, W) complex64  full k-space
    acs_frac : fraction of lines kept in both dims for low-pass filter

    Returns
    -------
    coil_maps : (C, H, W) complex64  normalised sensitivity maps
    """
    C, H, W = kspace.shape
    n_acs_h = max(8, int(H * acs_frac))
    n_acs_w = max(8, int(W * acs_frac))

    # Low-pass mask
    lp_mask = np.zeros((H, W), dtype=np.float32)
    h0, w0 = (H - n_acs_h) // 2, (W - n_acs_w) // 2
    lp_mask[h0:h0 + n_acs_h, w0:w0 + n_acs_w] = 1.0

    # Smooth coil images (low-pass IFFT)
    coil_imgs = _ifft2c(kspace * lp_mask[None])   # (C, H, W)

    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0)).astype(np.float32)
    rss = np.maximum(rss, 1e-8)

    coil_maps = (coil_imgs / rss[None]).astype(np.complex64)

    # Additional spatial smoothing per coil
    for c in range(C):
        re = gaussian_filter(coil_maps[c].real, sigma=3.0)
        im = gaussian_filter(coil_maps[c].imag, sigma=3.0)
        coil_maps[c] = (re + 1j * im).astype(np.complex64)

    return coil_maps


def find_fastmri_files(fastmri_root: str) -> list[str]:
    """Return sorted list of fastMRI multi-coil knee H5 files."""
    patterns = [
        os.path.join(fastmri_root, "*.h5"),
        os.path.join(fastmri_root, "**", "*.h5"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    # Filter to multi-coil knee files (filenames start with 'file')
    found = sorted(set(found))
    return found


def load_public_fastmri(n_samples: int = PUBLIC_N_SAMPLES,
                        target_shape: tuple = SHAPE) -> list[tuple]:
    """Load n_samples slices from fastMRI knee multi-coil data.

    Returns list of (scene_name, x_true, kspace_full, coil_maps, recipe_str).
    """
    fastmri_root = os.environ.get("FASTMRI_ROOT", "")
    files = find_fastmri_files(fastmri_root) if fastmri_root else []

    if not files:
        print("  [WARNING] FASTMRI_ROOT not set or no .h5 files found.")
        print("  [WARNING] Public tier will use SYNTHETIC PLACEHOLDER images.")
        print("  [WARNING] To use real fastMRI data:")
        print("  [WARNING]   export FASTMRI_ROOT=/path/to/knee_multicoil_train")
        print("  [WARNING] Download from: https://fastmri.med.nyu.edu/")
        return None   # signal to caller to use synthetic fallback

    # Pick deterministic subset: every Nth file
    step = max(1, len(files) // n_samples)
    chosen = files[::step][:n_samples]
    if len(chosen) < n_samples:
        chosen = (chosen * (n_samples // len(chosen) + 1))[:n_samples]

    scenes = []
    for i, fpath in enumerate(chosen):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        scene_name = f"fastmri_{fname}"
        print(f"  [public] {i:02d} Loading {fname} ...", end="", flush=True)
        try:
            kspace_full, coil_maps, x_true, n_coils_raw = load_fastmri_slice(
                fpath, target_shape=target_shape
            )
            print(f" coils={n_coils_raw} ok")
            scenes.append((scene_name, x_true, kspace_full, coil_maps, "fastmri_knee"))
        except Exception as exc:
            print(f" ERROR: {exc}  — skipping")
    return scenes if scenes else None


# ── Synthetic coil sensitivity maps (for dev/hidden) ───────────────────────────

def generate_coil_maps(shape, n_coils, rng, coil_radius_frac=0.58):
    """N coils on a ring: Gaussian magnitude + smooth spatially varying phase."""
    H, W = shape
    yy = np.linspace(-0.5, 0.5, H, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 0.5, W, dtype=np.float32)[None, :]
    coil_maps = np.zeros((n_coils, H, W), dtype=np.complex64)
    sigma = float(rng.uniform(0.20, 0.34))
    for c in range(n_coils):
        angle = 2.0 * np.pi * c / n_coils + float(rng.uniform(-0.08, 0.08))
        cy = coil_radius_frac * np.sin(angle)
        cx = coil_radius_frac * np.cos(angle)
        mag = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
        phase_n = gaussian_filter(
            rng.standard_normal((H, W)).astype(np.float32),
            sigma=float(rng.uniform(8.0, 22.0)),
        )
        phase_scale = float(rng.uniform(0.08, 0.35)) * np.pi
        phase = phase_n / (float(np.abs(phase_n).max()) + 1e-6) * phase_scale
        coil_maps[c] = (mag.astype(np.float32) * np.exp(1j * phase)).astype(np.complex64)
    return coil_maps


def generate_coil_perturbation(coil_maps, strength_frac, rng):
    """Smooth complex ε_c per coil; S_true = S_nominal * (1 + ε_c)."""
    n_coils, H, W = coil_maps.shape
    perturb = np.zeros((n_coils, H, W), dtype=np.complex64)
    for c in range(n_coils):
        re = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                             sigma=float(rng.uniform(5.0, 20.0)))
        im = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                             sigma=float(rng.uniform(5.0, 20.0)))
        p = (re + 1j * im).astype(np.complex64)
        perturb[c] = strength_frac * p / (float(np.abs(p).max()) + 1e-8)
    return perturb


# ── B0 field map ───────────────────────────────────────────────────────────────

def generate_b0_map(shape, rng):
    """Smooth field map normalised to [-1, 1]."""
    H, W = shape
    yy = np.linspace(-0.5, 0.5, H, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 0.5, W, dtype=np.float32)[None, :]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    grad = (np.cos(angle) * yy + np.sin(angle) * xx).astype(np.float32)
    noise = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                            sigma=float(rng.uniform(12.0, 40.0)))
    noise -= noise.mean()
    noise /= max(float(np.abs(noise).max()), 1e-6)
    b0 = 0.55 * grad + 0.45 * noise
    b0 /= max(float(np.abs(b0).max()), 1e-6)
    return b0.astype(np.float32)


# ── Gradient nonlinearity (geometric warp) ─────────────────────────────────────

def generate_warp_field(shape, strength_frac, rng):
    """Smooth displacement (dy, dx) in pixels; max ≈ strength_frac * min(H, W)."""
    H, W = shape
    max_d = strength_frac * min(H, W)
    fields = []
    for _ in range(2):
        f = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                            sigma=float(rng.uniform(25.0, 60.0)))
        f /= max(float(np.abs(f).max()), 1e-6)
        fields.append(f * max_d)
    return np.stack(fields, axis=0).astype(np.float32)  # (2, H, W)


def apply_warp(x, warp):
    """Warp image by displacement field; handles complex arrays."""
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = [np.clip(yy + warp[0], 0, H - 1),
              np.clip(xx + warp[1], 0, W - 1)]
    if np.iscomplexobj(x):
        re = map_coordinates(x.real.astype(np.float64), coords, order=1, mode="reflect")
        im = map_coordinates(x.imag.astype(np.float64), coords, order=1, mode="reflect")
        return (re + 1j * im).astype(np.complex64)
    return map_coordinates(x.astype(np.float64), coords, order=1, mode="reflect").astype(np.float32)


# ── k-trajectory error ─────────────────────────────────────────────────────────

def apply_k_trajectory_error(kspace_c, mask_1d, strength_frac, rng):
    """Per-line fractional k-space shift via phase ramp (gradient timing error)."""
    H, W = kspace_c.shape
    out = kspace_c.copy()
    sampled = np.where(mask_1d)[0]
    shifts = rng.uniform(-strength_frac, strength_frac, size=len(sampled))
    kx = np.arange(W, dtype=np.float32)
    for i, ky in enumerate(sampled):
        ramp = np.exp(1j * 2.0 * np.pi * shifts[i] * kx).astype(np.complex64)
        out[ky] = kspace_c[ky] * ramp
    return out


# ── Undersampling mask ─────────────────────────────────────────────────────────

def generate_vds_mask(n_lines, accel=4, acs_frac=0.08, seed=None):
    """Variable-density Cartesian ky mask, returns bool (n_lines,)."""
    rng = np.random.default_rng(seed)
    n_acs   = max(8, int(n_lines * acs_frac))
    n_total = max(n_acs, n_lines // accel)
    n_outer = n_total - n_acs
    mask = np.zeros(n_lines, dtype=bool)
    start = (n_lines - n_acs) // 2
    mask[start:start + n_acs] = True
    outer = np.where(~mask)[0]
    probs = np.exp(-((outer - n_lines // 2) ** 2) / (2.0 * (n_lines * 0.25) ** 2))
    probs /= probs.sum()
    chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False, p=probs)
    mask[chosen] = True
    return mask


# ── Multi-coil MRI forward model ──────────────────────────────────────────────

def mri_forward_multicoil(x_true, coil_maps_nominal, coil_perturb, mask_1d,
                           b0_hz, b0_map, warp_field, k_traj_frac, noise_sigma, rng):
    """True (mismatched) multi-coil MRI forward model.

    Nominal model (what algorithms assume):
        y_c = mask · F(S_c · x)

    True acquisition:
        x_warped = warp(x, δr)                              [gradient nonlin]
        x_mod    = x_warped · exp(i·2π·B0_hz·TE·b0_map)    [B0 mismatch]
        S_c_true = S_c_nominal · (1 + ε_c)                 [coil mismatch]
        y_c      = mask · k_traj_err(F(S_c_true · x_mod)) + n_c

    Returns y: (C, H, W) complex64
    """
    C, H, W = coil_maps_nominal.shape
    mask_2d = mask_1d[:, np.newaxis] * np.ones((1, W), dtype=bool)

    x_warped = apply_warp(x_true, warp_field)
    phi = (2.0 * np.pi * b0_hz * TE_S * b0_map).astype(np.float32)
    x_mod = x_warped.astype(np.complex64) * np.exp(1j * phi).astype(np.complex64)

    coil_maps_true = coil_maps_nominal * (1.0 + coil_perturb)

    y_multi = np.zeros((C, H, W), dtype=np.complex64)
    for c in range(C):
        kspace_c = _fft2c(x_mod * coil_maps_true[c]).astype(np.complex64)
        kspace_m = kspace_c * mask_2d
        kspace_m = apply_k_trajectory_error(kspace_m, mask_1d, k_traj_frac, rng)
        sig_std  = float(np.abs(kspace_c[mask_2d]).std()) + 1e-8
        noise    = ((rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
                    * noise_sigma * sig_std).astype(np.complex64)
        y_multi[c] = kspace_m + noise * mask_2d

    return y_multi.astype(np.complex64)


def rss_recon(y_kspace):
    """Zero-filled RSS reconstruction from undersampled multi-coil k-space."""
    imgs = _ifft2c(y_kspace)
    rss  = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0)).astype(np.float32)
    if rss.max() > 1e-6:
        rss /= rss.max()
    return rss


# ── Image helpers ──────────────────────────────────────────────────────────────

def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr, path):
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(path)


def _resize(arr, h, w):
    pil = Image.fromarray(np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L")
    return np.array(pil.resize((w, h), Image.LANCZOS)) / 255.0


def make_sample_images(x_true, y_kspace, coil_maps, mask_1d, b0_map,
                       sample_dir, spec):
    os.makedirs(sample_dir, exist_ok=True)
    C, H, W = y_kspace.shape

    rss        = rss_recon(y_kspace)
    kspace_log = _norm(np.log1p(np.mean(np.abs(y_kspace), axis=0)))
    mask_2d    = (mask_1d[:, None] * np.ones((1, W))).astype(np.float32)

    _save_png(x_true,          os.path.join(sample_dir, "ground_truth.png"))
    _save_png(rss,             os.path.join(sample_dir, "rss_reconstruction.png"))
    _save_png(kspace_log,      os.path.join(sample_dir, "kspace_magnitude.png"))
    _save_png(mask_2d,         os.path.join(sample_dir, "undersampling_mask.png"))
    _save_png(_norm(b0_map),   os.path.join(sample_dir, "b0_map.png"))

    # Coil sensitivity mosaic (3 rows × 5 cols for 15 coils)
    cols = 5
    rows = (C + cols - 1) // cols
    th, tw = 64, 64
    mosaic = np.zeros((rows * th, cols * tw), dtype=np.float32)
    for c in range(C):
        r, col = c // cols, c % cols
        mosaic[r*th:(r+1)*th, col*tw:(col+1)*tw] = _resize(np.abs(coil_maps[c]), th, tw)
    _save_png(mosaic, os.path.join(sample_dir, "coil_sensitivity.png"))

    # 2×3 overview
    th2, tw2 = 128, 128
    ov = np.zeros((2 * th2, 3 * tw2), dtype=np.float32)
    ov[0:th2,  0:tw2]     = _resize(x_true,                        th2, tw2)
    ov[0:th2,  tw2:2*tw2] = _resize(rss,                           th2, tw2)
    ov[0:th2,  2*tw2:]    = _resize(kspace_log,                    th2, tw2)
    ov[th2:,   0:tw2]     = _resize(mask_2d,                       th2, tw2)
    ov[th2:,   tw2:2*tw2] = _resize(_norm(b0_map),                 th2, tw2)
    ov[th2:,   2*tw2:]    = _resize(_norm(np.abs(coil_maps).mean(0)), th2, tw2)
    _save_png(ov, os.path.join(sample_dir, "overview.png"))

    with open(os.path.join(sample_dir, "spec.json"), "w") as fh:
        json.dump(spec, fh, indent=2)


# ── Shepp-Logan fallback (used when fastMRI data is unavailable) ───────────────

def shepp_logan_phantom(shape, variant=0):
    """Analytic Shepp-Logan phantom (float32 in [0, 1])."""
    H, W = shape
    yy = np.linspace(-1.0, 1.0, H, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float64)[None, :]

    # Ellipse parameters: (value, a, b, x0, y0, angle_deg)
    base = [
        ( 1.00, 0.69, 0.92,  0.00,  0.00,   0),
        (-0.98, 0.66, 0.87,  0.00,  0.00,   0),
        ( 0.80, 0.11, 0.31, -0.22,  0.00,  -18 + variant * 5),
        (-0.80, 0.16, 0.41,  0.22,  0.00,   18 - variant * 5),
        ( 0.35, 0.21, 0.25,  0.00,  0.35,   0),
        ( 0.35, 0.046,0.046, 0.00,  0.10,   0),
        ( 0.35, 0.046,0.046, 0.00, -0.10,   0),
        ( 0.35, 0.046,0.023,-0.08, -0.605,  0),
        ( 0.35, 0.023,0.023, 0.00, -0.606,  0),
        ( 0.35, 0.023,0.046, 0.06, -0.605,  0),
    ]
    img = np.zeros((H, W), dtype=np.float64)
    for val, a, b, x0, y0, ang in base:
        c_a, s_a = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        yr = (yy - y0) * c_a - (xx - x0) * s_a
        xr = (yy - y0) * s_a + (xx - x0) * c_a
        mask = ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0
        img[mask] += val
    img = img.clip(0.0, 1.0)

    # Per-variant tweak: add a small bright inclusion
    rng = np.random.default_rng(variant * 999 + 42)
    cx = float(rng.uniform(-0.3, 0.3))
    cy = float(rng.uniform(-0.3, 0.3))
    r  = float(rng.uniform(0.04, 0.10))
    mask_c = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
    img[mask_c] = img[mask_c] * 0.4 + 0.6

    return img.clip(0.0, 1.0).astype(np.float32)


# ── Tier builder ───────────────────────────────────────────────────────────────

def _make_coil_maps_for_sample(n_coils, shape, rng, kspace_full=None):
    """Return nominal coil maps.
    For real fastMRI samples, uses ACS-estimated maps (already computed).
    For synthetic samples, generates ring model.
    """
    if kspace_full is not None:
        # Re-estimate from the full k-space (already done during loading)
        return _estimate_coil_maps_acs(kspace_full, acs_frac=ACS_FRAC)
    return generate_coil_maps(shape, n_coils, rng)


def build_tier(tier, scenes, output_dir, spec_ranges_key, base_seed,
               is_fastmri_public=False):
    """Build one tier: H5 file + per-sample PNG images.

    scenes items:
      For synthetic: (scene_name, x_true, recipe_str)
      For fastMRI:   (scene_name, x_true, kspace_full, coil_maps_acs, recipe_str)
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_path    = os.path.join(output_dir, f"mri_challenge_{tier}.h5")
    images_dir = os.path.join(output_dir, "images")
    sr         = SPEC_RANGES[spec_ranges_key]
    rng        = np.random.default_rng(base_seed)
    table      = []

    with h5py.File(h5_path, "w") as hf:
        for i, scene_tuple in enumerate(scenes):
            if is_fastmri_public and len(scene_tuple) == 5:
                scene_name, x_true, kspace_full, coil_maps_acs, recipe = scene_tuple
            else:
                scene_name, x_true, recipe = scene_tuple
                kspace_full, coil_maps_acs = None, None

            grp = hf.create_group(f"sample_{i:02d}")

            def _u(k):
                return float(rng.uniform(sr[k]["min"], sr[k]["max"]))

            b0_hz        = _u("B0_inhomog_hz")
            grad_frac    = _u("gradient_nonlin_frac")
            coil_frac    = _u("coil_sensitivity_frac")
            ktraj_frac   = _u("k_trajectory_frac")
            noise_sigma  = _u("noise_sigma")

            # Coil maps: real ACS estimate (public) or synthetic ring (dev/hidden)
            if coil_maps_acs is not None:
                coil_maps = coil_maps_acs.astype(np.complex64)
            else:
                coil_maps = generate_coil_maps(SHAPE, N_COILS, rng)

            b0_map      = generate_b0_map(SHAPE, rng)
            warp_field  = generate_warp_field(SHAPE, grad_frac, rng)
            coil_perturb = generate_coil_perturbation(coil_maps, coil_frac, rng)
            mask_1d     = generate_vds_mask(SHAPE[0], ACCEL, ACS_FRAC,
                                            seed=base_seed + i * 997 + 3)

            y_kspace = mri_forward_multicoil(
                x_true, coil_maps, coil_perturb, mask_1d,
                b0_hz, b0_map, warp_field, ktraj_frac, noise_sigma, rng,
            )

            grp.create_dataset("x_true",     data=x_true,                 compression="gzip")
            grp.create_dataset("y_kspace",   data=y_kspace,               compression="gzip")
            grp.create_dataset("mask",       data=mask_1d.astype(np.uint8))
            grp.create_dataset("coil_maps",  data=coil_maps,              compression="gzip")
            grp.create_dataset("B0_map",     data=b0_map,                 compression="gzip")
            grp.create_dataset("warp_field", data=warp_field,             compression="gzip")

            true_spec = {
                "B0_inhomog_hz":         round(b0_hz, 6),
                "gradient_nonlin_frac":  round(grad_frac, 6),
                "coil_sensitivity_frac": round(coil_frac, 6),
                "k_trajectory_frac":     round(ktraj_frac, 6),
                "noise_sigma":           round(noise_sigma, 6),
            }
            metadata = {
                "scene":           scene_name,
                "shape":           list(SHAPE),
                "n_coils":         N_COILS,
                "accel_factor":    ACCEL,
                "acs_frac":        ACS_FRAC,
                "te_s":            TE_S,
                "recipe":          recipe,
                "n_sampled_lines": int(mask_1d.sum()),
                "source":          "fastmri_knee" if is_fastmri_public else "synthetic",
            }
            grp.attrs["metadata"]    = json.dumps(metadata)
            grp.attrs["spec_ranges"] = json.dumps(sr)
            grp.attrs["true_spec"]   = json.dumps(true_spec)

            sample_dir = os.path.join(images_dir, f"sample_{i:02d}_{scene_name}")
            make_sample_images(x_true, y_kspace, coil_maps, mask_1d, b0_map,
                               sample_dir,
                               {"scene": scene_name, "spec_ranges": sr,
                                "true_spec": true_spec})

            row = {**true_spec, "sample_idx": i, "scene": scene_name,
                   "recipe": recipe, "n_sampled_lines": int(mask_1d.sum())}
            table.append(row)
            print(f"  [{tier}] {i:02d} {scene_name}: "
                  f"B0={b0_hz:.1f}Hz grad={grad_frac:.4f} coil={coil_frac:.3f} "
                  f"ktraj={ktraj_frac:.4f} σ={noise_sigma:.4f} recipe={recipe}")

    return table


# ── README writer ──────────────────────────────────────────────────────────────

def write_tier_readme(tier, output_dir, table, spec_ranges_key, fastmri_public=False):
    sr = SPEC_RANGES[spec_ranges_key]
    rows = "".join(
        f"| sample_{s['sample_idx']:02d}  | {s['scene']:<26} | "
        f"{s['B0_inhomog_hz']:6.1f} | {s['gradient_nonlin_frac']:.4f} | "
        f"{s['coil_sensitivity_frac']:.3f} | {s['k_trajectory_frac']:.4f} | "
        f"{s['noise_sigma']:.4f} | {s['recipe']} |\n"
        for s in table
    )
    if fastmri_public:
        source = ("fastMRI multi-coil knee (2D Cartesian TSE, 320×320, 15 coils)\n"
                  "Reference: Zbontar et al., arXiv:1811.08839\n"
                  "Download:  https://fastmri.med.nyu.edu/")
    else:
        src_map = {
            "dev":    "Procedural knee-like phantoms (20 samples, TSE tissue statistics)",
            "hidden": "Adversarial knee phantoms (20 samples, severe mismatch)",
            "public": "Synthetic Shepp-Logan variants (PLACEHOLDER — set FASTMRI_ROOT for real data)",
        }
        source = src_map.get(tier, "Procedural synthetic")

    text = f"""# MRI {tier.capitalize()} Tier

## Source
{source}

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
{rows}
## HDF5 Datasets (per sample)

| Key           | Shape              | Dtype     | Description                             |
|---------------|--------------------|-----------|---------------------------------------------|
| `x_true`      | (320, 320)         | float32   | GT magnitude image [0, 1]               |
| `y_kspace`    | (15, 320, 320)     | complex64 | Undersampled k-space per coil           |
| `mask`        | (320,)             | uint8     | 1D ky undersampling mask                |
| `coil_maps`   | (15, 320, 320)     | complex64 | Nominal coil sensitivity maps           |
| `B0_map`      | (320, 320)         | float32   | True B0 field map (oracle)              |
| `warp_field`  | (2, 320, 320)      | float32   | True gradient warp (dy, dx) in pixels   |

## Image Files (per sample)

- `ground_truth.png`       — True MR magnitude image
- `rss_reconstruction.png` — Zero-filled RSS (shows aliasing artefacts)
- `kspace_magnitude.png`   — Log|y| averaged over coils
- `undersampling_mask.png` — Cartesian ky undersampling pattern
- `coil_sensitivity.png`   — Mosaic of |S_c| for all 15 coils (3×5 grid)
- `b0_map.png`             — B0 field inhomogeneity map
- `overview.png`           — 2×3 summary grid
- `spec.json`              — Per-sample mismatch specification
"""
    with open(os.path.join(output_dir, "README.md"), "w") as fh:
        fh.write(text)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("Multi-coil MRI benchmark (PWM parallel imaging / fastMRI knee)")
    print(f"Shape={SHAPE}  Coils={N_COILS}  Accel={ACCEL}x  "
          f"ACS={ACS_FRAC*100:.0f}%  TE={TE_S*1e3:.0f}ms")
    print("Mismatch: B0_inhomog · gradient_nonlin · coil_sensitivity · k_trajectory")
    print("=" * 70)

    # ── Public tier ──────────────────────────────────────────────────────────
    print(f"\n[public] fastMRI multi-coil knee ({PUBLIC_N_SAMPLES} samples)...")
    fastmri_scenes = load_public_fastmri(PUBLIC_N_SAMPLES, SHAPE)
    pub_is_fastmri = fastmri_scenes is not None

    if fastmri_scenes is None:
        print("  [public] Falling back to Shepp-Logan synthetic placeholders.")
        pub_scenes = [
            (f"shepp_logan_{i:02d}", shepp_logan_phantom(SHAPE, i), "shepp_logan")
            for i in range(PUBLIC_N_SAMPLES)
        ]
    else:
        pub_scenes = fastmri_scenes

    pub_dir = os.path.join(base_dir, "public")
    pub_t   = build_tier("public", pub_scenes, pub_dir, "public",
                          base_seed=1000, is_fastmri_public=pub_is_fastmri)
    write_tier_readme("public", pub_dir, pub_t, "public",
                      fastmri_public=pub_is_fastmri)

    # ── Dev tier ─────────────────────────────────────────────────────────────
    print("\n[dev] Procedural knee-like (20 samples)...")
    dev_scenes = [
        (f"proc_dev_{i:02d}", *generate_mri_gt(5000 + i, "dev", SHAPE))
        for i in range(20)
    ]
    dev_dir = os.path.join(base_dir, "dev")
    dev_t   = build_tier("dev", dev_scenes, dev_dir, "dev", base_seed=2000)
    write_tier_readme("dev", dev_dir, dev_t, "dev")

    # ── Hidden tier ──────────────────────────────────────────────────────────
    print("\n[hidden] Adversarial (20 samples)...")
    hid_scenes = [
        (f"proc_hidden_{i:02d}", *generate_mri_gt(8000 + i, "hidden", SHAPE))
        for i in range(20)
    ]
    hid_dir = os.path.join(base_dir, "hidden")
    hid_t   = build_tier("hidden", hid_scenes, hid_dir, "hidden", base_seed=3000)
    write_tier_readme("hidden", hid_dir, hid_t, "hidden")

    print("\n" + "=" * 70)
    print(f"Done.  public={len(pub_t)}  dev={len(dev_t)}  hidden={len(hid_t)} samples")
    if not pub_is_fastmri:
        print("NOTE: Public tier used synthetic fallback — set FASTMRI_ROOT for real data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
