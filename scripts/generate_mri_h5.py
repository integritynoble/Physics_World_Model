#!/usr/bin/env python3
"""
Generate MRI benchmark H5 files from fastMRI brain dataset.

H5 schema:
  sample_NN/
    x_true:  (320, 320) float32  — RSS-combined ground-truth image
    y:       (320, 320) complex64 — undersampled k-space (as float32 real/imag stacked)
    H_ideal: (320, 320) complex64 — fully-sampled k-space (reference)
  attrs: tier, variant, version, acceleration, center_fraction

DOWNLOAD (requires fastMRI license agreement):
  1. Register at https://fastmri.med.nyu.edu/ and agree to license
  2. Download brain_multicoil_val.tar.gz (~50 GB)
  3. Extract to /path/to/fastmri/brain_multicoil_val/

  OR use the smaller knee dataset (knee_multicoil_val) for testing.

Usage:
  python scripts/generate_mri_h5.py \\
      --fastmri_dir /path/to/fastmri/brain_multicoil_val \\
      --out_dir datasets/benchmark/mri \\
      [--split public|dev|hidden]

Split design:
  public  — first 25 slices (one per volume), nominal 4× acceleration
  dev     — first 64 slices, nominal 4× acceleration
  hidden  — slices 64-127, acceleration ~ Uniform[4, 8],
             off-resonance ~ Uniform[-100, 100] Hz simulated
"""

import argparse
import json
import os

import h5py
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
VARIANT = "mri"
VERSION = "1.0"
IMG_SIZE = 320
ACCELERATION_NOMINAL = 4
CENTER_FRACTION_NOMINAL = 0.08

RNG = np.random.default_rng(42)


# ── k-space sampling mask ─────────────────────────────────────────────────────
def variable_density_mask(shape: tuple, acceleration: int,
                           center_fraction: float,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Generate 1D variable-density undersampling mask (equispaced in phase-encode).
    Compatible with fastMRI's mask generation.

    Returns: (n_freq, n_phase) bool mask
    """
    n_freq, n_phase = shape
    n_center = int(round(n_phase * center_fraction))
    n_keep = n_phase // acceleration

    mask_1d = np.zeros(n_phase, dtype=bool)
    # Always keep center lines
    center_start = (n_phase - n_center) // 2
    mask_1d[center_start: center_start + n_center] = True

    # Randomly sample remaining lines from outer k-space
    outer = np.where(~mask_1d)[0]
    n_outer = max(0, n_keep - n_center)
    chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False)
    mask_1d[chosen] = True

    # Broadcast to 2D
    return np.tile(mask_1d, (n_freq, 1))


def apply_off_resonance(kspace: np.ndarray,
                        delta_f_hz: float,
                        tr_ms: float = 6.0) -> np.ndarray:
    """
    Simulate off-resonance by applying a linear phase ramp in image space.
    delta_f_hz: off-resonance frequency in Hz
    tr_ms: readout time per line in ms (approximate)
    """
    if abs(delta_f_hz) < 1e-3:
        return kspace
    n_phase = kspace.shape[1]
    t = np.linspace(0, tr_ms * n_phase / 1000, n_phase)  # time in seconds
    phase_ramp = np.exp(1j * 2 * np.pi * delta_f_hz * t)
    return kspace * phase_ramp[np.newaxis, :]


# ── fastMRI loader ────────────────────────────────────────────────────────────
def load_fastmri_kspace(fastmri_dir: str, volume_idx: int,
                         slice_idx: int = 15) -> np.ndarray:
    """
    Load one k-space slice from a fastMRI multicoil volume.

    fastMRI HDF5 structure:
      kspace: (n_slices, n_coils, n_freq, n_phase) complex64

    Returns: (n_freq, n_phase) complex64 RSS-reconstructed k-space
    """
    files = sorted([f for f in os.listdir(fastmri_dir) if f.endswith(".h5")])
    if volume_idx >= len(files):
        raise FileNotFoundError(
            f"Volume {volume_idx} not found in {fastmri_dir}. "
            f"Only {len(files)} volumes available.\n"
            "Download from https://fastmri.med.nyu.edu/"
        )

    fpath = os.path.join(fastmri_dir, files[volume_idx])
    with h5py.File(fpath, "r") as f:
        kspace_mc = f["kspace"][:]  # (n_slices, n_coils, H, W)

    n_slices = kspace_mc.shape[0]
    sl = min(slice_idx, n_slices - 1)
    kspace_1slice = kspace_mc[sl]  # (n_coils, H, W)

    # RSS combination in image space, then back to k-space
    imgs = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_1slice, axes=(-2, -1)),
                     axes=(-2, -1)),
        axes=(-2, -1)
    )  # (n_coils, H, W)
    rss = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0))  # (H, W)

    # Normalize and resize to IMG_SIZE × IMG_SIZE
    rss = rss / (rss.max() + 1e-8)
    if rss.shape != (IMG_SIZE, IMG_SIZE):
        from skimage.transform import resize
        rss = resize(rss, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

    # Back to k-space
    kspace_full = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(rss.astype(np.complex64)))
    )
    return kspace_full.astype(np.complex64)


def generate_synthetic_kspace(size: int = 320,
                               rng: np.random.Generator = None) -> np.ndarray:
    """Synthetic Shepp-Logan phantom k-space as fallback."""
    if rng is None:
        rng = np.random.default_rng()
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = shepp_logan_phantom()
    img = resize(phantom, (size, size), anti_aliasing=True).astype(np.float32)
    # Add random variation
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)

    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))).astype(np.complex64)
    return kspace


# ── Main generator ────────────────────────────────────────────────────────────
SPLIT_RANGES = {
    "public": (0, 25),
    "dev":    (0, 64),
    "hidden": (64, 128),
}


def generate_split(fastmri_dir: str, out_dir: str, split: str,
                   use_synthetic_fallback: bool = False) -> None:
    start_idx, end_idx = SPLIT_RANGES[split]
    n_samples = end_idx - start_idx

    out_path = os.path.join(out_dir, split, f"mri_challenge_{split}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Generating MRI {split} split: {n_samples} samples → {out_path}")

    with h5py.File(out_path, "w") as hf:
        hf.attrs["variant"] = VARIANT
        hf.attrs["version"] = VERSION
        hf.attrs["tier"] = split
        hf.attrs["acceleration_nominal"] = ACCELERATION_NOMINAL
        hf.attrs["center_fraction_nominal"] = CENTER_FRACTION_NOMINAL
        hf.attrs["img_size"] = IMG_SIZE
        hf.attrs["dataset"] = "fastmri_brain"
        hf.attrs["citation"] = (
            "Zbontar, J. et al. (2018) fastMRI: An Open Dataset, arXiv 1811.08839"
        )

        for i, vol_idx in enumerate(range(start_idx, end_idx)):
            grp = hf.create_group(f"sample_{i:02d}")

            # ── Load ground-truth k-space ──────────────────────────────────
            try:
                kspace_full = load_fastmri_kspace(fastmri_dir, vol_idx)
            except (FileNotFoundError, Exception):
                if use_synthetic_fallback:
                    kspace_full = generate_synthetic_kspace(IMG_SIZE, RNG)
                else:
                    raise

            # Reconstruct fully-sampled ground truth image
            x_true_complex = np.fft.ifftshift(
                np.fft.ifft2(np.fft.ifftshift(kspace_full))
            )
            x_true = np.abs(x_true_complex).astype(np.float32)
            x_true = x_true / (x_true.max() + 1e-8)

            # ── Mismatch params ────────────────────────────────────────────
            if split == "hidden":
                acc = int(RNG.integers(4, 9))                          # [4, 8]
                cf = float(RNG.uniform(0.04, 0.12))                    # center frac
                off_res_hz = float(RNG.uniform(-100, 100))             # off-resonance
            else:
                acc = ACCELERATION_NOMINAL
                cf = CENTER_FRACTION_NOMINAL
                off_res_hz = 0.0

            # ── Apply off-resonance to k-space ─────────────────────────────
            kspace_shifted = apply_off_resonance(kspace_full, off_res_hz)

            # ── Apply undersampling mask ───────────────────────────────────
            mask = variable_density_mask(
                kspace_shifted.shape, acc, cf, RNG
            )
            kspace_under = kspace_shifted * mask  # (320, 320) complex

            # Store as (320, 320, 2) float32 [real, imag] interleaved
            y_ri = np.stack([kspace_under.real, kspace_under.imag], axis=-1).astype(np.float32)
            h_ri = np.stack([kspace_full.real, kspace_full.imag], axis=-1).astype(np.float32)

            grp.create_dataset("x_true",  data=x_true,  compression="gzip")
            grp.create_dataset("y",       data=y_ri,    compression="gzip")
            grp.create_dataset("H_ideal", data=h_ri,    compression="gzip")
            grp.create_dataset("mask",    data=mask.astype(np.uint8), compression="gzip")

            grp.attrs["mismatch_params"] = json.dumps({
                "acceleration": acc,
                "center_fraction": cf,
                "off_resonance_hz": off_res_hz,
            })

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  [{i+1}/{n_samples}] vol {vol_idx} "
                      f"acc={acc}× off_res={off_res_hz:.0f}Hz")

    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Generate MRI benchmark H5 files")
    parser.add_argument("--fastmri_dir", type=str, required=True,
                        help="Path to fastMRI brain_multicoil_val directory")
    parser.add_argument("--out_dir", type=str,
                        default="datasets/benchmark/mri")
    parser.add_argument("--split", type=str, default="all",
                        choices=["public", "dev", "hidden", "all"])
    parser.add_argument("--synthetic_fallback", action="store_true",
                        help="Use Shepp-Logan phantoms if fastMRI not available")
    args = parser.parse_args()

    splits = ["public", "dev", "hidden"] if args.split == "all" else [args.split]
    for split in splits:
        generate_split(args.fastmri_dir, args.out_dir, split,
                       use_synthetic_fallback=args.synthetic_fallback)

    print("\nDone. Upload with:")
    print("  gsutil -m cp -r datasets/benchmark/mri/ "
          "gs://pwm-benchmark-datasets/challenge-data/v1.0/")


if __name__ == "__main__":
    main()
