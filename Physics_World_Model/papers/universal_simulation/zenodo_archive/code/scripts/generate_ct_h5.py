#!/usr/bin/env python3
"""
Generate CT benchmark H5 files from LoDoPaB-CT (Zenodo 3384092).

H5 schema (matches existing cacti/cassi format):
  sample_NN/
    x_true:  (362, 362) float32  — ground-truth attenuation map
    y:       (60, 513)  float32  — sparse-view sinogram (60 angles)
    H_ideal: (60, 513)  float32  — noise-free sinogram (reference)
  attrs: tier, variant, version, n_views, n_detectors

DOWNLOAD (one-time, ~10 GB total):
  pip install zenodo-get
  zenodo_get 3384092 -o /path/to/lodopab

  OR manually from:
  https://zenodo.org/record/3384092
  Files needed: ground_truth_train.zip, observation_train.zip,
                ground_truth_validation.zip, observation_validation.zip

Usage:
  python scripts/generate_ct_h5.py \\
      --lodopab_dir /path/to/lodopab \\
      --out_dir datasets/benchmark/ct \\
      [--split public|dev|hidden]

Split design:
  public  — validation patients 0-24   (25 samples, nominal 60-view)
  dev     — validation patients 0-63   (64 samples, nominal 60-view)
  hidden  — validation patients 64-127 (64 samples, n_views ~ Uniform[40,90])
             + beam-hardening coeff ~ Uniform[0.0, 0.15]
"""

import argparse
import json
import os

import h5py
import numpy as np

try:
    from skimage.transform import radon, iradon
except ImportError:
    raise ImportError("pip install scikit-image")

# ── Config ────────────────────────────────────────────────────────────────────
VARIANT = "ct"
VERSION = "1.0"
IMG_SIZE = 362       # LoDoPaB-CT native resolution
N_VIEWS_NOMINAL = 60
N_DETECTORS = 513    # standard for 362×362 at 1:1 pixel-to-detector ratio

SPLIT_RANGES = {
    "public": (0, 25),    # val patients 0-24
    "dev":    (0, 64),    # val patients 0-63
    "hidden": (64, 128),  # val patients 64-127
}

RNG = np.random.default_rng(42)


# ── Forward model ─────────────────────────────────────────────────────────────
def make_sinogram(x_true: np.ndarray, angles_deg: np.ndarray,
                  photon_count: float = 1e5,
                  beam_hardening_coeff: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sparse-view sinogram with optional Poisson noise + beam hardening.

    Args:
        x_true: (H, W) attenuation map in 1/cm (LoDoPaB units)
        angles_deg: projection angles in degrees
        photon_count: mean photons per detector pixel (for Poisson noise)
        beam_hardening_coeff: cupping artifact coefficient (0 = monochromatic)

    Returns:
        y_ideal: (n_views, n_det) noise-free sinogram
        y_noisy: (n_views, n_det) Poisson-noisy sinogram
    """
    # Radon transform — sinogram shape: (n_det, n_views)
    sino = radon(x_true, theta=angles_deg, circle=True)  # (n_det, n_views)
    y_ideal = sino.T.astype(np.float32)                   # (n_views, n_det)

    # Beam hardening (polynomial cupping: p_corrected = p + c * p^2)
    if beam_hardening_coeff > 0:
        y_ideal = y_ideal + beam_hardening_coeff * y_ideal ** 2

    # Poisson noise via Beer-Lambert: I = I0 * exp(-p) → add photon noise
    I_transmitted = photon_count * np.exp(-y_ideal)
    I_noisy = RNG.poisson(np.maximum(I_transmitted, 1e-6)).astype(np.float32)
    y_noisy = -np.log(np.maximum(I_noisy, 1.0) / photon_count).astype(np.float32)

    return y_ideal, y_noisy


def pad_or_crop(y: np.ndarray, target_cols: int) -> np.ndarray:
    """Pad or crop sinogram to target_cols detectors."""
    n_rows, n_cols = y.shape
    if n_cols == target_cols:
        return y
    if n_cols > target_cols:
        trim = (n_cols - target_cols) // 2
        return y[:, trim: trim + target_cols]
    pad = target_cols - n_cols
    return np.pad(y, ((0, 0), (pad // 2, pad - pad // 2)))


# ── LoDoPaB-CT loader ─────────────────────────────────────────────────────────
def load_lodopab_slice(lodopab_dir: str, split_name: str,
                       patient_idx: int) -> np.ndarray:
    """
    Load one ground-truth slice from LoDoPaB-CT directory structure.

    LoDoPaB-CT stores images as float32 arrays in HDF5:
      ground_truth_{train|validation}/data/ground_truth_{i:06d}.hdf5
        dataset "data": (1, 362, 362)

    Returns: (362, 362) float32 attenuation map
    """
    subset = "validation" if split_name in ("public", "dev", "hidden") else "train"
    gt_dir = os.path.join(lodopab_dir, f"ground_truth_{subset}", "data")

    fname = os.path.join(gt_dir, f"ground_truth_{patient_idx:06d}.hdf5")
    if not os.path.exists(fname):
        # Fallback: flat directory with .npy files (exported format)
        fname_npy = os.path.join(gt_dir, f"{patient_idx:06d}.npy")
        if os.path.exists(fname_npy):
            return np.load(fname_npy).astype(np.float32).squeeze()
        raise FileNotFoundError(
            f"LoDoPaB-CT slice not found: {fname}\n"
            "Download from https://zenodo.org/record/3384092"
        )

    with h5py.File(fname, "r") as f:
        x = f["data"][0].astype(np.float32)   # (362, 362)
    # Rescale to [0, 1] range if stored in HU-like units
    if x.max() > 10:
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x


def generate_synthetic_phantom(size: int = 362, rng: np.random.Generator = None) -> np.ndarray:
    """
    Shepp-Logan phantom as fallback when LoDoPaB-CT is not available.
    Generates a randomized multi-ellipse phantom.
    """
    if rng is None:
        rng = np.random.default_rng()
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    base = shepp_logan_phantom()
    x = resize(base, (size, size), anti_aliasing=True).astype(np.float32)

    # Add random ellipses to diversify
    n_extra = rng.integers(3, 8)
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    for _ in range(n_extra):
        ey = rng.integers(10, 60)
        ex = rng.integers(10, 60)
        y0 = rng.integers(cy - 80, cy + 80)
        x0 = rng.integers(cx - 80, cx + 80)
        val = rng.uniform(-0.1, 0.3)
        mask = ((X - x0) / ex) ** 2 + ((Y - y0) / ey) ** 2 <= 1
        x[mask] += val

    return np.clip(x, 0, 1).astype(np.float32)


# ── Main generator ────────────────────────────────────────────────────────────
def generate_split(lodopab_dir: str, out_dir: str, split: str,
                   use_synthetic_fallback: bool = False) -> None:
    start_idx, end_idx = SPLIT_RANGES[split]
    n_samples = end_idx - start_idx
    angles_nominal = np.linspace(0, 180, N_VIEWS_NOMINAL, endpoint=False)

    out_path = os.path.join(out_dir, split, f"ct_challenge_{split}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Generating CT {split} split: {n_samples} samples → {out_path}")

    with h5py.File(out_path, "w") as hf:
        # File-level attributes
        hf.attrs["variant"] = VARIANT
        hf.attrs["version"] = VERSION
        hf.attrs["tier"] = split
        hf.attrs["n_views_nominal"] = N_VIEWS_NOMINAL
        hf.attrs["n_detectors"] = N_DETECTORS
        hf.attrs["img_size"] = IMG_SIZE
        hf.attrs["dataset"] = "lodopab_ct"
        hf.attrs["citation"] = (
            "Leuschner, J. et al. (2021) LoDoPaB-CT, Scientific Data 8:109"
        )

        for i, pat_idx in enumerate(range(start_idx, end_idx)):
            grp_name = f"sample_{i:02d}"
            grp = hf.create_group(grp_name)

            # ── Load ground truth ──────────────────────────────────────────
            try:
                x_true = load_lodopab_slice(lodopab_dir, split, pat_idx)
            except FileNotFoundError:
                if use_synthetic_fallback:
                    x_true = generate_synthetic_phantom(IMG_SIZE, RNG)
                else:
                    raise

            # ── Mismatch params (hidden split: randomized) ─────────────────
            if split == "hidden":
                n_views = int(RNG.integers(40, 91))          # [40, 90]
                photon_count = float(10 ** RNG.uniform(3, 5)) # 10^3 – 10^5
                bh_coeff = float(RNG.uniform(0.0, 0.15))     # beam hardening
            else:
                n_views = N_VIEWS_NOMINAL
                photon_count = 1e5
                bh_coeff = 0.0

            angles = np.linspace(0, 180, n_views, endpoint=False)

            # ── Forward model ──────────────────────────────────────────────
            y_ideal, y_noisy = make_sinogram(x_true, angles,
                                             photon_count=photon_count,
                                             beam_hardening_coeff=bh_coeff)

            # Pad/crop to standard detector count
            y_ideal_std = pad_or_crop(y_ideal, N_DETECTORS)
            y_noisy_std = pad_or_crop(y_noisy, N_DETECTORS)

            # If fewer views than nominal: zero-pad rows so shape is constant
            if n_views < N_VIEWS_NOMINAL:
                pad_rows = N_VIEWS_NOMINAL - n_views
                y_ideal_std = np.pad(y_ideal_std, ((0, pad_rows), (0, 0)))
                y_noisy_std = np.pad(y_noisy_std, ((0, pad_rows), (0, 0)))
            elif n_views > N_VIEWS_NOMINAL:
                y_ideal_std = y_ideal_std[:N_VIEWS_NOMINAL]
                y_noisy_std = y_noisy_std[:N_VIEWS_NOMINAL]

            # ── Write datasets ─────────────────────────────────────────────
            grp.create_dataset("x_true",  data=x_true,       compression="gzip")
            grp.create_dataset("y",       data=y_noisy_std,  compression="gzip")
            grp.create_dataset("H_ideal", data=y_ideal_std,  compression="gzip")

            # Mismatch params as JSON attr (matches cassi convention)
            grp.attrs["mismatch_params"] = json.dumps({
                "n_views": n_views,
                "photon_count": photon_count,
                "beam_hardening_coeff": bh_coeff,
                "detector_spacing_mm": 1.0,
            })

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  [{i+1}/{n_samples}] patient {pat_idx} "
                      f"n_views={n_views} bh={bh_coeff:.3f}")

    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Generate CT benchmark H5 files")
    parser.add_argument("--lodopab_dir", type=str, required=True,
                        help="Path to LoDoPaB-CT root (from Zenodo 3384092)")
    parser.add_argument("--out_dir", type=str,
                        default="datasets/benchmark/ct",
                        help="Output directory for H5 files")
    parser.add_argument("--split", type=str, default="all",
                        choices=["public", "dev", "hidden", "all"])
    parser.add_argument("--synthetic_fallback", action="store_true",
                        help="Use Shepp-Logan phantoms if LoDoPaB-CT not available")
    args = parser.parse_args()

    splits = ["public", "dev", "hidden"] if args.split == "all" else [args.split]
    for split in splits:
        generate_split(args.lodopab_dir, args.out_dir, split,
                       use_synthetic_fallback=args.synthetic_fallback)

    print("\nDone. Upload with:")
    print("  gsutil -m cp -r datasets/benchmark/ct/ "
          "gs://pwm-benchmark-datasets/challenge-data/v1.0/")


if __name__ == "__main__":
    main()
