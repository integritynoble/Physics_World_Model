#!/usr/bin/env python3
"""
Generate CBCT benchmark H5 files.

Uses TCIA Head-Neck-PET-CT (if available) or simulation from CT volumes.
Cone-beam geometry approximated as fan-beam 2D for each axial slice.

H5 schema:
  sample_NN/
    x_true:  (256, 256) float32  — ground-truth CT slice
    y:       (180, 364) float32  — CBCT sinogram (fan-beam approximated)
    H_ideal: (180, 364) float32  — noise-free sinogram
  attrs: tier, variant, n_views, src_det_dist, isocenter_dist

DOWNLOAD options:
  Option A — TCIA (requires account):
    https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/
    pip install tcia_utils; from tcia_utils import nbia

  Option B — LoDoPaB-CT (same Zenodo 3384092 as CT):
    Reuse CT volumes with CBCT geometry (different SOD/SDD, scatter)

  Option C — Fully synthetic (default, --synthetic_fallback):
    Shepp-Logan + random ellipses

Usage:
  python scripts/generate_cbct_h5.py \\
      --ct_dir /path/to/ct_volumes \\    # DICOM or NIfTI root
      --out_dir datasets/benchmark/cbct \\
      [--synthetic_fallback]

Split design:
  public  — 25 slices, nominal geometry (SOD=750mm, SDD=1000mm, 180 views)
  dev     — 64 slices, nominal geometry
  hidden  — 64 slices, perturbed: n_views~[60,180], scatter_coeff~[0,0.05],
             isocenter_offset~[-5,5]mm
"""

import argparse
import json
import os

import h5py
import numpy as np

try:
    from skimage.transform import radon
except ImportError:
    raise ImportError("pip install scikit-image")

VARIANT = "cbct"
VERSION = "1.0"
IMG_SIZE = 256
N_VIEWS_NOMINAL = 180
N_DETECTORS = 364   # for 256×256 with fan-beam
SOD_MM = 750.0      # source-to-isocenter distance
SDD_MM = 1000.0     # source-to-detector distance

SPLIT_RANGES = {"public": (0, 25), "dev": (0, 64), "hidden": (64, 128)}
RNG = np.random.default_rng(42)


def fan_beam_sinogram(x: np.ndarray, n_views: int, n_det: int,
                      sod: float, sdd: float,
                      scatter_coeff: float = 0.0,
                      photon_count: float = 1e5) -> tuple:
    """
    Approximate fan-beam CBCT sinogram using parallel-to-fan rebinning.
    For simplicity uses parallel-beam Radon then rebins.
    """
    angles = np.linspace(0, 360, n_views, endpoint=False)
    # Use parallel beam Radon as approximation
    sino_parallel = radon(x, theta=angles[:n_views // 2], circle=True)
    # Mirror for full 360°
    sino_full = np.concatenate([sino_parallel, sino_parallel], axis=1).T

    # Resize to target detector count
    from skimage.transform import resize
    y_ideal = resize(sino_full, (n_views, n_det), anti_aliasing=True).astype(np.float32)
    # Normalize
    y_ideal = y_ideal / (y_ideal.max() + 1e-8)

    # Scatter: additive constant term
    if scatter_coeff > 0:
        y_ideal = y_ideal + scatter_coeff * np.mean(y_ideal)

    # Poisson noise
    I = photon_count * np.exp(-y_ideal)
    I_noisy = RNG.poisson(np.maximum(I, 1e-6)).astype(np.float32)
    y_noisy = -np.log(np.maximum(I_noisy, 1.0) / photon_count).astype(np.float32)

    return y_ideal, y_noisy


def load_ct_slice(ct_dir: str, idx: int, img_size: int) -> np.ndarray:
    """Try to load a CT slice from various formats."""
    # Try NIfTI
    nii_files = sorted([f for f in os.listdir(ct_dir)
                        if f.endswith(".nii") or f.endswith(".nii.gz")])
    if idx < len(nii_files):
        import nibabel as nib
        vol = nib.load(os.path.join(ct_dir, nii_files[idx])).get_fdata()
        sl = vol[:, :, vol.shape[2] // 2].astype(np.float32)
        from skimage.transform import resize
        sl = resize(sl, (img_size, img_size), anti_aliasing=True)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        return sl.astype(np.float32)

    # Try H5
    h5_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".h5")])
    if idx < len(h5_files):
        with h5py.File(os.path.join(ct_dir, h5_files[idx]), "r") as f:
            key = list(f.keys())[0]
            sl = f[key][0].astype(np.float32)
        from skimage.transform import resize
        sl = resize(sl, (img_size, img_size), anti_aliasing=True)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        return sl.astype(np.float32)

    raise FileNotFoundError(f"No CT volume {idx} in {ct_dir}")


def synthetic_phantom(size: int, rng) -> np.ndarray:
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    base = resize(shepp_logan_phantom(), (size, size), anti_aliasing=True).astype(np.float32)
    # Add random ellipses
    Y, X = np.ogrid[:size, :size]
    for _ in range(rng.integers(2, 6)):
        cy, cx = rng.integers(60, 196, 2)
        ry, rx = rng.integers(8, 50, 2)
        val = rng.uniform(-0.05, 0.2)
        mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        base[mask] += val
    return np.clip(base, 0, 1).astype(np.float32)


def generate_split(ct_dir: str, out_dir: str, split: str,
                   use_synthetic: bool = False) -> None:
    start, end = SPLIT_RANGES[split]
    n = end - start
    out_path = os.path.join(out_dir, split, f"cbct_challenge_{split}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Generating CBCT {split}: {n} samples → {out_path}")

    with h5py.File(out_path, "w") as hf:
        hf.attrs.update({
            "variant": VARIANT, "version": VERSION, "tier": split,
            "n_views_nominal": N_VIEWS_NOMINAL, "n_detectors": N_DETECTORS,
            "sod_mm": SOD_MM, "sdd_mm": SDD_MM,
            "dataset": "tcia_head_neck_pet_ct",
            "citation": "Vallières, M. et al. (2017) Head-Neck-PET-CT, TCIA",
        })

        for i, idx in enumerate(range(start, end)):
            grp = hf.create_group(f"sample_{i:02d}")

            try:
                x_true = load_ct_slice(ct_dir, idx, IMG_SIZE)
            except Exception:
                if use_synthetic:
                    x_true = synthetic_phantom(IMG_SIZE, RNG)
                else:
                    raise

            if split == "hidden":
                n_views = int(RNG.integers(60, 181))
                scatter = float(RNG.uniform(0.0, 0.05))
                iso_offset = float(RNG.uniform(-5, 5))
                photon = float(10 ** RNG.uniform(4, 5))
            else:
                n_views = N_VIEWS_NOMINAL
                scatter = 0.0
                iso_offset = 0.0
                photon = 1e5

            y_ideal, y_noisy = fan_beam_sinogram(
                x_true, n_views, N_DETECTORS,
                SOD_MM + iso_offset, SDD_MM,
                scatter_coeff=scatter, photon_count=photon,
            )

            grp.create_dataset("x_true",  data=x_true,  compression="gzip")
            grp.create_dataset("y",       data=y_noisy, compression="gzip")
            grp.create_dataset("H_ideal", data=y_ideal, compression="gzip")
            grp.attrs["mismatch_params"] = json.dumps({
                "n_views": n_views, "scatter_coeff": scatter,
                "isocenter_offset_mm": iso_offset, "photon_count": photon,
            })

            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] idx={idx} n_views={n_views}")

    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="datasets/benchmark/cbct")
    parser.add_argument("--split", type=str, default="all",
                        choices=["public", "dev", "hidden", "all"])
    parser.add_argument("--synthetic_fallback", action="store_true")
    args = parser.parse_args()

    if args.ct_dir is None and not args.synthetic_fallback:
        print("No --ct_dir given. Using synthetic phantoms (add --synthetic_fallback to confirm).")
        args.synthetic_fallback = True

    splits = ["public", "dev", "hidden"] if args.split == "all" else [args.split]
    for split in splits:
        generate_split(args.ct_dir or "", args.out_dir, split,
                       use_synthetic=args.synthetic_fallback)

    print("\nDone. Upload with:")
    print("  gsutil -m cp -r datasets/benchmark/cbct/ "
          "gs://pwm-benchmark-datasets/challenge-data/v1.0/")


if __name__ == "__main__":
    main()
