#!/usr/bin/env python3
"""
Generate CACTI benchmark H5 files from DAVIS-2017 video dataset.

H5 schema (matches existing cacti format exactly):
  sample_NN/
    x_true:  (256, 256, 8) float64  — ground-truth video cube (T=8 frames)
    y:       (256, 256)    float64  — compressed snapshot measurement
    H_ideal: (256, 256, 8) float64  — binary modulation mask
  attrs: tier, variant, version

DOWNLOAD (free):
  DAVIS 2017 (480p):
    https://davischallenge.org/davis2017/code.html
  Direct download (~1.7 GB):
    wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

  After extract, structure is:
    /path/to/davis/DAVIS/JPEGImages/480p/{sequence_name}/{frame:05d}.jpg

Usage:
  python scripts/generate_cacti_h5.py \\
      --davis_dir /path/to/davis/DAVIS \\
      --out_dir datasets/benchmark/cacti \\
      [--split public|dev|hidden]

Split design:
  public  — 20 sequences, nominal binary mask (50% duty cycle, random seed 42)
  dev     — 50 sequences, nominal binary mask
  hidden  — 50 sequences, masks with:
             - random mask translation (dx, dy) ~ [-2, 2] pixels
             - random mask rotation ~ [-1°, 1°]
             - Gaussian mask blur sigma ~ [0, 0.5]
             - plus brightness/contrast jitter on frames
"""

import argparse
import json
import os
import glob

import h5py
import numpy as np

try:
    from PIL import Image
    from skimage.transform import resize, rotate
except ImportError:
    raise ImportError("pip install Pillow scikit-image")

VARIANT = "cacti"
VERSION = "2.0"
IMG_SIZE = 256
N_FRAMES = 8       # temporal frames per compressed snapshot

SPLIT_DEFS = {
    "public": {"n_samples": 20, "offset": 0},
    "dev":    {"n_samples": 50, "offset": 0},
    "hidden": {"n_samples": 50, "offset": 50},
}

RNG = np.random.default_rng(42)


# ── DAVIS loader ──────────────────────────────────────────────────────────────
def list_davis_sequences(davis_dir: str, res: str = "480p") -> list:
    """Return sorted list of sequence names."""
    jpg_dir = os.path.join(davis_dir, "JPEGImages", res)
    if not os.path.isdir(jpg_dir):
        # Try alternate layout
        jpg_dir = os.path.join(davis_dir, "JPEGImages")
    if not os.path.isdir(jpg_dir):
        raise FileNotFoundError(
            f"DAVIS JPEGImages not found at {jpg_dir}\n"
            "Download from https://davischallenge.org/davis2017/code.html"
        )
    return sorted(os.listdir(jpg_dir))


def load_davis_sequence(davis_dir: str, seq_name: str,
                        n_frames: int = N_FRAMES,
                        img_size: int = IMG_SIZE,
                        res: str = "480p") -> np.ndarray:
    """
    Load n_frames consecutive frames from a DAVIS sequence.
    Returns: (img_size, img_size, n_frames) float64, values in [0, 1].
    """
    jpg_dir = os.path.join(davis_dir, "JPEGImages", res, seq_name)
    if not os.path.isdir(jpg_dir):
        jpg_dir = os.path.join(davis_dir, "JPEGImages", seq_name)

    frame_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")) +
                         glob.glob(os.path.join(jpg_dir, "*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {jpg_dir}")

    # Pick n_frames evenly spaced across the sequence
    indices = np.linspace(0, len(frame_files) - 1, n_frames).astype(int)
    cube = np.zeros((img_size, img_size, n_frames), dtype=np.float64)

    for t, idx in enumerate(indices):
        img = Image.open(frame_files[idx]).convert("L")  # grayscale
        img_arr = np.array(img, dtype=np.float64) / 255.0
        cube[:, :, t] = resize(img_arr, (img_size, img_size), anti_aliasing=True)

    return cube


def augment_cube(cube: np.ndarray, rng,
                 brightness_shift: float = 0.0,
                 contrast_scale: float = 1.0) -> np.ndarray:
    """Apply brightness/contrast jitter to all frames."""
    cube = cube * contrast_scale + brightness_shift
    return np.clip(cube, 0, 1).astype(np.float64)


# ── Mask generation ───────────────────────────────────────────────────────────
def generate_mask(img_size: int, n_frames: int, rng,
                  dx: float = 0.0, dy: float = 0.0,
                  angle_deg: float = 0.0,
                  blur_sigma: float = 0.0,
                  duty_cycle: float = 0.5) -> np.ndarray:
    """
    Generate binary modulation mask: (img_size, img_size, n_frames).
    Each frame has an independent random binary mask.
    """
    mask = (rng.random((img_size, img_size, n_frames)) < duty_cycle).astype(np.float64)

    if abs(dx) > 0.01 or abs(dy) > 0.01:
        from scipy.ndimage import shift
        for t in range(n_frames):
            mask[:, :, t] = shift(mask[:, :, t], [dy, dx], mode="wrap")
            mask[:, :, t] = (mask[:, :, t] > 0.5).astype(np.float64)

    if abs(angle_deg) > 0.01:
        for t in range(n_frames):
            mask[:, :, t] = (
                rotate(mask[:, :, t], angle_deg, mode="wrap") > 0.5
            ).astype(np.float64)

    if blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        for t in range(n_frames):
            blurred = gaussian_filter(mask[:, :, t].astype(np.float32), sigma=blur_sigma)
            mask[:, :, t] = blurred.astype(np.float64)  # keep continuous for hidden

    return mask


# ── CACTI forward model ───────────────────────────────────────────────────────
def cacti_forward(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    CACTI compression: y[i,j] = sum_t mask[i,j,t] * cube[i,j,t]
    Returns: (img_size, img_size) float64
    """
    return np.sum(mask * cube, axis=2).astype(np.float64)


# ── Main generator ────────────────────────────────────────────────────────────
def generate_split(davis_dir: str, out_dir: str, split: str,
                   use_synthetic: bool = False) -> None:
    cfg = SPLIT_DEFS[split]
    n_samples = cfg["n_samples"]
    offset = cfg["offset"]

    out_path = os.path.join(out_dir, split, f"cacti_challenge_{split}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Get sequence list
    try:
        all_seqs = list_davis_sequences(davis_dir)
    except FileNotFoundError:
        if use_synthetic:
            all_seqs = []
        else:
            raise

    print(f"Generating CACTI {split}: {n_samples} samples → {out_path}")
    if all_seqs:
        print(f"  Using {len(all_seqs)} DAVIS sequences (offset={offset})")

    with h5py.File(out_path, "w") as hf:
        hf.attrs.update({
            "variant": VARIANT, "version": VERSION, "tier": split,
            "n_frames": N_FRAMES, "img_size": IMG_SIZE,
            "dataset": "davis2017",
            "citation": (
                "Pont-Tuset, J. et al. (2016) The 2016 DAVIS Challenge on Video "
                "Object Segmentation, arXiv 1611.00476"
            ),
        })

        for i in range(n_samples):
            seq_idx = (offset + i) % max(len(all_seqs), 1)
            grp = hf.create_group(f"sample_{i:02d}")

            # ── Load video frames ──────────────────────────────────────────
            try:
                if not all_seqs:
                    raise FileNotFoundError("no sequences")
                seq_name = all_seqs[seq_idx]
                cube = load_davis_sequence(davis_dir, seq_name)
            except Exception:
                if use_synthetic:
                    # Random smooth video as fallback
                    cube = np.zeros((IMG_SIZE, IMG_SIZE, N_FRAMES), dtype=np.float64)
                    base = RNG.random((IMG_SIZE, IMG_SIZE)).astype(np.float32)
                    for t in range(N_FRAMES):
                        drift = RNG.normal(0, 0.02, base.shape).astype(np.float32)
                        base = np.clip(base + drift, 0, 1)
                        cube[:, :, t] = base
                    seq_name = f"synthetic_{i:03d}"
                else:
                    raise

            # ── Mismatch for hidden ────────────────────────────────────────
            if split == "hidden":
                mask_dx = float(RNG.uniform(-2, 2))
                mask_dy = float(RNG.uniform(-2, 2))
                mask_angle = float(RNG.uniform(-1, 1))
                mask_blur = float(RNG.uniform(0, 0.5))
                brightness = float(RNG.uniform(-0.05, 0.05))
                contrast = float(RNG.uniform(0.9, 1.1))
                duty_cycle = float(RNG.uniform(0.4, 0.6))
                cube = augment_cube(cube, RNG,
                                    brightness_shift=brightness,
                                    contrast_scale=contrast)
            else:
                mask_dx = mask_dy = mask_angle = mask_blur = 0.0
                duty_cycle = 0.5
                brightness = contrast = 0.0

            # ── Generate mask & compress ───────────────────────────────────
            mask = generate_mask(IMG_SIZE, N_FRAMES, RNG,
                                 dx=mask_dx, dy=mask_dy,
                                 angle_deg=mask_angle,
                                 blur_sigma=mask_blur,
                                 duty_cycle=duty_cycle)
            y = cacti_forward(cube, mask)

            grp.create_dataset("x_true",  data=cube, compression="gzip")
            grp.create_dataset("y",       data=y,    compression="gzip")
            grp.create_dataset("H_ideal", data=mask, compression="gzip")
            grp.attrs["mismatch_params"] = json.dumps({
                "mask_dx": mask_dx, "mask_dy": mask_dy,
                "mask_angle_deg": mask_angle, "mask_blur_sigma": mask_blur,
                "duty_cycle": duty_cycle, "brightness": brightness,
                "contrast": contrast, "sequence": seq_name,
            })

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  [{i+1}/{n_samples}] {seq_name}")

    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_dir", type=str, default=None,
                        help="Path to DAVIS root (contains JPEGImages/)")
    parser.add_argument("--out_dir", type=str, default="datasets/benchmark/cacti")
    parser.add_argument("--split", type=str, default="all",
                        choices=["public", "dev", "hidden", "all"])
    parser.add_argument("--synthetic_fallback", action="store_true",
                        help="Generate synthetic video cubes if DAVIS not available")
    args = parser.parse_args()

    if args.davis_dir is None and not args.synthetic_fallback:
        print("No --davis_dir. Pass --synthetic_fallback to use random video cubes.")
        args.synthetic_fallback = True

    splits = ["public", "dev", "hidden"] if args.split == "all" else [args.split]
    for split in splits:
        generate_split(args.davis_dir or "", args.out_dir, split,
                       use_synthetic=args.synthetic_fallback)

    print("\nDone. Upload with:")
    print("  gsutil -m cp -r datasets/benchmark/cacti/ "
          "gs://pwm-benchmark-datasets/challenge-data/v1.0/")


if __name__ == "__main__":
    main()
