#!/usr/bin/env python3
"""
Generate CASSI benchmark H5 files from the CAVE multispectral dataset.

H5 schema (matches existing sd_cassi format):
  sample_NN/
    x_true:       (256, 256, 28) float32  — spectral cube
    y:            (256, 310)     float32  — compressed CASSI measurement
    y_ideal:      (256, 310)     float32  — noise-free measurement
    H_ideal:      (256, 256)     float32  — diagonal mask (1-row shifted)
    H_continuous: (256, 256)     float32  — smooth mask version
  attrs: tier, variant, version, n_bands, step

DOWNLOAD (free, requires request):
  CAVE Multispectral Image Database:
    http://www.cs.columbia.edu/CAVE/database/multispectral/
  31 scenes × 512×512 × 31 bands (400-700nm, 10nm steps)

  After download, place each scene as:
    /path/to/cave/{scene_name}/{scene_name}_01.png  (band 1)
    ...
    /path/to/cave/{scene_name}/{scene_name}_31.png  (band 31)

  OR use KAIST (alternative):
    https://vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset.html
    30 scenes × 2704×3376 × 31 bands

Usage:
  python scripts/generate_cassi_h5.py \\
      --cave_dir /path/to/cave \\
      --out_dir datasets/benchmark/cassi \\
      [--split public|dev|hidden]

Split design:
  public  — scenes 0-7  (8 scenes, nominal mask, no mismatch)
  dev     — scenes 0-19 (20 scenes, nominal mask)
  hidden  — scenes 0-19 with transforms: random crop, rotation ±10°,
             mask shift (dx~[-2,2], dy~[-2,2]), dispersion slope ±5%
"""

import argparse
import json
import os

import h5py
import numpy as np

try:
    from PIL import Image
    from skimage.transform import rotate, resize
except ImportError:
    raise ImportError("pip install Pillow scikit-image")

VARIANT = "cassi"
VERSION = "1.0"
IMG_SIZE = 256
N_BANDS = 28       # use 28 of the 31 CAVE bands (matching existing CASSI format)
DISPERSION_STEP = 2  # pixels per band shift (SD-CASSI)
N_MEAS_COLS = IMG_SIZE + (N_BANDS - 1) * DISPERSION_STEP  # = 256 + 54 = 310

SPLIT_DEFS = {
    "public": {"scene_range": (0, 8),   "n_crops": 1},
    "dev":    {"scene_range": (0, 20),  "n_crops": 1},
    "hidden": {"scene_range": (0, 20),  "n_crops": 1},
}

RNG = np.random.default_rng(42)

CAVE_SCENES = [
    "balloons", "beads", "cd", "chart_and_stuffed_toy", "clay",
    "cloth", "egyptian_statue", "face", "fake_and_real_food",
    "fake_and_real_lemon_slices", "fake_and_real_lemons",
    "fake_and_real_peppers", "feathers", "flowers", "glass_tiles",
    "jelly_beans", "oil_painting", "paints", "photo_and_face",
    "pompoms", "real_and_fake_apples", "real_and_fake_peppers",
    "sponges", "stuffed_toys", "superballs", "thread_spools",
    "토마토", "watercolors", "yarn",
]


# ── CAVE loader ───────────────────────────────────────────────────────────────
def load_cave_scene(cave_dir: str, scene_name: str,
                    target_size: int = IMG_SIZE,
                    n_bands: int = N_BANDS) -> np.ndarray:
    """
    Load a CAVE scene as a (H, W, n_bands) float32 cube [0, 1].
    Handles both PNG (per-band) and NPY (pre-converted) formats.
    """
    scene_dir = os.path.join(cave_dir, scene_name)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(
            f"CAVE scene not found: {scene_dir}\n"
            "Download from http://www.cs.columbia.edu/CAVE/database/multispectral/"
        )

    # Check for pre-saved numpy array
    npy_path = os.path.join(scene_dir, "cube.npy")
    if os.path.exists(npy_path):
        cube = np.load(npy_path).astype(np.float32)
        if cube.shape[2] > n_bands:
            cube = cube[:, :, :n_bands]
    else:
        # Load PNG bands
        bands = []
        for b in range(1, n_bands + 1):
            # CAVE naming: scene_01.png ... scene_31.png
            band_path = os.path.join(scene_dir, f"{scene_name}_{b:02d}.png")
            if not os.path.exists(band_path):
                # try without leading zeros
                band_path = os.path.join(scene_dir, f"{scene_name}_{b}.png")
            img = np.array(Image.open(band_path)).astype(np.float32)
            if img.ndim == 3:
                img = img.mean(axis=2)  # convert RGB to gray
            bands.append(img)
        cube = np.stack(bands, axis=2)  # (H, W, n_bands)

    # Normalize to [0, 1]
    cube = cube / (cube.max() + 1e-8)

    # Resize to target
    if cube.shape[0] != target_size or cube.shape[1] != target_size:
        cube_resized = np.zeros((target_size, target_size, n_bands), np.float32)
        for b in range(n_bands):
            cube_resized[:, :, b] = resize(
                cube[:, :, b], (target_size, target_size), anti_aliasing=True
            ).astype(np.float32)
        cube = cube_resized

    return cube.astype(np.float32)


def random_augment_cube(cube: np.ndarray, rng) -> np.ndarray:
    """Apply random rotation + crop for dev/hidden diversity."""
    angle = rng.uniform(-10, 10)
    H, W, C = cube.shape
    aug = np.zeros_like(cube)
    for c in range(C):
        rotated = rotate(cube[:, :, c], angle, mode="reflect")
        # Random crop to IMG_SIZE
        if rotated.shape[0] > IMG_SIZE:
            dy = rng.integers(0, rotated.shape[0] - IMG_SIZE + 1)
            dx = rng.integers(0, rotated.shape[1] - IMG_SIZE + 1)
            rotated = rotated[dy:dy + IMG_SIZE, dx:dx + IMG_SIZE]
        aug[:, :, c] = rotated
    return aug.astype(np.float32)


# ── CASSI forward model ───────────────────────────────────────────────────────
def generate_cassi_mask(size: int, rng,
                        dx: float = 0.0, dy: float = 0.0,
                        blur_sigma: float = 0.0) -> tuple:
    """
    Generate binary coded aperture mask.
    Returns (H_ideal, H_continuous): ideal binary and smoothed versions.
    """
    H_ideal = (rng.random((size, size)) > 0.5).astype(np.float32)

    # Apply sub-pixel shift via interpolation
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        from scipy.ndimage import shift
        H_ideal = shift(H_ideal, [dy, dx], mode="reflect")
        H_ideal = (H_ideal > 0.5).astype(np.float32)

    if blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        H_continuous = gaussian_filter(H_ideal.astype(np.float32), sigma=blur_sigma)
    else:
        H_continuous = H_ideal.copy()

    return H_ideal, H_continuous


def cassi_forward(cube: np.ndarray, mask: np.ndarray,
                  step: int = DISPERSION_STEP,
                  dispersion_slope: float = 1.0,
                  noise_std: float = 0.01) -> tuple:
    """
    SD-CASSI forward model:
      y[i, j] = sum_b mask[i, j-b*step] * x[i, j-b*step, b]

    Returns (y_ideal, y_noisy): (H, W + (B-1)*step) float32
    """
    H, W, B = cube.shape
    W_meas = W + (B - 1) * step

    y_ideal = np.zeros((H, W_meas), dtype=np.float32)
    mask_disp = np.zeros((H, W_meas), dtype=np.float32)

    for b in range(B):
        shift_b = int(b * step * dispersion_slope)
        for j in range(W):
            j_out = j + shift_b
            if 0 <= j_out < W_meas:
                y_ideal[:, j_out] += mask[:, j] * cube[:, j, b]
                mask_disp[:, j_out] += mask[:, j]

    # Normalize by mask overlap
    denom = np.maximum(mask_disp, 1.0)
    y_normalized = y_ideal  # keep raw sum for reconstruction challenge

    # Add Gaussian noise
    y_noisy = y_normalized + rng_global.normal(0, noise_std, y_normalized.shape).astype(np.float32)
    y_noisy = np.clip(y_noisy, 0, None)

    return y_normalized, y_noisy.astype(np.float32)


rng_global = RNG  # module-level rng for cassi_forward


# ── Main generator ────────────────────────────────────────────────────────────
def generate_split(cave_dir: str, out_dir: str, split: str,
                   use_synthetic: bool = False) -> None:
    cfg = SPLIT_DEFS[split]
    s_start, s_end = cfg["scene_range"]
    scenes = CAVE_SCENES[s_start:s_end]
    n_samples = len(scenes)

    out_path = os.path.join(out_dir, split, f"cassi_challenge_{split}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Generating CASSI {split}: {n_samples} scenes → {out_path}")

    with h5py.File(out_path, "w") as hf:
        hf.attrs.update({
            "variant": VARIANT, "version": VERSION, "tier": split,
            "n_bands": N_BANDS, "step": DISPERSION_STEP,
            "dataset": "cave_multispectral",
            "citation": "Yasuma, F. et al. (2010) Generalized assorted pixel camera, IEEE TIP 19(9)",
        })

        for i, scene_name in enumerate(scenes):
            grp = hf.create_group(f"sample_{i:02d}")

            # ── Load spectral cube ─────────────────────────────────────────
            try:
                cube = load_cave_scene(cave_dir, scene_name)
            except Exception:
                if use_synthetic:
                    cube = np.random.rand(IMG_SIZE, IMG_SIZE, N_BANDS).astype(np.float32)
                else:
                    raise

            # ── Augment for hidden split ───────────────────────────────────
            if split == "hidden":
                cube = random_augment_cube(cube, RNG)
                mask_dx = float(RNG.uniform(-2, 2))
                mask_dy = float(RNG.uniform(-2, 2))
                mask_rotation = float(RNG.uniform(-0.5, 0.5))
                disp_slope = float(RNG.uniform(0.95, 1.05))
                noise_std = float(RNG.uniform(0.005, 0.02))
            else:
                mask_dx, mask_dy, mask_rotation = 0.0, 0.0, 0.0
                disp_slope = 1.0
                noise_std = 0.01

            # ── Generate mask ──────────────────────────────────────────────
            H_ideal, H_continuous = generate_cassi_mask(
                IMG_SIZE, RNG, dx=mask_dx, dy=mask_dy
            )

            # ── Forward model ──────────────────────────────────────────────
            y_ideal, y_noisy = cassi_forward(
                cube, H_ideal,
                step=DISPERSION_STEP,
                dispersion_slope=disp_slope,
                noise_std=noise_std,
            )

            grp.create_dataset("x_true",       data=cube,        compression="gzip")
            grp.create_dataset("y",            data=y_noisy,     compression="gzip")
            grp.create_dataset("y_ideal",      data=y_ideal,     compression="gzip")
            grp.create_dataset("H_ideal",      data=H_ideal,     compression="gzip")
            grp.create_dataset("H_continuous", data=H_continuous, compression="gzip")
            grp.attrs["mismatch_params"] = json.dumps({
                "mask_dx": mask_dx, "mask_dy": mask_dy,
                "mask_rotation": mask_rotation,
                "dispersion_slope": disp_slope,
                "noise_std": noise_std,
            })

            print(f"  [{i+1}/{n_samples}] {scene_name} disp={disp_slope:.3f}")

    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cave_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="datasets/benchmark/cassi")
    parser.add_argument("--split", type=str, default="all",
                        choices=["public", "dev", "hidden", "all"])
    parser.add_argument("--synthetic_fallback", action="store_true")
    args = parser.parse_args()

    if args.cave_dir is None and not args.synthetic_fallback:
        print("No --cave_dir. Using synthetic data. Pass --synthetic_fallback to confirm.")
        args.synthetic_fallback = True

    splits = ["public", "dev", "hidden"] if args.split == "all" else [args.split]
    for split in splits:
        generate_split(args.cave_dir or "", args.out_dir, split,
                       use_synthetic=args.synthetic_fallback)

    print("\nDone. Upload with:")
    print("  gsutil -m cp -r datasets/benchmark/cassi/ "
          "gs://pwm-benchmark-datasets/challenge-data/v1.0/")


if __name__ == "__main__":
    main()
