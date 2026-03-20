"""
Generate visualization images for ALL standard datasets.
- 2D data: save as grayscale/color PNG
- 3D cubes: save each slice as PNG
- 1D signals: save as line-plot PNG
- Complex (Fourier): save magnitude as PNG

Output: datasets/benchmark/{mod}/standard/images/sample_{NN}/
  x_true.png (or x_true_slice_00.png ... for 3D)
  y_ideal.png (or y_ideal_slice_00.png ... for 3D)

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import os
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
DATA_DIR = ROOT / "datasets" / "benchmark"

# Use matplotlib only for 1D plots
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass


def to_uint8(arr):
    """Normalize array to [0, 255] uint8."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    normed = (arr - mn) / (mx - mn)
    return (normed * 255).clip(0, 255).astype(np.uint8)


def save_2d(arr, path):
    """Save 2D array as grayscale PNG."""
    img = Image.fromarray(to_uint8(arr), mode="L")
    img.save(str(path))


def save_rgb(arr, path):
    """Save HxWx3 array as RGB PNG."""
    img = Image.fromarray(to_uint8(arr), mode="RGB")
    img.save(str(path))


def save_1d_plot(arr, path, title=""):
    """Save 1D signal as line plot PNG."""
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        ax.plot(arr, linewidth=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("index", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
    else:
        # Fallback: render as a thin image (height=64)
        N = len(arr)
        H = 64
        arr_norm = to_uint8(arr)
        img = np.zeros((H, N), dtype=np.uint8)
        for i, v in enumerate(arr_norm):
            row = H - 1 - int(v / 255.0 * (H - 1))
            img[row:, i] = v
        Image.fromarray(img, mode="L").save(str(path))


def save_complex(arr, path):
    """Save complex-valued 2D array (e.g., Fourier) as log-magnitude PNG."""
    if arr.ndim == 3 and arr.shape[-1] == 2:
        # Real + imag stacked
        magnitude = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    elif np.iscomplexobj(arr):
        magnitude = np.abs(arr)
    else:
        magnitude = np.abs(arr)
    log_mag = np.log1p(magnitude)
    save_2d(log_mag, path)


def generate_images_for_sample(h5_path, img_dir):
    """Generate images for one HDF5 sample."""
    img_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_path), "r") as f:
        x_true = f["x_true"][:]
        y_ideal = f["y_ideal"][:]

    # Save x_true images
    save_data(x_true, img_dir, "x_true")

    # Save y_ideal images
    save_data(y_ideal, img_dir, "y_ideal")


def save_data(arr, img_dir, prefix):
    """Save array as image(s) depending on dimensionality."""
    arr = np.nan_to_num(arr, nan=0.0)

    if arr.ndim == 1:
        # 1D signal → line plot
        save_1d_plot(arr, img_dir / f"{prefix}.png", title=prefix)

    elif arr.ndim == 2:
        # 2D image → grayscale PNG
        save_2d(arr, img_dir / f"{prefix}.png")

    elif arr.ndim == 3:
        if arr.shape[-1] == 2:
            # Complex Fourier (H, W, 2) → magnitude
            save_complex(arr, img_dir / f"{prefix}.png")
        elif arr.shape[-1] == 3:
            # RGB image (H, W, 3)
            save_rgb(arr, img_dir / f"{prefix}.png")
        elif arr.shape[-1] <= 4:
            # Small channel count → save each channel
            for c in range(arr.shape[-1]):
                save_2d(arr[:, :, c], img_dir / f"{prefix}_ch{c:02d}.png")
            # Also save first channel as main image
            save_2d(arr[:, :, 0], img_dir / f"{prefix}.png")
        elif arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            # 3D volume (D, H, W) — D is smallest → slice along D
            D = arr.shape[0]
            # Save representative slices (up to 8)
            n_slices = min(D, 8)
            indices = np.linspace(0, D - 1, n_slices, dtype=int)
            for idx in indices:
                save_2d(arr[idx], img_dir / f"{prefix}_slice_{idx:02d}.png")
            # Save middle slice as main image
            save_2d(arr[D // 2], img_dir / f"{prefix}.png")
        else:
            # Spectral cube (H, W, C) — C is last
            C = arr.shape[-1]
            # Save representative bands
            n_bands = min(C, 8)
            indices = np.linspace(0, C - 1, n_bands, dtype=int)
            for idx in indices:
                save_2d(arr[:, :, idx], img_dir / f"{prefix}_band_{idx:02d}.png")
            # Save RGB composite (3 bands) as main image
            if C >= 3:
                r_idx = C // 4
                g_idx = C // 2
                b_idx = 3 * C // 4
                rgb = np.stack([arr[:, :, r_idx], arr[:, :, g_idx], arr[:, :, b_idx]], axis=-1)
                save_rgb(rgb, img_dir / f"{prefix}.png")
            else:
                save_2d(arr[:, :, 0], img_dir / f"{prefix}.png")

    elif arr.ndim == 4:
        # 4D: (n_angles, D, W) or (D, H, W, C) → save middle slices
        mid = arr.shape[0] // 2
        save_data(arr[mid], img_dir, prefix)
        # Also save a few other slices
        for idx in [0, arr.shape[0] // 4, 3 * arr.shape[0] // 4]:
            if idx != mid:
                save_data(arr[idx], img_dir, f"{prefix}_dim0_{idx:02d}")

    else:
        # Higher dimensional: flatten to 2D
        flat = arr.reshape(arr.shape[0], -1)
        save_2d(flat[:256, :256], img_dir / f"{prefix}.png")


def process_modality(modality):
    """Generate images for all samples in a modality."""
    std_dir = DATA_DIR / modality / "standard"
    if not std_dir.exists():
        return 0

    h5_files = sorted(std_dir.glob("*.h5"))
    if not h5_files:
        return 0

    img_base = std_dir / "images"
    count = 0

    for h5_path in h5_files:
        # Extract sample index from filename: standard_modality_NN.h5
        stem = h5_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            sample_id = f"sample_{parts[1]}"
        else:
            sample_id = f"sample_{count:02d}"

        img_dir = img_base / sample_id

        # Skip if images already exist
        if img_dir.exists() and any(img_dir.glob("*.png")):
            count += 1
            continue

        try:
            generate_images_for_sample(h5_path, img_dir)
            count += 1
        except Exception as e:
            print(f"    ERROR {modality}/{h5_path.name}: {e}")

    return count


def main():
    print("Generating images for all standard datasets...")
    print()

    modalities = sorted([
        d for d in os.listdir(str(DATA_DIR))
        if (DATA_DIR / d / "standard").is_dir()
        and not d.startswith("_")
        and not d.endswith(".md")
    ])

    total_mods = len(modalities)
    total_samples = 0
    done_mods = 0

    for idx, mod in enumerate(modalities, 1):
        n = process_modality(mod)
        if n > 0:
            done_mods += 1
            total_samples += n
            # Print progress every 10 modalities
            if idx % 10 == 0 or idx == total_mods:
                print(f"  [{idx}/{total_mods}] {mod}: {n} samples imaged  (total: {total_samples})")

    print(f"\nDone: {done_mods} modalities, {total_samples} samples imaged.")


if __name__ == "__main__":
    main()
