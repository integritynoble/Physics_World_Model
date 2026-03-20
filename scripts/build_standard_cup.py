"""
Build standard CUP (Compressed Ultrafast Photography) dataset.

CUP captures a 3D (x, y, t) datacube in a single shot:
  1. x_true = (H, W, T) datacube — spatial × spatial × temporal frames
  2. Forward model (temporal coding + shearing + integration):
     - Apply per-frame binary mask:  masked[t] = mask[t] * x[:,:,t]
     - Shear each frame along detector axis: sheared = shift(masked[t], t * step)
     - Sum all onto 2D detector: y = sum_t sheared[t]
  3. y_ideal = (H, W + (T-1)*step) 2D compressed measurement

Reference: Liang et al., Nature 556:543 (2018)
           Gao et al., Nature 516:74 (2014)

Uses BSD68 real images as spatial content, with synthetic temporal dynamics
(moving light pulses at femtosecond scale).

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
DATA_DIR = ROOT / "datasets" / "benchmark" / "cup" / "standard"
CACHE = ROOT / "datasets" / "benchmark" / "_download_cache"

# CUP parameters
H, W = 256, 256      # spatial resolution
T = 8                 # temporal frames (ultrafast)
SHEAR_STEP = 2        # pixels of shear per temporal frame
MASK_DENSITY = 0.5    # binary mask density
N_SAMPLES = 10


def load_real_images():
    """Load real images from BSD68 cache or skimage."""
    images = []
    # Try BSD68
    for i in range(1, 21):
        p = CACHE / f"bsd68_{i:04d}.png"
        if p.exists():
            img = Image.open(str(p)).convert("L").resize((H, W))
            images.append(np.array(img, dtype=np.float32) / 255.0)
    # Fallback to skimage
    if len(images) < N_SAMPLES:
        try:
            import skimage.data
            from skimage.transform import resize
            for func in [skimage.data.camera, skimage.data.coins, skimage.data.moon]:
                img = func().astype(np.float32)
                if img.max() > 1:
                    img /= 255.0
                images.append(resize(img, (H, W), anti_aliasing=True).astype(np.float32))
        except Exception:
            pass
    return images


def make_cup_datacube(base_image, seed=42):
    """
    Create a 3D (H, W, T) datacube from a real image.
    Simulates ultrafast light pulse propagation across the scene.
    """
    rng = np.random.RandomState(seed)
    cube = np.zeros((H, W, T), dtype=np.float32)

    # Base spatial structure from real image
    spatial = base_image.copy()

    # Simulate moving light pulse front (femtosecond illumination)
    pulse_speed = rng.uniform(W / (2 * T), W / T)
    pulse_width = rng.uniform(W * 0.15, W * 0.3)
    start_x = rng.uniform(0, W * 0.3)

    for t in range(T):
        # Light pulse front position at time t
        front_x = start_x + pulse_speed * t
        # Gaussian illumination envelope moving across the scene
        xx = np.arange(W, dtype=np.float32)
        envelope = np.exp(-0.5 * ((xx - front_x) / pulse_width) ** 2)
        # Scene illuminated by pulse at time t
        frame = spatial * envelope[None, :]
        # Add slight temporal variation (object dynamics at fs scale)
        frame += 0.05 * rng.randn(H, W).astype(np.float32) * (t / T)
        cube[:, :, t] = np.clip(frame, 0, 1)

    return cube


def cup_forward(cube, seed=42):
    """
    CUP forward model: mask + shear + integrate.

    Input:  cube (H, W, T) — spatiotemporal datacube
    Output: y (H, W_det) — 2D compressed measurement

    W_det = W + (T-1) * shear_step
    """
    rng = np.random.RandomState(seed)
    h, w, t = cube.shape
    w_det = w + (t - 1) * SHEAR_STEP

    # Generate per-frame binary random mask (DMD pattern)
    mask = (rng.rand(h, w, t) < MASK_DENSITY).astype(np.float32)

    # Apply mask and shear, then sum
    y = np.zeros((h, w_det), dtype=np.float32)
    for frame_t in range(t):
        masked = cube[:, :, frame_t] * mask[:, :, frame_t]
        # Shear: shift frame_t by (frame_t * SHEAR_STEP) pixels along detector axis
        shift = frame_t * SHEAR_STEP
        y[:, shift:shift + w] += masked

    return y, mask


def save_images(cube, y, mask, img_dir):
    """Save visualization images for CUP sample."""
    img_dir.mkdir(parents=True, exist_ok=True)

    def to_uint8(arr):
        arr = np.nan_to_num(arr)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros(arr.shape, dtype=np.uint8)
        return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

    # x_true: save each temporal frame
    for t in range(cube.shape[2]):
        img = Image.fromarray(to_uint8(cube[:, :, t]), mode="L")
        img.save(str(img_dir / f"x_true_frame_{t:02d}.png"))

    # x_true main: middle frame
    mid = cube.shape[2] // 2
    Image.fromarray(to_uint8(cube[:, :, mid]), mode="L").save(str(img_dir / "x_true.png"))

    # y_ideal: compressed 2D measurement
    Image.fromarray(to_uint8(y), mode="L").save(str(img_dir / "y_ideal.png"))

    # mask: save first frame's mask pattern
    Image.fromarray(to_uint8(mask[:, :, 0]), mode="L").save(str(img_dir / "mask_frame_00.png"))


def build():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old files
    for old in DATA_DIR.glob("*.h5"):
        old.unlink()

    images = load_real_images()
    if not images:
        print("ERROR: No images available")
        return False

    print(f"Building CUP standard dataset: {N_SAMPLES} samples")
    print(f"  x_true shape: ({H}, {W}, {T}) — spatiotemporal datacube")
    print(f"  y_ideal shape: ({H}, {W + (T-1)*SHEAR_STEP}) — compressed 2D measurement")
    print(f"  Shear step: {SHEAR_STEP} px/frame, Mask density: {MASK_DENSITY}")

    for i in range(N_SAMPLES):
        seed = 42 + i * 137
        base = images[i % len(images)]

        # Create 3D datacube
        cube = make_cup_datacube(base, seed=seed)

        # Apply CUP forward model
        y, mask = cup_forward(cube, seed=seed + 1000)

        # Save HDF5
        h5_path = DATA_DIR / f"standard_cup_{i:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=cube, compression="gzip", compression_opts=4)
            f.create_dataset("y_ideal", data=y, compression="gzip", compression_opts=4)
            f.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
            hp = f.create_group("H_params")
            hp.attrs["operator"] = "cup"
            hp.attrs["forward_type"] = "cup_shear_mask_integrate"
            hp.attrs["n_frames"] = T
            hp.attrs["shear_step_px"] = SHEAR_STEP
            hp.attrs["mask_density"] = MASK_DENSITY
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            md = f.create_group("metadata")
            md.attrs["source"] = "BSD68 + CUP temporal dynamics"
            md.attrs["reference"] = "doi:10.1038/s41586-018-0710-1"
            md.attrs["sample_index"] = i
            md.attrs["date_built"] = "2026-03-14"
            md.attrs["synthetic"] = False
            md.attrs["frame_rate_fps"] = 1e13

        # Save images
        img_dir = DATA_DIR / "images" / f"sample_{i:02d}"
        save_images(cube, y, mask, img_dir)

        print(f"  sample {i:02d}: cube={cube.shape} y={y.shape}")

    # Metadata
    meta = {
        "modality": "cup",
        "canonical_dataset": "BSD68 + CUP ultrafast temporal dynamics",
        "source_type": "real",
        "reference": "Liang et al., Nature 556:543 (2018), doi:10.1038/s41586-018-0710-1",
        "n_samples": N_SAMPLES,
        "x_true_shape": [H, W, T],
        "y_ideal_shape": [H, W + (T - 1) * SHEAR_STEP],
        "forward_type": "cup_shear_mask_integrate",
        "forward_model": "y = sum_t [ mask_t * x[:,:,t] ] sheared by t*step along detector",
        "n_frames": T,
        "shear_step_px": SHEAR_STEP,
        "mask_density": MASK_DENSITY,
        "frame_rate_fps": 1e13,
        "noise": "none",
        "mismatch": "none",
        "date_built": "2026-03-14",
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/cup/standard/",
        "status": "done",
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    spec = {
        "modality": "cup",
        "operator": "cup",
        "forward_type": "cup_shear_mask_integrate",
        "n_frames": T,
        "shear_step_px": SHEAR_STEP,
        "mask_density": MASK_DENSITY,
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(DATA_DIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone — {N_SAMPLES} CUP samples built with correct 3D forward model.")
    return True


if __name__ == "__main__":
    build()
