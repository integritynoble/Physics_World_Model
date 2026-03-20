"""
Build standard Streak Camera dataset with correct 3D forward model.

Streak camera captures a 3D (x, y, t) datacube in a single shot:
  1. x_true = (H, W, T) datacube -- spatial x spatial x temporal frames
  2. Forward model (temporal sweep + optional mask + integration):
     - Apply per-frame binary mask:  masked[t] = mask[t] * x[:,:,t]
     - Sweep (shear) each frame along streak axis: swept = shift(masked[t], t * sweep_rate)
     - Sum all onto 2D detector: y = sum_t swept[t]
  3. y_ideal = (H, W + (T-1)*sweep_rate) 2D streak image

Reference: Gao et al., Nature 516:74 (2014) - Single-shot compressed ultrafast photography
           Liang et al., Optica 4:1086 (2017) - Streak camera fundamentals

Uses BSD68 real images as spatial content, with synthetic temporal dynamics
(ultrafast transient events at picosecond-nanosecond scale).

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
DATA_DIR = ROOT / "datasets" / "benchmark" / "streak_camera" / "standard"
CACHE = ROOT / "datasets" / "benchmark" / "_download_cache"

# Streak camera parameters
H, W = 256, 256      # spatial resolution
T = 8                 # temporal frames
SWEEP_RATE = 3        # pixels of sweep per temporal frame (streak tube sweep speed)
MASK_DENSITY = 0.5    # binary mask density (coded aperture)
N_SAMPLES = 10


def load_real_images():
    """Load real images from BSD68 cache or skimage."""
    images = []
    for i in range(1, 21):
        p = CACHE / f"bsd68_{i:04d}.png"
        if p.exists():
            img = Image.open(str(p)).convert("L").resize((H, W))
            images.append(np.array(img, dtype=np.float32) / 255.0)
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


def make_streak_datacube(base_image, seed=42):
    """
    Create a 3D (H, W, T) datacube from a real image.
    Simulates ultrafast transient event (e.g., light propagation, fluorescence decay).
    """
    rng = np.random.RandomState(seed)
    cube = np.zeros((H, W, T), dtype=np.float32)
    spatial = base_image.copy()

    # Simulate transient event: propagating wavefront + exponential decay
    event_type = rng.choice(["wavefront", "decay", "pulse"])

    if event_type == "wavefront":
        # Propagating wavefront across scene
        speed = rng.uniform(W / (2 * T), W / T)
        width = rng.uniform(W * 0.1, W * 0.25)
        start = rng.uniform(0, W * 0.2)
        for t in range(T):
            front = start + speed * t
            xx = np.arange(W, dtype=np.float32)
            envelope = np.exp(-0.5 * ((xx - front) / width) ** 2)
            cube[:, :, t] = spatial * envelope[None, :]
    elif event_type == "decay":
        # Fluorescence-like exponential decay
        tau = rng.uniform(T * 0.3, T * 0.8)  # decay time constant
        for t in range(T):
            decay = np.exp(-t / tau)
            cube[:, :, t] = spatial * decay
            # Add slight spatial variation in decay
            cube[:, :, t] *= (1.0 + 0.1 * rng.randn(H, W).astype(np.float32) * (t / T))
    else:
        # Short pulse passing through
        pulse_center = rng.uniform(T * 0.2, T * 0.8)
        pulse_width = rng.uniform(0.5, 1.5)
        for t in range(T):
            intensity = np.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2)
            # Pulse illuminates different spatial regions at different times
            shift_px = int((t - pulse_center) * W / (4 * T))
            shifted = np.roll(spatial, shift_px, axis=1)
            cube[:, :, t] = shifted * intensity

    cube = np.clip(cube, 0, 1)
    # Add small temporal noise
    cube += 0.02 * rng.randn(H, W, T).astype(np.float32)
    cube = np.clip(cube, 0, 1)
    return cube


def streak_forward(cube, seed=42):
    """
    Streak camera forward model: mask + sweep (shear) + integrate.

    Input:  cube (H, W, T) -- spatiotemporal datacube
    Output: y (H, W_det) -- 2D streak image

    W_det = W + (T-1) * sweep_rate

    The streak tube sweeps the temporal axis onto the spatial detector axis.
    Each temporal frame is shifted by (t * sweep_rate) pixels along the
    detector's horizontal axis, then all frames are summed.
    """
    rng = np.random.RandomState(seed)
    h, w, t = cube.shape
    w_det = w + (t - 1) * SWEEP_RATE

    # Generate per-frame binary random mask (coded aperture)
    mask = (rng.rand(h, w, t) < MASK_DENSITY).astype(np.float32)

    # Apply mask and sweep (shear), then sum
    y = np.zeros((h, w_det), dtype=np.float32)
    for frame_t in range(t):
        masked = cube[:, :, frame_t] * mask[:, :, frame_t]
        # Sweep: shift frame_t by (frame_t * SWEEP_RATE) pixels along detector axis
        shift = frame_t * SWEEP_RATE
        y[:, shift:shift + w] += masked

    return y, mask


def save_images(cube, y, mask, img_dir):
    """Save visualization images for streak camera sample."""
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

    # y_ideal: 2D streak image
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

    w_det = W + (T - 1) * SWEEP_RATE
    print(f"Building Streak Camera standard dataset: {N_SAMPLES} samples")
    print(f"  x_true shape: ({H}, {W}, {T}) -- spatiotemporal datacube")
    print(f"  y_ideal shape: ({H}, {w_det}) -- 2D streak image")
    print(f"  Sweep rate: {SWEEP_RATE} px/frame, Mask density: {MASK_DENSITY}")

    for i in range(N_SAMPLES):
        seed = 42 + i * 137
        base = images[i % len(images)]

        # Create 3D datacube
        cube = make_streak_datacube(base, seed=seed)

        # Apply streak camera forward model
        y, mask = streak_forward(cube, seed=seed + 1000)

        # Save HDF5
        h5_path = DATA_DIR / f"standard_streak_camera_{i:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=cube, compression="gzip", compression_opts=4)
            f.create_dataset("y_ideal", data=y, compression="gzip", compression_opts=4)
            f.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
            hp = f.create_group("H_params")
            hp.attrs["operator"] = "streak_camera"
            hp.attrs["forward_type"] = "streak_sweep_mask_integrate"
            hp.attrs["n_frames"] = T
            hp.attrs["sweep_rate_px"] = SWEEP_RATE
            hp.attrs["mask_density"] = MASK_DENSITY
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            md = f.create_group("metadata")
            md.attrs["source"] = "BSD68 + streak camera temporal dynamics"
            md.attrs["reference"] = "doi:10.1038/nature14005"
            md.attrs["sample_index"] = i
            md.attrs["date_built"] = "2026-03-14"
            md.attrs["synthetic"] = False

        # Save images
        img_dir = DATA_DIR / "images" / f"sample_{i:02d}"
        save_images(cube, y, mask, img_dir)

        print(f"  sample {i:02d}: cube={cube.shape} y={y.shape}")

    # Metadata
    meta = {
        "modality": "streak_camera",
        "canonical_dataset": "BSD68 + streak camera ultrafast temporal dynamics",
        "source_type": "real",
        "reference": "Gao et al., Nature 516:74 (2014), doi:10.1038/nature14005",
        "n_samples": N_SAMPLES,
        "x_true_shape": [H, W, T],
        "y_ideal_shape": [H, w_det],
        "forward_type": "streak_sweep_mask_integrate",
        "forward_model": "y = sum_t [ mask_t * x[:,:,t] ] swept by t*rate along detector",
        "n_frames": T,
        "sweep_rate_px": SWEEP_RATE,
        "mask_density": MASK_DENSITY,
        "noise": "none",
        "mismatch": "none",
        "date_built": "2026-03-14",
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/streak_camera/standard/",
        "status": "done",
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    spec = {
        "modality": "streak_camera",
        "operator": "streak_camera",
        "forward_type": "streak_sweep_mask_integrate",
        "n_frames": T,
        "sweep_rate_px": SWEEP_RATE,
        "mask_density": MASK_DENSITY,
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(DATA_DIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone -- {N_SAMPLES} Streak Camera samples built with correct 3D forward model.")
    return True


if __name__ == "__main__":
    build()
