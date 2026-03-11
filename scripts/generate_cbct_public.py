#!/usr/bin/env python3
"""Generate CBCT public tier dataset (12 samples).

Cone-beam CT forward model:
- Shepp-Logan phantoms with anatomical variation
- 256 angles over 360 degrees
- Radon-based forward projection
- Poisson noise + beam hardening mismatch
"""

import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "cbct" / "public"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 12
N_ANGLES = 256
N_DETECTORS = 363
IMG_SIZE = 256
I0 = 8000.0  # photon count


def shepp_logan(n=256):
    """Generate Shepp-Logan phantom with variation."""
    phantom = np.zeros((n, n), dtype=np.float32)
    cx, cy = n // 2, n // 2

    ellipses = [
        (1.0, 0.69, 0.92, 0, 0, 0),
        (-0.8, 0.6624, 0.8740, 0, -0.0184, 0),
        (-0.2, 0.1100, 0.3100, 0.22, 0, -18),
        (-0.2, 0.1600, 0.4100, -0.22, 0, 18),
        (0.1, 0.2100, 0.2500, 0, 0.35, 0),
        (0.1, 0.0460, 0.0460, 0, 0.1, 0),
        (0.1, 0.0460, 0.0460, 0, -0.1, 0),
        (-0.02, 0.0460, 0.0230, -0.08, -0.605, 0),
        (-0.02, 0.0230, 0.0230, 0, -0.606, 0),
        (-0.02, 0.0230, 0.0460, 0.06, -0.605, 0),
    ]

    y_grid, x_grid = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]

    for (val, a, b, x0, y0, theta_deg) in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = cos_t * (x_grid - x0) + sin_t * (y_grid - y0)
        yr = -sin_t * (x_grid - x0) + cos_t * (y_grid - y0)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        phantom[mask] += val

    return np.clip(phantom, 0, None)


def radon_transform(image, angles_deg):
    """Simple parallel-beam Radon transform."""
    from scipy.ndimage import map_coordinates

    n = image.shape[0]
    n_det = int(np.ceil(n * np.sqrt(2)))
    if n_det % 2 == 0:
        n_det += 1

    sinogram = np.zeros((len(angles_deg), n_det), dtype=np.float32)
    center = n / 2.0
    det_center = n_det / 2.0

    # Detector coordinates
    d = np.arange(n_det) - det_center

    for i, angle in enumerate(angles_deg):
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # For each detector bin, integrate along perpendicular
        # Use line integral via rotation
        for j, dj in enumerate(d):
            # Line through detector position dj at angle theta
            # Points along the perpendicular direction
            t = np.linspace(-n / 2, n / 2, n * 2)
            xs = dj * cos_t - t * sin_t + center
            ys = dj * sin_t + t * cos_t + center

            valid = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
            if valid.any():
                vals = map_coordinates(image, [ys[valid], xs[valid]], order=1)
                sinogram[i, j] = vals.mean() * len(t) / n

    return sinogram


def fast_radon(image, angles_deg, n_det=None):
    """Fast Radon using scipy.ndimage rotation + sum."""
    from scipy.ndimage import rotate

    n = image.shape[0]
    # Pad to diagonal to avoid clipping during rotation
    diag = int(np.ceil(n * np.sqrt(2))) + 4
    pad = (diag - n) // 2 + 2
    img_padded = np.pad(image, pad, mode='constant')
    pn = img_padded.shape[0]

    if n_det is None:
        n_det = diag
    # Ensure n_det <= pn
    n_det = min(n_det, pn)

    sinogram = np.zeros((len(angles_deg), n_det), dtype=np.float32)
    center = pn // 2
    det_start = center - n_det // 2

    for i, angle in enumerate(angles_deg):
        rotated = rotate(img_padded, -angle, reshape=False, order=1)
        projection = rotated.sum(axis=0)
        chunk = projection[det_start:det_start + n_det]
        # Resize to n_det if needed
        if len(chunk) != n_det:
            from scipy.ndimage import zoom
            chunk = zoom(chunk, n_det / len(chunk), order=1)
        sinogram[i] = chunk[:n_det]

    return sinogram


def add_cbct_noise(sinogram_ideal, i0=8000.0, mismatch_seed=0):
    """Add Poisson noise + beam hardening mismatch."""
    rng = np.random.RandomState(mismatch_seed)

    # Beer-Lambert: I = I0 * exp(-sino)
    intensity = i0 * np.exp(-sinogram_ideal.astype(np.float64))
    intensity = np.maximum(intensity, 1.0)

    # Poisson noise
    noisy = rng.poisson(intensity).astype(np.float64)
    noisy = np.maximum(noisy, 1.0)

    # Log-linearize back
    y = -np.log(noisy / i0).astype(np.float32)

    # Beam hardening (polynomial correction)
    bh = rng.uniform(0.0, 0.10)
    y = y + bh * y ** 2

    return y


def generate_sample(seed, n_angles=256, n_det=363, img_size=256):
    """Generate one CBCT sample."""
    rng = np.random.RandomState(seed)

    # Phantom with random variation
    phantom = shepp_logan(img_size)

    # Random scaling/contrast
    scale = rng.uniform(0.7, 1.2)
    phantom = phantom * scale

    # Uniform angles 0-360
    angles_deg = np.linspace(0, 360, n_angles, endpoint=False).astype(np.float32)

    # Forward projection (fast Radon)
    sino_ideal = fast_radon(phantom, angles_deg, n_det)
    sino_ideal = np.clip(sino_ideal, 0, None).astype(np.float32)

    # Noisy measurement
    y = add_cbct_noise(sino_ideal, i0=I0, mismatch_seed=seed + 100)

    return {
        "x_true": phantom.astype(np.float32),
        "sinogram_ideal": sino_ideal,
        "y": y,
        "H_ideal": angles_deg,
    }


def generate_true_spec(seed):
    rng = np.random.RandomState(seed + 100)
    return {
        "source_offset_x": float(rng.uniform(-2.0, 2.0)),
        "source_offset_z": float(rng.uniform(-1.5, 1.5)),
        "detector_tilt": float(rng.uniform(-0.5, 0.5)),
        "detector_shift_u": float(rng.uniform(-3.0, 3.0)),
        "beam_hardening": float(rng.uniform(0.0, 0.10)),
        "scatter_fraction": float(rng.uniform(0.0, 0.08)),
        "photon_count_I0": I0,
    }


def save_png(arr, path):
    """Save 2D array as PNG."""
    try:
        from PIL import Image
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(arr_norm).save(path)
    except Exception:
        pass


def main():
    print("Generating CBCT public tier dataset...")
    print(f"  Samples: {N_SAMPLES}, Angles: {N_ANGLES}, Detectors: {N_DETECTORS}")

    h5_path = OUT_DIR / "cbct_challenge_public.h5"
    images_dir = OUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    true_specs = {}

    with h5py.File(h5_path, "w") as hf:
        for i in range(N_SAMPLES):
            seed = i * 42 + 7
            print(f"  sample_{i:02d}...", end=" ", flush=True)

            sample = generate_sample(seed)
            ts = generate_true_spec(seed)
            true_specs[f"sample_{i:02d}"] = ts

            grp = hf.create_group(f"sample_{i:02d}")
            for key, val in sample.items():
                grp.create_dataset(key, data=val, compression="gzip")

            # Save images
            save_png(sample["x_true"], images_dir / f"sample_{i:02d}_groundtruth.png")
            save_png(sample["y"], images_dir / f"sample_{i:02d}_measurement.png")
            print("OK")

    # Save true_spec.json
    ts_path = OUT_DIR / "true_spec.json"
    with open(ts_path, "w") as f:
        json.dump(true_specs, f, indent=2)

    print(f"\nSaved: {h5_path}")
    print(f"Saved: {ts_path}")
    print(f"Images: {images_dir}")
    print(f"Dataset size: {h5_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
