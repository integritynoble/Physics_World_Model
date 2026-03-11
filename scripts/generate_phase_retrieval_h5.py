"""
Generate benchmark HDF5 datasets for phase_retrieval (CDI - Coherent Diffractive Imaging).

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Output: datasets/benchmark/phase_retrieval/{phase_retrieval_challenge_{tier}.h5, images/}
"""

import os
import json
import numpy as np
import h5py
from PIL import Image

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/phase_retrieval"
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "images"), exist_ok=True)

TIERS = {
    "public": 12,
    "dev": 20,
    "hidden": 20,
}

RNG_SEED_BASE = 42

# Support mask: 1 inside 180x180 central region, 0 outside
def make_support_mask(size=256, support=180):
    H = np.zeros((size, size), dtype=np.float32)
    margin = (size - support) // 2
    H[margin:margin+support, margin:margin+support] = 1.0
    return H

H_IDEAL = make_support_mask()

def make_sample(rng, idx_global):
    """
    Generate one CDI sample.
    x_true: (256,256) float32 real-valued object [0,1]
    y: (256,256) float32 diffraction intensity
    H_ideal: (256,256) float32 support mask
    reconstruction_baseline: from sqrt(y) -> IFFT -> |real| -> clip
    spec.json params
    """
    # --- spec params ---
    oversampling_ratio = float(rng.uniform(2.0, 8.0))
    noise_photons = float(10 ** rng.uniform(3.0, 6.0))   # 1e3 to 1e6
    beam_size_px = float(rng.uniform(50.0, 200.0))

    # --- x_true: smooth random object with some structure ---
    # Use a sum of Gaussians + random blobs to look like a realistic sample
    x = np.zeros((256, 256), dtype=np.float32)
    n_blobs = rng.integers(3, 10)
    for _ in range(n_blobs):
        cx = rng.uniform(38, 218)
        cy = rng.uniform(38, 218)
        sx = rng.uniform(8, 40)
        sy = rng.uniform(8, 40)
        amp = rng.uniform(0.2, 1.0)
        yy, xx = np.mgrid[0:256, 0:256]
        blob = amp * np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
        x += blob
    # Clip and normalize to [0,1]
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    x = x.astype(np.float32)

    # --- Forward model: y = |FFT(x)|^2 + Poisson noise ---
    F = np.fft.fft2(x)
    intensity = np.abs(F)**2  # (256,256)

    # Poisson noise: scale to photon counts, add noise
    scale = noise_photons / (intensity.sum() + 1e-12)
    intensity_scaled = intensity * scale
    noisy_counts = rng.poisson(np.maximum(intensity_scaled, 0)).astype(np.float32)
    # Rescale back
    y = noisy_counts / (scale + 1e-30)
    # Shift to center (fftshift for display)
    y = np.fft.fftshift(y).astype(np.float32)

    # --- H_ideal: support mask ---
    H = H_IDEAL.copy()

    # --- Baseline reconstruction: sqrt(y) -> IFFT -> |real| -> clip [0,1] ---
    amp_pattern = np.sqrt(np.maximum(y, 0))
    # ifftshift before IFFT to undo the display shift
    amp_unshifted = np.fft.ifftshift(amp_pattern)
    recon = np.fft.ifft2(amp_unshifted)
    recon_abs = np.abs(recon.real).astype(np.float32)
    if recon_abs.max() > 0:
        recon_abs = recon_abs / recon_abs.max()
    recon_abs = np.clip(recon_abs, 0, 1).astype(np.float32)

    spec = {
        "oversampling_ratio": round(oversampling_ratio, 3),
        "noise_photons": round(noise_photons, 1),
        "beam_size_px": round(beam_size_px, 2),
    }
    true_spec = {
        "n_blobs": int(n_blobs),
        "object_max": float(x.max()),
        "object_mean": float(x.mean()),
        "forward_model": "CDI_intensity",
        "noise_model": "Poisson",
        "intensity_total_photons": round(float(intensity_scaled.sum()), 1),
    }

    return x, y, H, recon_abs, spec, true_spec


def save_png(arr, path, cmap="gray"):
    """Save a 2D float32 array [0,1] as a grayscale PNG."""
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(path)


def generate_tier(tier_name, n_samples, seed_offset):
    h5_path = os.path.join(ROOT, f"phase_retrieval_challenge_{tier_name}.h5")
    img_dir = os.path.join(ROOT, "images", tier_name)
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n=== Generating phase_retrieval {tier_name} ({n_samples} samples) ===")

    specs_all = []
    true_specs_all = []

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = "phase_retrieval"
        hf.attrs["tier"] = tier_name
        hf.attrs["n_samples"] = n_samples
        hf.attrs["description"] = "CDI (Coherent Diffractive Imaging) benchmark dataset"

        for i in range(n_samples):
            rng = np.random.default_rng(RNG_SEED_BASE + seed_offset + i)
            x_true, y, H_ideal, recon, spec, true_spec = make_sample(rng, i)

            grp_name = f"sample_{i:02d}"
            grp = hf.create_group(grp_name)
            grp.create_dataset("x_true", data=x_true, dtype=np.float32, compression="gzip")
            grp.create_dataset("y", data=y, dtype=np.float32, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, dtype=np.float32, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype=np.float32, compression="gzip")
            grp.attrs["spec"] = json.dumps(spec)
            grp.attrs["true_spec"] = json.dumps(true_spec)

            # Save PNGs (only first 6 per tier to keep disk reasonable)
            if i < 6:
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                # Log-scale y for visualization
                y_log = np.log1p(y)
                y_log_norm = y_log / (y_log.max() + 1e-12)
                save_png(y_log_norm, os.path.join(img_dir, f"sample_{i:02d}_y_log.png"))
                save_png(H_ideal, os.path.join(img_dir, f"sample_{i:02d}_H_ideal.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon_baseline.png"))

            specs_all.append(spec)
            true_specs_all.append(true_spec)

            # Verification print for first sample
            if i == 0:
                print(f"  sample_00 shapes: x_true={x_true.shape}, y={y.shape}, H={H_ideal.shape}, recon={recon.shape}")
                print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]")
                print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"  H_ideal range: [{H_ideal.min():.4f}, {H_ideal.max():.4f}]")
                print(f"  recon range: [{recon.min():.4f}, {recon.max():.4f}]")
                print(f"  spec: {spec}")

    # Save aggregated spec files
    with open(os.path.join(ROOT, f"spec_{tier_name}.json"), "w") as f:
        json.dump(specs_all, f, indent=2)
    with open(os.path.join(ROOT, f"true_spec_{tier_name}.json"), "w") as f:
        json.dump(true_specs_all, f, indent=2)

    print(f"  Saved: {h5_path}")
    print(f"  H5 file size: {os.path.getsize(h5_path)/1e6:.2f} MB")


if __name__ == "__main__":
    seed_offsets = {"public": 0, "dev": 100, "hidden": 200}
    for tier, n in TIERS.items():
        generate_tier(tier, n, seed_offsets[tier])
    print("\n=== phase_retrieval generation complete ===")
