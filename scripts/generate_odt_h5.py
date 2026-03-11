"""
Generate benchmark HDF5 datasets for odt (Optical Diffraction Tomography).

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Output: datasets/benchmark/odt/{odt_challenge_{tier}.h5, images/}

Forward model:
- 36 projection angles: 0, 5, 10, ..., 175 degrees
- Born approximation: projection = sum along rotated axis
- y: (36, 256) sinogram-style measurements
- H_ideal: (36,) angles in degrees
- Baseline: FBP reconstruction from sinogram
"""

import os
import json
import numpy as np
import h5py
from scipy.ndimage import rotate
from PIL import Image

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/odt"
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "images"), exist_ok=True)

TIERS = {
    "public": 12,
    "dev": 20,
    "hidden": 20,
}

RNG_SEED_BASE = 271

ANGLES = np.linspace(0, 175, 36, dtype=np.float32)  # 36 angles


def radon_transform(x, angles):
    """
    Compute sinogram via Radon transform using scipy.ndimage.rotate.
    x: (256,256) float32 refractive index distribution
    angles: (36,) degrees
    Returns: (36, 256) float32 sinogram
    """
    N = x.shape[0]  # 256
    sinogram = np.zeros((len(angles), N), dtype=np.float32)
    for i, angle in enumerate(angles):
        # Rotate the image and sum along axis=0 (vertical projection)
        rotated = rotate(x, angle, reshape=False, order=1, mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0).astype(np.float32)
    return sinogram


def fbp_reconstruction(sinogram, angles):
    """
    Filtered Back Projection from sinogram.
    sinogram: (36, 256) float32
    angles: (36,) degrees
    Returns: (256,256) float32 reconstruction
    """
    N_angles, N_det = sinogram.shape
    N = N_det  # output size = 256

    # Ram-Lak filter in frequency domain
    freqs = np.fft.fftfreq(N, d=1.0).astype(np.float64)
    ramp = np.abs(freqs).astype(np.float64)

    # Filter each projection
    filtered_sino = np.zeros_like(sinogram, dtype=np.float64)
    for i in range(N_angles):
        proj_fft = np.fft.fft(sinogram[i].astype(np.float64))
        filtered_sino[i] = np.real(np.fft.ifft(proj_fft * ramp))

    # Back-project
    recon = np.zeros((N, N), dtype=np.float64)
    yy, xx = np.mgrid[-N//2:N//2, -N//2:N//2].astype(np.float64)

    for i, angle in enumerate(angles):
        theta = np.deg2rad(float(angle))
        # Projection coordinate: t = x*cos(theta) + y*sin(theta)
        t = xx * np.cos(theta) + yy * np.sin(theta)
        # Map t to detector pixel index
        t_idx = t + N // 2
        t_idx_int = np.round(t_idx).astype(np.int32)
        # Clip to valid range
        t_idx_clamp = np.clip(t_idx_int, 0, N - 1)
        recon += filtered_sino[i][t_idx_clamp]

    recon = recon * (np.pi / (2 * N_angles))
    return recon.astype(np.float32)


def make_sample(rng, idx_global):
    """Generate one ODT sample."""
    # --- spec params ---
    RI_contrast = float(rng.uniform(0.001, 0.05))
    wavelength_nm = float(rng.uniform(488.0, 633.0))
    NA_illum = float(rng.uniform(0.1, 0.4))

    # --- x_true: (256,256) float32 refractive index map, range [0,1] ---
    # Model: cell-like structures (spheroids + organelles)
    x = np.zeros((256, 256), dtype=np.float32)

    # Cell body (large ellipse)
    n_cells = rng.integers(1, 5)
    for _ in range(n_cells):
        cx = rng.uniform(60, 196)
        cy = rng.uniform(60, 196)
        rx = rng.uniform(20, 60)
        ry = rng.uniform(20, 60)
        amp_cell = rng.uniform(0.3, 0.7)
        yy, xx = np.mgrid[0:256, 0:256]
        mask = ((xx - cx)**2 / rx**2 + (yy - cy)**2 / ry**2) <= 1.0
        x[mask] += amp_cell

        # Nucleus (smaller ellipse inside)
        nx = cx + rng.uniform(-10, 10)
        ny = cy + rng.uniform(-10, 10)
        nrx = rx * rng.uniform(0.3, 0.5)
        nry = ry * rng.uniform(0.3, 0.5)
        amp_nuc = rng.uniform(0.2, 0.5)
        n_mask = ((xx - nx)**2 / nrx**2 + (yy - ny)**2 / nry**2) <= 1.0
        x[n_mask] += amp_nuc

        # Organelles (small dots)
        n_org = rng.integers(2, 8)
        for _ in range(n_org):
            ox = cx + rng.uniform(-rx*0.7, rx*0.7)
            oy = cy + rng.uniform(-ry*0.7, ry*0.7)
            org_r = rng.uniform(2, 6)
            amp_org = rng.uniform(0.05, 0.2)
            o_mask = ((xx - ox)**2 + (yy - oy)**2) <= org_r**2
            x[o_mask] += amp_org

    # Normalize to [0,1]
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    x = x.astype(np.float32)

    # --- H_ideal: angles (36,) ---
    H_ideal = ANGLES.copy()

    # --- Forward model: Radon transform (Born approximation) ---
    sinogram = radon_transform(x, ANGLES)  # (36, 256)

    # Add small Gaussian noise
    noise_level = rng.uniform(0.001, 0.005) * sinogram.max()
    sinogram = sinogram + rng.standard_normal(sinogram.shape).astype(np.float32) * noise_level
    sinogram = np.clip(sinogram, 0, None).astype(np.float32)
    y = sinogram  # (36, 256)

    # --- Baseline: FBP reconstruction ---
    recon = fbp_reconstruction(y, ANGLES)  # (256, 256)
    recon = np.clip(recon, 0, None).astype(np.float32)
    if recon.max() > 0:
        recon = recon / recon.max()
    recon = np.clip(recon, 0, 1).astype(np.float32)

    spec = {
        "RI_contrast": round(RI_contrast, 5),
        "wavelength_nm": round(wavelength_nm, 1),
        "NA_illum": round(NA_illum, 4),
    }
    true_spec = {
        "n_angles": 36,
        "angle_range_deg": [0.0, 175.0],
        "x_true_max": float(x.max()),
        "x_true_mean": float(x.mean()),
        "sinogram_max": float(sinogram.max()),
        "forward_model": "Radon_Born_approximation",
        "noise_model": "Gaussian_additive",
    }

    return x, y, H_ideal, recon, spec, true_spec


def save_png(arr, path):
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(path)


def save_sinogram_png(sino, path):
    """Save sinogram (36,256) as PNG."""
    sino_norm = sino / (sino.max() + 1e-12)
    sino_u8 = (sino_norm * 255).astype(np.uint8)
    Image.fromarray(sino_u8, mode="L").save(path)


def generate_tier(tier_name, n_samples, seed_offset):
    h5_path = os.path.join(ROOT, f"odt_challenge_{tier_name}.h5")
    img_dir = os.path.join(ROOT, "images", tier_name)
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n=== Generating odt {tier_name} ({n_samples} samples) ===")

    specs_all = []
    true_specs_all = []

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = "odt"
        hf.attrs["tier"] = tier_name
        hf.attrs["n_samples"] = n_samples
        hf.attrs["description"] = "ODT (Optical Diffraction Tomography) benchmark dataset"

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

            if i < 6:
                save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
                save_sinogram_png(y, os.path.join(img_dir, f"sample_{i:02d}_sinogram.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon_fbp.png"))

            specs_all.append(spec)
            true_specs_all.append(true_spec)

            if i == 0:
                print(f"  sample_00 shapes: x_true={x_true.shape}, y={y.shape}, H_ideal={H_ideal.shape}, recon={recon.shape}")
                print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]")
                print(f"  y (sinogram) range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"  H_ideal (angles, first 5): {H_ideal[:5]}")
                print(f"  recon range: [{recon.min():.4f}, {recon.max():.4f}]")
                print(f"  spec: {spec}")

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
    print("\n=== odt generation complete ===")
