"""
Generate benchmark HDF5 datasets for fpm (Fourier Ptychographic Microscopy).

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Output: datasets/benchmark/fpm/{fpm_challenge_{tier}.h5, images/}

Forward model:
- 9 LED illumination angles in a 3x3 grid
- Each LED shifts the sample spectrum by (kx, ky)
- 4x downsampling in Fourier domain → 64x64 low-res images
"""

import os
import json
import numpy as np
import h5py
from scipy.ndimage import zoom
from PIL import Image

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/fpm"
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "images"), exist_ok=True)

TIERS = {
    "public": 12,
    "dev": 20,
    "hidden": 20,
}

RNG_SEED_BASE = 137

# 9 LED positions in a 3x3 grid
# kx, ky in units of 1/pixel (frequency shift)
# LED spacing ~ 10 degrees, converted to Fourier shift
# For a 256-pixel array, a 10-degree tilt ~ 0.04 cycles/pixel shift
def make_led_positions(led_spacing_deg):
    """
    3x3 grid of LED positions.
    Returns (9, 2) array of (kx, ky) in 1/pixel units.
    """
    # Convert angle to spatial frequency shift: kx = sin(theta) / wavelength
    # Approximate: kx [1/px] = led_spacing_deg / 180 * pi * NA_scale
    # Use simplified model: shift = spacing_deg * scale_factor
    scale = led_spacing_deg / (180.0 / np.pi) * 0.8  # empirical factor
    positions = []
    for iy in range(-1, 2):
        for ix in range(-1, 2):
            kx = ix * scale
            ky = iy * scale
            positions.append([kx, ky])
    return np.array(positions, dtype=np.float32)  # (9,2)


def fpm_forward(x_true, led_positions, downsample=4):
    """
    FPM forward model: generate N low-res images.
    x_true: (256,256) high-res complex or real image
    led_positions: (N,2) kx,ky shifts in 1/pixel

    For each LED:
      1. Shift spectrum by (kx,ky) -> select a 64x64 patch
      2. IFFT the patch -> intensity |...|^2
    Returns y: (N,64,64) float32
    """
    H, W = x_true.shape  # 256,256
    N_led = led_positions.shape[0]
    h_out = H // downsample  # 64
    w_out = W // downsample  # 64

    # Compute full-res FFT (centered)
    F = np.fft.fft2(x_true.astype(np.complex64))
    F_shift = np.fft.fftshift(F)  # (256,256) centered spectrum

    y = np.zeros((N_led, h_out, w_out), dtype=np.float32)

    cx, cy = H // 2, W // 2  # center of spectrum

    for n, (kx, ky) in enumerate(led_positions):
        # LED shift in pixels in Fourier domain
        # kx [1/px] * H [px] = pixel shift in frequency domain
        dx = int(round(kx * H))
        dy = int(round(ky * W))

        # Extract 64x64 patch centered at (cy+dy, cx+dx)
        r0 = cy + dy - h_out // 2
        c0 = cx + dx - w_out // 2
        r1 = r0 + h_out
        c1 = c0 + w_out

        # Pad-safe extraction
        patch = np.zeros((h_out, w_out), dtype=np.complex64)
        # Clamp to valid range
        pr0 = max(0, -r0)
        pc0 = max(0, -c0)
        pr1 = h_out - max(0, r1 - H)
        pc1 = w_out - max(0, c1 - W)
        sr0 = max(0, r0)
        sc0 = max(0, c0)
        sr1 = sr0 + (pr1 - pr0)
        sc1 = sc0 + (pc1 - pc0)

        if pr1 > pr0 and pc1 > pc0:
            patch[pr0:pr1, pc0:pc1] = F_shift[sr0:sr1, sc0:sc1]

        # IFFT the patch -> field
        patch_unshift = np.fft.ifftshift(patch)
        field = np.fft.ifft2(patch_unshift)
        intensity = np.abs(field)**2
        y[n] = intensity.astype(np.float32)

    return y


def make_sample(rng, idx_global):
    """Generate one FPM sample."""
    # --- spec params ---
    NA = float(rng.uniform(0.1, 0.4))
    pixel_size_um = float(rng.uniform(1.0, 6.5))
    led_spacing_deg = float(rng.uniform(5.0, 15.0))

    # --- LED positions (9,2) ---
    led_positions = make_led_positions(led_spacing_deg)

    # --- x_true: (256,256) float32 high-res sample ---
    # Rich detail: biological cell-like texture
    x = np.zeros((256, 256), dtype=np.float32)
    # Background structure
    n_cells = rng.integers(3, 8)
    for _ in range(n_cells):
        cx = rng.uniform(30, 226)
        cy = rng.uniform(30, 226)
        r = rng.uniform(15, 50)
        sx = r * rng.uniform(0.6, 1.4)
        sy = r * rng.uniform(0.6, 1.4)
        amp = rng.uniform(0.3, 1.0)
        yy, xx = np.mgrid[0:256, 0:256]
        blob = amp * np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
        x += blob

    # Add fine detail (high-frequency content important for FPM)
    noise_detail = rng.standard_normal((256, 256)).astype(np.float32) * 0.05
    x = x + noise_detail
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    x = x.astype(np.float32)

    # --- Forward model ---
    y = fpm_forward(x, led_positions, downsample=4)  # (9, 64, 64)

    # Add small Gaussian noise to measurements
    noise_level = rng.uniform(0.001, 0.01) * y.max()
    y = y + rng.standard_normal(y.shape).astype(np.float32) * noise_level
    y = np.clip(y, 0, None).astype(np.float32)

    # --- H_ideal: (9,2) LED positions ---
    H_ideal = led_positions  # (9,2)

    # --- Baseline: bilinear upsample of central LED image y[4] -> (256,256) ---
    central_img = y[4]  # 64x64
    zoom_factor = 256.0 / 64.0  # = 4.0
    recon = zoom(central_img, zoom_factor, order=1).astype(np.float32)  # (256,256) bilinear
    if recon.max() > 0:
        recon = recon / recon.max()
    recon = np.clip(recon, 0, 1).astype(np.float32)

    spec = {
        "NA": round(NA, 4),
        "pixel_size_um": round(pixel_size_um, 3),
        "led_spacing_deg": round(led_spacing_deg, 3),
    }
    true_spec = {
        "n_leds": 9,
        "downsample_factor": 4,
        "x_true_max": float(x.max()),
        "x_true_mean": float(x.mean()),
        "y_max": float(y.max()),
        "y_shape": list(y.shape),
        "forward_model": "FPM_intensity",
        "noise_model": "Gaussian_additive",
    }

    return x, y, H_ideal, recon, spec, true_spec


def save_png(arr, path):
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(path)


def generate_tier(tier_name, n_samples, seed_offset):
    h5_path = os.path.join(ROOT, f"fpm_challenge_{tier_name}.h5")
    img_dir = os.path.join(ROOT, "images", tier_name)
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n=== Generating fpm {tier_name} ({n_samples} samples) ===")

    specs_all = []
    true_specs_all = []

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = "fpm"
        hf.attrs["tier"] = tier_name
        hf.attrs["n_samples"] = n_samples
        hf.attrs["description"] = "FPM (Fourier Ptychographic Microscopy) benchmark dataset"

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
                # Save central LED image (normalized)
                y_central = y[4]
                y_norm = y_central / (y_central.max() + 1e-12)
                save_png(y_norm, os.path.join(img_dir, f"sample_{i:02d}_y_central_led.png"))
                save_png(recon, os.path.join(img_dir, f"sample_{i:02d}_recon_baseline.png"))
                # Save all 9 LED images as a 3x3 grid
                grid = np.zeros((3*64, 3*64), dtype=np.float32)
                for led_i in range(9):
                    row = led_i // 3
                    col = led_i % 3
                    img_led = y[led_i]
                    img_led_norm = img_led / (img_led.max() + 1e-12)
                    grid[row*64:(row+1)*64, col*64:(col+1)*64] = img_led_norm
                save_png(grid, os.path.join(img_dir, f"sample_{i:02d}_y_9leds_grid.png"))

            specs_all.append(spec)
            true_specs_all.append(true_spec)

            if i == 0:
                print(f"  sample_00 shapes: x_true={x_true.shape}, y={y.shape}, H_ideal={H_ideal.shape}, recon={recon.shape}")
                print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]")
                print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"  H_ideal (LED positions):\n{H_ideal}")
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
    print("\n=== fpm generation complete ===")
