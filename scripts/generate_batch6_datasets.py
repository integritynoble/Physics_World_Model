#!/usr/bin/env python3
"""
Generate benchmark datasets for batch-6 modalities:
  ivus, lattice_lightsheet (skip - exists), libs, light_field, lucky_imaging,
  machine_vision, magnetic_particle, maldi_msi, matrix, mfm

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Seeds: 6000+i*17 (public), 8800+i*17 (dev), 9900+i*17 (hidden)
"""

import json
import os

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import wiener

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model"
OUT_BASE = os.path.join(ROOT, "datasets/benchmark")

TIERS = {
    "public":  {"n": 12, "seed_base": 6000},
    "dev":     {"n": 20, "seed_base": 8800},
    "hidden":  {"n": 20, "seed_base": 9900},
}
SEED_STEP = 17


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def make_dirs(modality, tier):
    d = os.path.join(OUT_BASE, modality, tier)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    return d


def save_png(arr, path):
    """Save a 2-D float32 array as a greyscale PNG (auto-scaled)."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    img = Image.fromarray((a * 255).astype(np.uint8))
    img.save(path)


def write_spec(d, spec, true_spec):
    with open(os.path.join(d, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(d, "true_spec.json"), "w") as f:
        json.dump(true_spec, f, indent=2)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def gaussian_psf_2d(size, sigma):
    """Return a normalised 2-D Gaussian PSF of given size and sigma."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def convolve2d(image, psf):
    """Convolve image with PSF using FFT (same shape)."""
    from numpy.fft import fft2, ifft2, fftshift
    F = fft2(image)
    H = fft2(fftshift(psf), s=image.shape)
    return np.real(ifft2(F * H)).astype(np.float32)


def wiener_deconv(y, psf, noise_power=0.01):
    """Wiener deconvolution in frequency domain."""
    from numpy.fft import fft2, ifft2, fftshift
    Y = fft2(y)
    H = fft2(fftshift(psf), s=y.shape)
    H_conj = np.conj(H)
    denom = np.abs(H)**2 + noise_power
    X_hat = (H_conj / denom) * Y
    return np.real(ifft2(X_hat)).astype(np.float32)


def rl_deconv(y, psf, n_iter=5):
    """Richardson-Lucy deconvolution (n_iter iterations)."""
    from scipy.signal import fftconvolve
    psf_flip = psf[::-1, ::-1]
    x = np.ones_like(y, dtype=np.float32) * y.mean()
    eps = 1e-6
    for _ in range(n_iter):
        conv = fftconvolve(x, psf, mode="same") + eps
        ratio = y / conv
        x = x * fftconvolve(ratio, psf_flip, mode="same")
        x = np.clip(x, 0, None)
    return x.astype(np.float32)


def radon_project(image, n_angles=90):
    """Simple parallel-beam Radon transform using line integrals via rotation."""
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    N = image.shape[0]
    sino = np.zeros((n_angles, N), dtype=np.float32)
    for i, angle in enumerate(angles):
        rotated = ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0.0)
        sino[i] = rotated.sum(axis=0)
    return sino


def fbp_reconstruct(sino, image_size=128):
    """Filtered back-projection (ramp filter in Fourier domain)."""
    n_angles, n_det = sino.shape
    # Ramp filter
    freqs = np.fft.rfftfreq(n_det)
    ramp = np.abs(freqs)
    filtered = np.zeros_like(sino)
    for i in range(n_angles):
        F = np.fft.rfft(sino[i])
        filtered[i] = np.fft.irfft(F * ramp, n=n_det)

    # Back-project
    recon = np.zeros((image_size, image_size), dtype=np.float32)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    cx, cy = image_size // 2, image_size // 2
    xs = np.arange(image_size) - cx
    ys = np.arange(image_size) - cy
    XX, YY = np.meshgrid(xs, ys)
    for i, angle in enumerate(angles):
        t = XX * np.cos(angle) + YY * np.sin(angle)
        t_idx = t + n_det // 2
        t_idx = np.clip(t_idx, 0, n_det - 1).astype(int)
        recon += filtered[i][t_idx]
    recon *= np.pi / (2 * n_angles)
    return recon.astype(np.float32)


def synthetic_vessel(rng, size=128):
    """Generate a synthetic vessel cross-section (ring with lumen)."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2 + rng.integers(-5, 6), size // 2 + rng.integers(-5, 6)
    r_outer = rng.integers(30, 50)
    r_inner = rng.integers(15, r_outer - 5)
    wall_val = rng.uniform(0.6, 1.0)
    lumen_val = rng.uniform(0.05, 0.2)
    plaque_val = rng.uniform(0.8, 1.0)
    for y in range(size):
        for x in range(size):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            if d <= r_inner:
                img[y, x] = lumen_val
            elif d <= r_outer:
                img[y, x] = wall_val
    # Add a plaque region
    angle_p = rng.uniform(0, 2 * np.pi)
    pr = r_inner + (r_outer - r_inner) * 0.4
    px = int(cx + pr * np.cos(angle_p))
    py = int(cy + pr * np.sin(angle_p))
    rp = rng.integers(3, 10)
    for y in range(max(0, py - rp), min(size, py + rp)):
        for x in range(max(0, px - rp), min(size, px + rp)):
            if np.sqrt((x - px)**2 + (y - py)**2) <= rp:
                img[y, x] = plaque_val
    return img


def synthetic_cell_slice(rng, size=128):
    """Generate a synthetic fluorescent cell volume slice."""
    img = np.zeros((size, size), dtype=np.float32)
    # Cell body
    cx, cy = size // 2 + rng.integers(-10, 11), size // 2 + rng.integers(-10, 11)
    r_cell = rng.integers(30, 55)
    for y in range(size):
        for x in range(size):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            if d <= r_cell:
                img[y, x] = rng.uniform(0.1, 0.3) * (1 - d / r_cell)
    # Nucleus
    r_nuc = r_cell // 3
    for y in range(size):
        for x in range(size):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            if d <= r_nuc:
                img[y, x] = rng.uniform(0.7, 1.0) * (1 - d / r_nuc)
    # Organelles
    n_org = rng.integers(3, 8)
    for _ in range(n_org):
        ox = cx + int(rng.uniform(-r_cell * 0.6, r_cell * 0.6))
        oy = cy + int(rng.uniform(-r_cell * 0.6, r_cell * 0.6))
        ro = rng.integers(3, 8)
        val = rng.uniform(0.5, 0.9)
        for y in range(max(0, oy - ro), min(size, oy + ro)):
            for x in range(max(0, ox - ro), min(size, ox + ro)):
                if np.sqrt((x - ox)**2 + (y - oy)**2) <= ro:
                    img[y, x] = val
    return np.clip(img, 0, 1).astype(np.float32)


def synthetic_elemental_map(rng, size=128, n_elements=5):
    """Generate synthetic elemental emission maps (size x size x n_elements)."""
    maps = np.zeros((size, size, n_elements), dtype=np.float32)
    for e in range(n_elements):
        # Each element has a blob pattern
        n_blobs = rng.integers(2, 6)
        for _ in range(n_blobs):
            cx = rng.integers(10, size - 10)
            cy = rng.integers(10, size - 10)
            sigma = rng.uniform(5, 20)
            intensity = rng.uniform(0.3, 1.0)
            ax = np.arange(size)
            xx, yy = np.meshgrid(ax, ax)
            blob = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            maps[:, :, e] += blob.astype(np.float32)
    maps = np.clip(maps, 0, 1).astype(np.float32)
    return maps


def synthetic_astronomical(rng, size=128):
    """Generate a synthetic sharp astronomical image (stars + galaxy)."""
    img = np.zeros((size, size), dtype=np.float32)
    # Background gradient
    ax = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    img += (0.02 * xx + 0.02 * yy).astype(np.float32)
    # Stars
    n_stars = rng.integers(10, 30)
    for _ in range(n_stars):
        sx = rng.integers(0, size)
        sy = rng.integers(0, size)
        val = rng.uniform(0.5, 1.0)
        img[sy, sx] = val
    # Central galaxy blob
    cx, cy = size // 2 + rng.integers(-15, 16), size // 2 + rng.integers(-15, 16)
    sigma_g = rng.uniform(8, 18)
    ax2 = np.arange(size)
    xx2, yy2 = np.meshgrid(ax2, ax2)
    galaxy = 0.8 * np.exp(-((xx2 - cx)**2 + (yy2 - cy)**2) / (2 * sigma_g**2))
    img += galaxy.astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def atmospheric_blur(rng, image, n_frames=10):
    """Generate n_frames short-exposure blurred images (atmospheric turbulence simulation)."""
    frames = np.zeros((n_frames, image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(n_frames):
        sigma_atm = rng.uniform(1.0, 4.0)
        blurred = ndimage.gaussian_filter(image, sigma=sigma_atm).astype(np.float32)
        # Add Poisson-like noise
        noise = rng.normal(0, 0.02, image.shape).astype(np.float32)
        frames[i] = np.clip(blurred + noise, 0, 1)
    return frames


def sharpness_score(img):
    """Laplacian variance as sharpness measure."""
    lap = ndimage.laplace(img.astype(np.float64))
    return lap.var()


def synthetic_segmentation(rng, size=128):
    """Generate a synthetic multi-class segmentation map (0-3 classes)."""
    img = np.zeros((size, size), dtype=np.float32)
    n_objects = rng.integers(3, 8)
    for obj in range(n_objects):
        cls = rng.integers(1, 4) / 3.0  # map 1,2,3 -> 0.33, 0.66, 1.0
        cx = rng.integers(10, size - 10)
        cy = rng.integers(10, size - 10)
        r = rng.integers(8, 25)
        shape = rng.choice(["circle", "rect"])
        for y in range(max(0, cy - r), min(size, cy + r)):
            for x in range(max(0, cx - r), min(size, cx + r)):
                if shape == "circle":
                    if np.sqrt((x - cx)**2 + (y - cy)**2) <= r:
                        img[y, x] = cls
                else:
                    img[y, x] = cls
    return img.astype(np.float32)


def synthetic_mpi_distribution(rng, size=128):
    """Generate synthetic iron-oxide particle distribution for MPI."""
    img = np.zeros((size, size), dtype=np.float32)
    n_clusters = rng.integers(2, 6)
    for _ in range(n_clusters):
        cx = rng.integers(15, size - 15)
        cy = rng.integers(15, size - 15)
        sigma = rng.uniform(5, 15)
        intensity = rng.uniform(0.4, 1.0)
        ax = np.arange(size)
        xx, yy = np.meshgrid(ax, ax)
        blob = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        img += blob.astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def synthetic_mass_spec_image(rng, size=128, n_masses=5):
    """Generate synthetic MALDI MSI ion images (size x size x n_masses)."""
    maps = np.zeros((size, size, n_masses), dtype=np.float32)
    for m in range(n_masses):
        # Tissue region blobs at different locations per mass
        n_regions = rng.integers(1, 4)
        for _ in range(n_regions):
            cx = rng.integers(10, size - 10)
            cy = rng.integers(10, size - 10)
            sigma = rng.uniform(8, 25)
            intensity = rng.uniform(0.2, 1.0)
            ax = np.arange(size)
            xx, yy = np.meshgrid(ax, ax)
            blob = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            maps[:, :, m] += blob.astype(np.float32)
    maps = np.clip(maps, 0, 1).astype(np.float32)
    return maps


def synthetic_matrix(rng, size=64, rank=5):
    """Generate a low-rank matrix as ground truth."""
    U = rng.standard_normal((size, rank)).astype(np.float32)
    V = rng.standard_normal((rank, size)).astype(np.float32)
    M = (U @ V).astype(np.float32)
    # Normalize to [0,1]
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)
    return M


def synthetic_magnetic_force(rng, size=128):
    """Generate synthetic magnetic force microscopy map."""
    img = np.zeros((size, size), dtype=np.float32)
    n_domains = rng.integers(3, 8)
    for _ in range(n_domains):
        cx = rng.integers(10, size - 10)
        cy = rng.integers(10, size - 10)
        sigma = rng.uniform(8, 25)
        sign = rng.choice([-1, 1])
        intensity = rng.uniform(0.3, 1.0)
        ax = np.arange(size)
        xx, yy = np.meshgrid(ax, ax)
        blob = sign * intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        img += blob.astype(np.float32)
    img = img / (np.abs(img).max() + 1e-8)
    return img.astype(np.float32)


def mfm_tip_response(size=128):
    """MFM tip response: derivative of Gaussian (approximating dipole)."""
    psf = gaussian_psf_2d(size, sigma=3.0)
    # Apply Laplacian (second derivative) as tip response
    psf_tip = ndimage.laplace(psf)
    # Normalize
    mx = np.abs(psf_tip).max()
    if mx > 0:
        psf_tip = psf_tip / mx
    return psf_tip.astype(np.float32)


# ---------------------------------------------------------------------------
# IVUS generator
# ---------------------------------------------------------------------------

def generate_ivus(tier, n, seed_base):
    d = make_dirs("ivus", tier)
    h5_path = os.path.join(d, f"ivus_challenge_{tier}.h5")
    psf_sigma = 2.5
    psf = gaussian_psf_2d(31, sigma=psf_sigma)
    psf_full = gaussian_psf_2d(128, sigma=psf_sigma)

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_vessel(rng, 128)
            # y = blur + speckle noise
            y_blurred = ndimage.gaussian_filter(x_true, sigma=psf_sigma)
            speckle = rng.exponential(scale=1.0, size=x_true.shape).astype(np.float32)
            speckle = speckle / speckle.mean()  # mean=1 multiplicative noise
            y = (y_blurred * speckle).astype(np.float32)
            # H_ideal = PSF (stored as full 128x128 for consistent shape)
            H_ideal = psf_full
            # Reconstruction: RL deconvolution 5 iters
            recon = rl_deconv(y, psf_full, n_iter=5)
            recon = np.clip(recon, 0, 1).astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            # PNG preview
            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y, os.path.join(d, "images", f"sample_{i:02d}_y.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "psf_sigma_pixels": {"min": 2.0, "max": 3.0, "unit": "pixels"},
        "speckle_factor": {"min": 0.5, "max": 2.0, "unit": ""},
        "catheter_freq_mhz": {"min": 20, "max": 40, "unit": "MHz"},
    }
    true_spec = {
        "vessel_outer_radius_px": {"min": 30, "max": 50, "unit": "pixels"},
        "vessel_inner_radius_px": {"min": 15, "max": 45, "unit": "pixels"},
        "plaque_intensity": {"min": 0.8, "max": 1.0, "unit": ""},
    }
    write_spec(d, spec, true_spec)
    print(f"  ivus/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# LIBS generator
# ---------------------------------------------------------------------------

def generate_libs(tier, n, seed_base):
    d = make_dirs("libs", tier)
    h5_path = os.path.join(d, f"libs_challenge_{tier}.h5")

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_elemental_map(rng, 128, 5)  # (128,128,5)
            # y: measured spectra with Poisson-like noise + background
            noise = rng.poisson(lam=np.clip(x_true * 100, 0, None)).astype(np.float32) / 100.0
            background = rng.uniform(0.01, 0.05, x_true.shape).astype(np.float32)
            y = (noise + background).astype(np.float32)
            # H_ideal = identity (spatial mapping is direct)
            H_ideal = np.eye(128, dtype=np.float32)
            # Reconstruction: threshold + normalize per channel
            recon = np.zeros_like(y, dtype=np.float32)
            for e in range(5):
                ch = y[:, :, e]
                thresh = ch.mean() + 0.5 * ch.std()
                ch_t = np.clip(ch - thresh, 0, None)
                mx = ch_t.max()
                if mx > 0:
                    ch_t = ch_t / mx
                recon[:, :, e] = ch_t

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            # PNG preview: composite of first 3 channels as RGB
            rgb = np.zeros((128, 128, 3), dtype=np.float32)
            for c in range(3):
                ch = x_true[:, :, c]
                mx = ch.max()
                rgb[:, :, c] = ch / mx if mx > 0 else ch
            img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            img_pil.save(os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            rgb_y = np.zeros((128, 128, 3), dtype=np.float32)
            for c in range(3):
                ch = y[:, :, c]
                mx = ch.max()
                rgb_y[:, :, c] = ch / mx if mx > 0 else ch
            img_pil2 = Image.fromarray((rgb_y * 255).astype(np.uint8))
            img_pil2.save(os.path.join(d, "images", f"sample_{i:02d}_y.png"))

    spec = {
        "laser_wavelength_nm": {"min": 1064, "max": 1064, "unit": "nm"},
        "laser_energy_mj": {"min": 5, "max": 50, "unit": "mJ"},
        "n_elements": {"min": 5, "max": 5, "unit": ""},
    }
    true_spec = {
        "element_concentration": {"min": 0.0, "max": 1.0, "unit": "normalized"},
        "spatial_resolution_um": {"min": 10, "max": 100, "unit": "um"},
    }
    write_spec(d, spec, true_spec)
    print(f"  libs/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# LIGHT FIELD generator
# ---------------------------------------------------------------------------

def generate_light_field(tier, n, seed_base):
    d = make_dirs("light_field", tier)
    h5_path = os.path.join(d, f"light_field_challenge_{tier}.h5")

    n_views = 4  # 4x4 angular views
    view_size = 64

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            # x_true: refocused (128x128) image - synthetic scene
            x_true = synthetic_cell_slice(rng, 128)  # reuse cell pattern as generic scene

            # y: (4,4,64,64) light field array - sub-aperture views with sub-pixel shifts
            y = np.zeros((n_views, n_views, view_size, view_size), dtype=np.float32)
            # Downsample x_true to view_size as center view
            x_small = x_true[::2, ::2]  # 64x64
            disparity_scale = rng.uniform(1.0, 3.0)
            for u in range(n_views):
                for v in range(n_views):
                    # Shift based on angular position
                    shift_x = (u - n_views // 2) * disparity_scale
                    shift_y = (v - n_views // 2) * disparity_scale
                    view = ndimage.shift(x_small, (shift_y, shift_x), mode="nearest")
                    noise = rng.normal(0, 0.02, view.shape).astype(np.float32)
                    y[u, v] = np.clip(view + noise, 0, 1).astype(np.float32)

            # H_ideal: shift matrix encoded as (4,4) disparity map
            H_ideal = np.zeros((4, 4), dtype=np.float32)
            for u in range(4):
                for v in range(4):
                    H_ideal[u, v] = (u - 1.5) * disparity_scale

            # Reconstruction: shift-and-add all views
            recon_full = np.zeros((view_size, view_size), dtype=np.float32)
            for u in range(n_views):
                for v in range(n_views):
                    # Shift back
                    shift_x = -(u - n_views // 2) * disparity_scale
                    shift_y = -(v - n_views // 2) * disparity_scale
                    realigned = ndimage.shift(y[u, v], (shift_y, shift_x), mode="nearest")
                    recon_full += realigned
            recon_full /= (n_views * n_views)
            recon_full = np.clip(recon_full, 0, 1).astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon_full, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(recon_full, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))
            save_png(y[1, 1], os.path.join(d, "images", f"sample_{i:02d}_y_center.png"))

    spec = {
        "n_angular_views": {"min": 4, "max": 4, "unit": ""},
        "view_resolution": {"min": 64, "max": 64, "unit": "pixels"},
        "disparity_pixels": {"min": 1.0, "max": 3.0, "unit": "pixels"},
    }
    true_spec = {
        "refocused_resolution": {"min": 128, "max": 128, "unit": "pixels"},
        "depth_planes": {"min": 1, "max": 5, "unit": ""},
    }
    write_spec(d, spec, true_spec)
    print(f"  light_field/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# LUCKY IMAGING generator
# ---------------------------------------------------------------------------

def generate_lucky_imaging(tier, n, seed_base):
    d = make_dirs("lucky_imaging", tier)
    h5_path = os.path.join(d, f"lucky_imaging_challenge_{tier}.h5")

    n_frames = 10

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_astronomical(rng, 128)
            # y: 10 short-exposure blurred frames
            y = atmospheric_blur(rng, x_true, n_frames=n_frames)
            # H_ideal: identity (128x128)
            H_ideal = np.eye(128, dtype=np.float32)
            # Reconstruction: best frame selection (highest sharpness)
            scores = [sharpness_score(y[k]) for k in range(n_frames)]
            best_idx = int(np.argmax(scores))
            recon = y[best_idx].astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y[0], os.path.join(d, "images", f"sample_{i:02d}_y_frame0.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "n_frames": {"min": 10, "max": 10, "unit": ""},
        "atm_blur_sigma_pixels": {"min": 1.0, "max": 4.0, "unit": "pixels"},
        "frame_noise_sigma": {"min": 0.01, "max": 0.03, "unit": ""},
    }
    true_spec = {
        "seeing_arcsec": {"min": 0.5, "max": 3.0, "unit": "arcsec"},
        "image_resolution_pixels": {"min": 128, "max": 128, "unit": "pixels"},
    }
    write_spec(d, spec, true_spec)
    print(f"  lucky_imaging/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# MACHINE VISION generator
# ---------------------------------------------------------------------------

def generate_machine_vision(tier, n, seed_base):
    d = make_dirs("machine_vision", tier)
    h5_path = os.path.join(d, f"machine_vision_challenge_{tier}.h5")

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_segmentation(rng, 128)  # segmentation map [0,1]
            # y: segmentation map + Gaussian noise
            noise = rng.normal(0, 0.05, x_true.shape).astype(np.float32)
            y = np.clip(x_true + noise, 0, 1).astype(np.float32)
            # H_ideal: identity
            H_ideal = np.eye(128, dtype=np.float32)
            # Reconstruction: Gaussian smoothing
            recon = ndimage.gaussian_filter(y, sigma=1.0).astype(np.float32)
            recon = np.clip(recon, 0, 1).astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y, os.path.join(d, "images", f"sample_{i:02d}_y.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "noise_sigma": {"min": 0.03, "max": 0.07, "unit": ""},
        "n_object_classes": {"min": 1, "max": 3, "unit": ""},
        "image_resolution": {"min": 128, "max": 128, "unit": "pixels"},
    }
    true_spec = {
        "n_objects": {"min": 3, "max": 8, "unit": ""},
        "object_size_px": {"min": 8, "max": 25, "unit": "pixels"},
    }
    write_spec(d, spec, true_spec)
    print(f"  machine_vision/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# MAGNETIC PARTICLE IMAGING generator
# ---------------------------------------------------------------------------

def generate_magnetic_particle(tier, n, seed_base):
    d = make_dirs("magnetic_particle", tier)
    h5_path = os.path.join(d, f"magnetic_particle_challenge_{tier}.h5")

    n_angles = 90

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_mpi_distribution(rng, 128)
            # y: radon projections (90 angles x 128 detectors)
            sino = radon_project(x_true, n_angles=n_angles)
            noise = rng.normal(0, 0.01 * sino.max(), sino.shape).astype(np.float32)
            y = (sino + noise).astype(np.float32)
            # H_ideal: clean sinogram
            H_ideal = sino.astype(np.float32)
            # Reconstruction: FBP
            recon = fbp_reconstruct(y, image_size=128)
            recon = np.clip(recon, 0, None)
            mx = recon.max()
            if mx > 0:
                recon = recon / mx
            recon = recon.astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y, os.path.join(d, "images", f"sample_{i:02d}_y.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "n_projection_angles": {"min": 90, "max": 90, "unit": ""},
        "drive_field_amplitude_mt": {"min": 10, "max": 20, "unit": "mT"},
        "particle_size_nm": {"min": 20, "max": 30, "unit": "nm"},
    }
    true_spec = {
        "n_particle_clusters": {"min": 2, "max": 6, "unit": ""},
        "cluster_sigma_px": {"min": 5, "max": 15, "unit": "pixels"},
    }
    write_spec(d, spec, true_spec)
    print(f"  magnetic_particle/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# MALDI MSI generator
# ---------------------------------------------------------------------------

def generate_maldi_msi(tier, n, seed_base):
    d = make_dirs("maldi_msi", tier)
    h5_path = os.path.join(d, f"maldi_msi_challenge_{tier}.h5")

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_mass_spec_image(rng, 128, 5)  # (128,128,5)
            # y: measured intensities with multiplicative + additive noise
            mult_noise = rng.lognormal(mean=0, sigma=0.2, size=x_true.shape).astype(np.float32)
            add_noise = rng.normal(0, 0.02, x_true.shape).astype(np.float32)
            y = np.clip(x_true * mult_noise + np.abs(add_noise), 0, None).astype(np.float32)
            # H_ideal: identity (128x128)
            H_ideal = np.eye(128, dtype=np.float32)
            # Reconstruction: normalize y per channel
            recon = np.zeros_like(y, dtype=np.float32)
            for m in range(5):
                ch = y[:, :, m]
                mx = ch.max()
                recon[:, :, m] = ch / mx if mx > 0 else ch

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            # PNG: composite RGB of first 3 masses
            rgb = np.zeros((128, 128, 3), dtype=np.float32)
            for c in range(3):
                ch = x_true[:, :, c]
                mx = ch.max()
                rgb[:, :, c] = ch / mx if mx > 0 else ch
            Image.fromarray((rgb * 255).astype(np.uint8)).save(
                os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))

    spec = {
        "n_mass_channels": {"min": 5, "max": 5, "unit": ""},
        "spatial_resolution_um": {"min": 10, "max": 100, "unit": "um"},
        "mass_range_da": {"min": 200, "max": 1000, "unit": "Da"},
    }
    true_spec = {
        "tissue_regions": {"min": 1, "max": 4, "unit": ""},
        "ion_intensity": {"min": 0.0, "max": 1.0, "unit": "normalized"},
    }
    write_spec(d, spec, true_spec)
    print(f"  maldi_msi/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# MATRIX COMPLETION generator
# ---------------------------------------------------------------------------

def generate_matrix(tier, n, seed_base):
    d = make_dirs("matrix", tier)
    h5_path = os.path.join(d, f"matrix_challenge_{tier}.h5")

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_matrix(rng, 64, rank=5)  # (64,64)
            # Binary mask: 50% observed
            mask = rng.random(x_true.shape) > 0.5  # True = observed
            H_ideal = mask.astype(np.float32)
            # y: observed entries (missing = 0)
            y = (x_true * mask).astype(np.float32)
            # Reconstruction: mean-fill missing
            observed_mean = x_true[mask].mean() if mask.any() else 0.5
            recon = y.copy()
            recon[~mask] = observed_mean
            recon = recon.astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y, os.path.join(d, "images", f"sample_{i:02d}_y.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "matrix_size": {"min": 64, "max": 64, "unit": ""},
        "observation_fraction": {"min": 0.5, "max": 0.5, "unit": ""},
        "true_rank": {"min": 5, "max": 5, "unit": ""},
    }
    true_spec = {
        "rank": {"min": 5, "max": 5, "unit": ""},
        "condition_number": {"min": 1.0, "max": 50.0, "unit": ""},
    }
    write_spec(d, spec, true_spec)
    print(f"  matrix/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# MFM generator
# ---------------------------------------------------------------------------

def generate_mfm(tier, n, seed_base):
    d = make_dirs("mfm", tier)
    h5_path = os.path.join(d, f"mfm_challenge_{tier}.h5")

    tip_psf = mfm_tip_response(128)
    # Wiener noise power for deconvolution
    noise_power = 0.05

    with h5py.File(h5_path, "w") as f:
        for i in range(n):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true = synthetic_magnetic_force(rng, 128)  # [-1, 1]
            # y: convolve with MFM tip response (Laplacian-of-Gaussian)
            y_clean = convolve2d(x_true, tip_psf)
            noise = rng.normal(0, 0.02, x_true.shape).astype(np.float32)
            y = (y_clean + noise).astype(np.float32)
            # H_ideal: tip PSF (128x128)
            H_ideal = tip_psf
            # Reconstruction: Wiener deconvolution
            recon = wiener_deconv(y, tip_psf, noise_power=noise_power)
            mx = np.abs(recon).max()
            if mx > 0:
                recon = recon / mx
            recon = recon.astype(np.float32)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true, dtype="float32")
            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32")

            save_png(x_true, os.path.join(d, "images", f"sample_{i:02d}_x_true.png"))
            save_png(y, os.path.join(d, "images", f"sample_{i:02d}_y.png"))
            save_png(recon, os.path.join(d, "images", f"sample_{i:02d}_recon.png"))

    spec = {
        "tip_lift_height_nm": {"min": 20, "max": 100, "unit": "nm"},
        "tip_radius_nm": {"min": 20, "max": 50, "unit": "nm"},
        "scan_noise_level": {"min": 0.01, "max": 0.03, "unit": ""},
    }
    true_spec = {
        "n_magnetic_domains": {"min": 3, "max": 8, "unit": ""},
        "domain_size_px": {"min": 8, "max": 25, "unit": "pixels"},
    }
    write_spec(d, spec, true_spec)
    print(f"  mfm/{tier}: {n} samples -> {h5_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = {
    "ivus":              generate_ivus,
    "libs":              generate_libs,
    "light_field":       generate_light_field,
    "lucky_imaging":     generate_lucky_imaging,
    "machine_vision":    generate_machine_vision,
    "magnetic_particle": generate_magnetic_particle,
    "maldi_msi":         generate_maldi_msi,
    "matrix":            generate_matrix,
    "mfm":               generate_mfm,
}

if __name__ == "__main__":
    import time
    t0 = time.time()
    print("=== Batch-6 Dataset Generation ===")
    print(f"Output root: {OUT_BASE}\n")

    for modality, gen_fn in GENERATORS.items():
        print(f"[{modality}]")
        for tier, cfg in TIERS.items():
            gen_fn(tier, cfg["n"], cfg["seed_base"])

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print("\nSummary:")
    for modality in GENERATORS:
        for tier in TIERS:
            h5p = os.path.join(OUT_BASE, modality, tier, f"{modality}_challenge_{tier}.h5")
            if os.path.exists(h5p):
                sz = os.path.getsize(h5p) / 1024
                with h5py.File(h5p, "r") as f:
                    n = len(f.keys())
                print(f"  {modality}/{tier}: {n} samples, {sz:.1f} KB")
            else:
                print(f"  {modality}/{tier}: MISSING!")
