"""
Generate benchmark datasets for batch 2 modalities:
cars, cathodoluminescence, cest_mri, ceus, clem,
coded_exposure, confocal_endomicroscopy, confocal_livecell, coronagraphy, cryo_et

Tiers: public (12), dev (20), hidden (20)
"""

import h5py
import numpy as np
import json
import pathlib
from scipy.ndimage import gaussian_filter, convolve

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available — skipping PNG previews")

ROOT = pathlib.Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model")
DATASETS_ROOT = ROOT / "datasets" / "benchmark"

TIERS = {
    "public": 12,
    "dev": 20,
    "hidden": 20,
}

SEED_OFFSETS = {
    "public": 2000,
    "dev": 8000,
    "hidden": 9500,
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def save_png(arr, path):
    """Save a 2D float array as a grayscale PNG, normalising to [0,255]."""
    if not HAS_PIL:
        return
    try:
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]  # take first channel for preview
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
        Image.fromarray(arr.astype(np.uint8)).save(path)
    except Exception as e:
        print(f"  PNG save failed for {path}: {e}")


def make_dirs(modality, tier):
    d = DATASETS_ROOT / modality / tier
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d


def write_spec(tier_dir, modality, tier, n_samples):
    spec = {
        "modality": modality,
        "tier": tier,
        "n_samples": n_samples,
        "measurement_key": "y",
        "groundtruth_key": "x_true",
    }
    (tier_dir / "spec.json").write_text(json.dumps(spec, indent=2))


def write_true_spec(tier_dir, true_params):
    (tier_dir / "true_spec.json").write_text(json.dumps(true_params, indent=2))


def write_h5(tier_dir, modality, tier, samples):
    """
    samples: list of dicts with keys x_true, y, H_ideal, reconstruction_baseline
    """
    fname = tier_dir / f"{modality}_challenge_{tier}.h5"
    with h5py.File(fname, "w") as f:
        for i, s in enumerate(samples):
            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=s["x_true"].astype(np.float32))
            grp.create_dataset("y", data=s["y"].astype(np.float32))
            grp.create_dataset("H_ideal", data=s["H_ideal"].astype(np.float32))
            grp.create_dataset("reconstruction_baseline",
                               data=s["reconstruction_baseline"].astype(np.float32))
    return fname


# ---------------------------------------------------------------------------
# Physics generators
# ---------------------------------------------------------------------------

def make_random_image(rng, size=128):
    """Random smooth phantom via superposition of Gaussian blobs."""
    img = np.zeros((size, size), dtype=np.float32)
    n_blobs = rng.integers(5, 20)
    for _ in range(n_blobs):
        cx = rng.integers(0, size)
        cy = rng.integers(0, size)
        sigma = rng.uniform(3, 20)
        amp = rng.uniform(0.2, 1.0)
        x, y = np.mgrid[0:size, 0:size]
        blob = amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        img += blob.astype(np.float32)
    img = img / (img.max() + 1e-8)
    return img


# -- CARS --
def gen_cars(rng):
    x_true = make_random_image(rng)  # Raman susceptibility map
    nonresonant_bg = rng.uniform(0.1, 0.5)
    noise = rng.normal(0, 0.02, x_true.shape).astype(np.float32)
    y = x_true * nonresonant_bg + noise
    y = y.astype(np.float32)
    H_ideal = np.array([[nonresonant_bg]], dtype=np.float32)
    # baseline: simple background subtraction
    reconstruction_baseline = (y - nonresonant_bg * 0.5).clip(0, None)
    reconstruction_baseline = reconstruction_baseline.astype(np.float32)
    true_params = {
        "nonresonant_background": float(nonresonant_bg),
        "noise_std": 0.02,
        "model": "CARS susceptibility × NRB + noise",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- Cathodoluminescence --
def gen_cathodoluminescence(rng):
    x_true = make_random_image(rng)  # emission map
    sigma_psf = rng.uniform(1.5, 4.0)
    blurred = gaussian_filter(x_true, sigma=sigma_psf)
    # Poisson noise: scale up, sample, scale down
    scale = 100.0
    y = rng.poisson(blurred * scale).astype(np.float32) / scale
    H_ideal = np.array([[sigma_psf]], dtype=np.float32)
    # RL deconvolution (5 iters)
    reconstruction_baseline = rl_deconv_2d(y, sigma_psf, n_iter=5)
    true_params = {
        "psf_sigma": float(sigma_psf),
        "scale_factor": scale,
        "model": "Gaussian PSF blur + Poisson noise",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


def rl_deconv_2d(y, sigma, n_iter=5):
    """Simple Richardson-Lucy deconvolution."""
    est = y.copy() + 1e-8
    for _ in range(n_iter):
        ratio = y / (gaussian_filter(est, sigma) + 1e-8)
        est = est * gaussian_filter(ratio, sigma)
        est = np.clip(est, 0, None)
    return est.astype(np.float32)


# -- CEST MRI --
def gen_cest_mri(rng):
    x_true = make_random_image(rng)  # CEST effect map (0-1)
    noise_std = rng.uniform(0.01, 0.05)
    noise = rng.normal(0, noise_std, x_true.shape).astype(np.float32)
    # Z-spectrum representation: noisy CEST effect map
    y = x_true + noise
    y = y.astype(np.float32)
    H_ideal = np.array([[noise_std]], dtype=np.float32)
    # baseline: Lorentzian fit approximation → return y (identity-like)
    reconstruction_baseline = y.copy()
    true_params = {
        "noise_std": float(noise_std),
        "model": "CEST Z-spectrum with Gaussian noise",
        "exchange_rate_hz": float(rng.uniform(50, 500)),
        "solute_concentration_mM": float(rng.uniform(5, 100)),
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- CEUS --
def gen_ceus(rng):
    x_true = make_random_image(rng)  # perfusion map
    # Speckle noise: multiplicative
    speckle = rng.exponential(scale=1.0, size=x_true.shape).astype(np.float32)
    bubble_noise = rng.uniform(0, 0.1, x_true.shape).astype(np.float32)
    y = (x_true * speckle + bubble_noise).astype(np.float32)
    H_ideal = np.array([[1.0]], dtype=np.float32)  # identity forward model
    # baseline: temporal averaging approximation (smooth)
    reconstruction_baseline = gaussian_filter(y, sigma=2.0).astype(np.float32)
    true_params = {
        "noise_type": "speckle_multiplicative_plus_bubble",
        "bubble_noise_level": 0.1,
        "model": "Perfusion × speckle + bubble noise",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- CLEM --
def gen_clem(rng):
    # Dual channel: fluorescence + EM (128,128,2)
    ch_fluor = make_random_image(rng)
    ch_em = make_random_image(rng)
    x_true = np.stack([ch_fluor, ch_em], axis=-1)  # (128,128,2)

    # Different noise per channel
    noise_fluor = rng.poisson(ch_fluor * 50).astype(np.float32) / 50.0
    noise_em = rng.normal(0, 0.03, ch_em.shape).astype(np.float32) + ch_em

    y = np.stack([noise_fluor, noise_em], axis=-1).astype(np.float32)
    H_ideal = np.array([[1.0, 1.0]], dtype=np.float32)  # identity per channel
    # baseline: mean across channels → (128,128)
    recon_2d = x_true.mean(axis=-1)
    # store as (128,128,2) by repeating
    reconstruction_baseline = np.stack([recon_2d, recon_2d], axis=-1).astype(np.float32)
    true_params = {
        "fluorescence_channel": "Poisson noise, scale=50",
        "em_channel": "Gaussian noise, std=0.03",
        "model": "CLEM dual-channel registration",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- Coded Exposure --
def gen_coded_exposure(rng):
    x_true = make_random_image(rng)  # sharp image
    # Horizontal motion blur kernel
    kernel_len = rng.integers(5, 20)
    kernel = np.zeros((1, kernel_len), dtype=np.float32)
    kernel[0, :] = 1.0 / kernel_len
    y = convolve(x_true, kernel, mode='reflect').astype(np.float32)
    # Add small noise
    y += rng.normal(0, 0.005, y.shape).astype(np.float32)
    H_ideal = kernel.astype(np.float32)
    # Wiener deconvolution (frequency domain)
    reconstruction_baseline = wiener_deconv(y, kernel, snr=30.0)
    true_params = {
        "blur_kernel_length": int(kernel_len),
        "kernel_type": "horizontal_uniform_motion",
        "noise_std": 0.005,
        "model": "Coded exposure motion blur",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


def wiener_deconv(y, kernel, snr=30.0):
    """Wiener deconvolution in frequency domain."""
    from numpy.fft import fft2, ifft2
    padded = np.zeros_like(y)
    padded[:kernel.shape[0], :kernel.shape[1]] = kernel
    H = fft2(padded)
    Y = fft2(y)
    H_conj = np.conj(H)
    denom = H_conj * H + (1.0 / snr)
    X_hat = (H_conj / denom) * Y
    recon = np.real(ifft2(X_hat)).astype(np.float32)
    return np.clip(recon, 0, 1)


# -- Confocal Endomicroscopy --
def gen_confocal_endomicroscopy(rng):
    x_true = make_random_image(rng)  # tissue structure
    sigma_psf = rng.uniform(2.0, 5.0)
    blurred = gaussian_filter(x_true, sigma=sigma_psf)
    noise_std = rng.uniform(0.01, 0.05)
    y = (blurred + rng.normal(0, noise_std, blurred.shape)).astype(np.float32)
    y = np.clip(y, 0, None)
    H_ideal = np.array([[sigma_psf, noise_std]], dtype=np.float32)
    reconstruction_baseline = rl_deconv_2d(y, sigma_psf, n_iter=5)
    true_params = {
        "psf_sigma": float(sigma_psf),
        "noise_std": float(noise_std),
        "model": "Gaussian PSF + Gaussian noise (confocal endo)",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- Confocal Live Cell --
def gen_confocal_livecell(rng):
    x_true = make_random_image(rng)  # cell structure (finer features)
    # narrower PSF than endomicroscopy
    sigma_psf = rng.uniform(0.8, 2.5)
    blurred = gaussian_filter(x_true, sigma=sigma_psf)
    scale = 200.0
    y = rng.poisson(blurred * scale).astype(np.float32) / scale
    H_ideal = np.array([[sigma_psf]], dtype=np.float32)
    reconstruction_baseline = rl_deconv_2d(y, sigma_psf, n_iter=5)
    true_params = {
        "psf_sigma": float(sigma_psf),
        "poisson_scale": scale,
        "model": "Gaussian PSF + Poisson noise (confocal live-cell)",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- Coronagraphy --
def gen_coronagraphy(rng):
    """
    x_true: exoplanet field with faint sources around bright central star.
    Coronagraph mask suppresses the central star region.
    y: masked field after coronagraph subtraction.
    """
    size = 128
    x_true = np.zeros((size, size), dtype=np.float32)
    # bright central star
    cx, cy = size // 2, size // 2
    star_amp = rng.uniform(5.0, 10.0)
    sigma_star = rng.uniform(2.0, 5.0)
    x, y_grid = np.mgrid[0:size, 0:size]
    star = star_amp * np.exp(-((x - cx)**2 + (y_grid - cy)**2) / (2 * sigma_star**2))
    x_true += star.astype(np.float32)

    # faint planet sources
    n_planets = rng.integers(1, 5)
    planet_positions = []
    for _ in range(n_planets):
        px = rng.integers(10, size - 10)
        py = rng.integers(10, size - 10)
        # avoid centre
        while abs(px - cx) < 15 and abs(py - cy) < 15:
            px = rng.integers(10, size - 10)
            py = rng.integers(10, size - 10)
        planet_amp = rng.uniform(0.05, 0.3)
        sigma_planet = rng.uniform(1.0, 3.0)
        planet = planet_amp * np.exp(-((x - px)**2 + (y_grid - py)**2) / (2 * sigma_planet**2))
        x_true += planet.astype(np.float32)
        planet_positions.append({"x": int(px), "y": int(py), "amp": float(planet_amp)})

    # Coronagraph: suppress central disk (Lyot-style mask)
    mask_radius = rng.uniform(8, 15)
    dist = np.sqrt((x - cx)**2 + (y_grid - cy)**2)
    coronagraph_mask = (dist > mask_radius).astype(np.float32)
    # Reference star subtraction removes star PSF; residual = planets + speckle
    ref_star = star.copy()
    speckle = rng.normal(0, 0.05, x_true.shape).astype(np.float32)
    y_meas = (x_true - ref_star) * coronagraph_mask + speckle
    y_meas = y_meas.astype(np.float32)

    H_ideal = np.stack([coronagraph_mask, ref_star.astype(np.float32)], axis=0)  # (2,128,128)
    # Store as (2, H, W) flattened → we'll keep 3D array in H_ideal
    H_ideal_flat = np.array([mask_radius, star_amp, sigma_star], dtype=np.float32)

    # baseline: reference subtraction (already done essentially)
    reconstruction_baseline = np.clip(y_meas * coronagraph_mask, 0, None)

    true_params = {
        "star_amplitude": float(star_amp),
        "star_sigma": float(sigma_star),
        "mask_radius_px": float(mask_radius),
        "n_planets": int(n_planets),
        "planet_positions": planet_positions,
        "model": "Coronagraphy Lyot-mask reference subtraction",
    }
    return dict(x_true=x_true, y=y_meas, H_ideal=H_ideal_flat,
                reconstruction_baseline=reconstruction_baseline), true_params


# -- Cryo-ET --
def gen_cryo_et(rng):
    """
    x_true: 2D slice (64x64) from a simulated 3D tomogram.
    y: (60, 64) tilted projections (sinogram-like).
    reconstruction_baseline: FBP approximation via back-projection.
    """
    size = 64
    # Build a small 3D volume and take 2D slice
    vol = np.zeros((size, size, size), dtype=np.float32)
    n_blobs = rng.integers(3, 10)
    for _ in range(n_blobs):
        cx = rng.integers(5, size - 5)
        cy = rng.integers(5, size - 5)
        cz = rng.integers(5, size - 5)
        sigma = rng.uniform(2, 8)
        amp = rng.uniform(0.2, 1.0)
        gx, gy, gz = np.mgrid[0:size, 0:size, 0:size]
        blob = amp * np.exp(-((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2) / (2 * sigma**2))
        vol += blob.astype(np.float32)
    vol /= vol.max() + 1e-8

    x_true = vol[:, :, size // 2]  # 2D slice (64,64)

    # Tilted projections: 60 angles from -60 to +60 degrees
    from scipy.ndimage import rotate
    angles = np.linspace(-60, 60, 60)
    projections = []
    for angle in angles:
        rotated = rotate(x_true, angle, reshape=False, mode='constant', cval=0.0)
        proj = rotated.sum(axis=1)  # project along one axis → (64,)
        projections.append(proj)
    y = np.array(projections, dtype=np.float32)  # (60, 64)

    # Add dose-limited noise
    noise_std = rng.uniform(0.01, 0.05)
    y += rng.normal(0, noise_std, y.shape).astype(np.float32)

    H_ideal = angles.astype(np.float32)  # tilt angles

    # Simple back-projection reconstruction
    reconstruction_baseline = simple_backproject(y, angles, size)

    true_params = {
        "n_tilt_angles": 60,
        "tilt_range_deg": [-60, 60],
        "noise_std": float(noise_std),
        "n_blobs": int(n_blobs),
        "model": "Cryo-ET tilt series + FBP reconstruction",
    }
    return dict(x_true=x_true, y=y, H_ideal=H_ideal,
                reconstruction_baseline=reconstruction_baseline), true_params


def simple_backproject(sinogram, angles, size):
    """Simple back-projection (no ramp filter)."""
    from scipy.ndimage import rotate
    recon = np.zeros((size, size), dtype=np.float32)
    for i, angle in enumerate(angles):
        proj = sinogram[i]  # (size,)
        # Expand projection across the image
        back = np.tile(proj, (size, 1))  # (size, size)
        # Rotate back
        back_rot = rotate(back, -angle, reshape=False, mode='constant', cval=0.0)
        recon += back_rot
    recon /= len(angles)
    recon = np.clip(recon, 0, None)
    recon = recon / (recon.max() + 1e-8)
    return recon.astype(np.float32)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    "cars": gen_cars,
    "cathodoluminescence": gen_cathodoluminescence,
    "cest_mri": gen_cest_mri,
    "ceus": gen_ceus,
    "clem": gen_clem,
    "coded_exposure": gen_coded_exposure,
    "confocal_endomicroscopy": gen_confocal_endomicroscopy,
    "confocal_livecell": gen_confocal_livecell,
    "coronagraphy": gen_coronagraphy,
    "cryo_et": gen_cryo_et,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_modality(modality, gen_fn):
    print(f"\n=== Generating: {modality} ===")
    for tier, n_samples in TIERS.items():
        tier_dir = make_dirs(modality, tier)
        seed_base = SEED_OFFSETS[tier]
        samples = []
        true_params_all = {}

        for i in range(n_samples):
            seed = seed_base + i * 17
            rng = np.random.default_rng(seed)
            sample_data, true_params = gen_fn(rng)
            samples.append(sample_data)
            true_params_all[f"sample_{i:02d}"] = true_params

            # Save PNG preview
            x_arr = sample_data["x_true"]
            if x_arr.ndim == 3:
                preview = x_arr[..., 0]
            else:
                preview = x_arr
            save_png(preview, tier_dir / "images" / f"sample_{i:02d}_gt.png")
            y_arr = sample_data["y"]
            if y_arr.ndim == 3:
                preview_y = y_arr[..., 0]
            elif y_arr.ndim == 2:
                preview_y = y_arr
            else:
                preview_y = y_arr
            save_png(preview_y, tier_dir / "images" / f"sample_{i:02d}_y.png")

        h5_path = write_h5(tier_dir, modality, tier, samples)
        write_spec(tier_dir, modality, tier, n_samples)
        write_true_spec(tier_dir, true_params_all)

        print(f"  [{tier}] {n_samples} samples -> {h5_path.name}")

    print(f"  Done: {modality}")


def verify_outputs():
    print("\n=== Verification ===")
    ok = True
    for modality in GENERATORS:
        for tier, n_samples in TIERS.items():
            tier_dir = DATASETS_ROOT / modality / tier
            h5_file = tier_dir / f"{modality}_challenge_{tier}.h5"
            spec_file = tier_dir / "spec.json"
            true_spec_file = tier_dir / "true_spec.json"
            images_dir = tier_dir / "images"

            missing = []
            if not h5_file.exists():
                missing.append("H5 file")
            if not spec_file.exists():
                missing.append("spec.json")
            if not true_spec_file.exists():
                missing.append("true_spec.json")
            if not images_dir.exists():
                missing.append("images/")

            if missing:
                print(f"  FAIL {modality}/{tier}: missing {missing}")
                ok = False
            else:
                # Quick check HDF5 structure
                with h5py.File(h5_file, "r") as f:
                    n_groups = len(f.keys())
                    sample0_keys = list(f["sample_00"].keys()) if "sample_00" in f else []
                expected_keys = {"x_true", "y", "H_ideal", "reconstruction_baseline"}
                if n_groups != n_samples:
                    print(f"  WARN {modality}/{tier}: expected {n_samples} groups, got {n_groups}")
                elif not expected_keys.issubset(set(sample0_keys)):
                    print(f"  WARN {modality}/{tier}: sample_00 missing keys: {expected_keys - set(sample0_keys)}")
                else:
                    print(f"  OK   {modality}/{tier}: {n_groups} samples, keys={sorted(sample0_keys)}")
    if ok:
        print("\nAll outputs verified successfully.")
    else:
        print("\nSome outputs had issues — see above.")
    return ok


def main():
    print(f"Output root: {DATASETS_ROOT}")
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    for modality, gen_fn in GENERATORS.items():
        generate_modality(modality, gen_fn)

    verify_outputs()
    print("\nBatch 2 dataset generation complete.")


if __name__ == "__main__":
    main()
