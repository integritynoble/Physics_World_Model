#!/usr/bin/env python3
"""
Generate benchmark datasets for batch-7 modalities:
  minflux, muon_tomo, neutron_diffraction, neutron_tomo, nirs_brain,
  nsom, ocean_acoustic_tomo, ocean_color, octa, panorama

Output structure per modality:
  datasets/benchmark/{modality}/{tier}/
    {modality}_challenge_{tier}.h5     (sample_NN groups: x_true, y, H_ideal, reconstruction_baseline)
    spec.json
    true_spec.json
    images/                            (PNG previews per sample)

Tier sizes:  public=12, dev=20, hidden=20
Seeds: public = 1100 + i*17, dev = 7100 + i*17, hidden = 9200 + i*17
"""

import json
import os

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import wiener

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets", "benchmark")
ROOT = os.path.normpath(ROOT)

TIERS = {
    "public": {"n": 12, "seed_base": 1100},
    "dev":    {"n": 20, "seed_base": 7100},
    "hidden": {"n": 20, "seed_base": 9200},
}


def seed_for(tier: str, i: int) -> int:
    return TIERS[tier]["seed_base"] + i * 17


def save_png(arr: np.ndarray, path: str) -> None:
    """Save float32 array (any shape) as 8-bit PNG, collapses channel dim if present."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = arr.copy()
    if a.ndim == 3 and a.shape[2] in (3, 4):
        # RGB/RGBA — normalise each channel independently
        out = np.zeros_like(a, dtype=np.uint8)
        for c in range(a.shape[2]):
            ch = a[:, :, c]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                ch = (ch - mn) / (mx - mn)
            out[:, :, c] = (ch * 255).astype(np.uint8)
        if a.shape[2] == 4:
            Image.fromarray(out[:, :, :3], "RGB").save(path)
        else:
            Image.fromarray(out, "RGB").save(path)
    elif a.ndim == 3 and a.shape[0] in (1, 3, 4):
        # channel-first — recurse
        save_png(np.moveaxis(a, 0, -1), path)
    else:
        # Greyscale (possibly 3-D with 1 channel)
        if a.ndim == 3:
            a = a[:, :, 0]
        mn, mx = a.min(), a.max()
        if mx > mn:
            a = (a - mn) / (mx - mn)
        img8 = (a * 255).astype(np.uint8)
        Image.fromarray(img8, "L").save(path)


def fbp_simple(sinogram: np.ndarray) -> np.ndarray:
    """Minimal filtered back-projection (ram-lak filter in frequency domain)."""
    n_angles, n_det = sinogram.shape
    # Ram-Lak filter
    freq = np.fft.rfftfreq(n_det)
    filt = np.abs(freq)
    filtered = np.fft.irfft(np.fft.rfft(sinogram, axis=1) * filt[None, :], axis=1)
    # Back-project
    img_size = n_det
    out = np.zeros((img_size, img_size), dtype=np.float32)
    cx = cy = img_size // 2
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    xs = np.arange(img_size) - cx
    ys = np.arange(img_size) - cy
    XX, YY = np.meshgrid(xs, ys)
    for k, theta in enumerate(angles):
        t = XX * np.cos(theta) + YY * np.sin(theta)
        t_idx = np.clip(
            np.round(t + n_det // 2).astype(int), 0, n_det - 1
        )
        out += filtered[k][t_idx]
    out = out * (np.pi / (2 * n_angles))
    mn, mx = out.min(), out.max()
    if mx > mn:
        out = (out - mn) / (mx - mn)
    return out.astype(np.float32)


def radon_simple(x: np.ndarray, n_angles: int = 90) -> np.ndarray:
    """Simple parallel-beam Radon transform."""
    n = x.shape[0]
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    sino = np.zeros((n_angles, n), dtype=np.float32)
    cx = cy = n // 2
    xs = np.arange(n) - cx
    ys = np.arange(n) - cy
    XX, YY = np.meshgrid(xs, ys)
    for k, theta in enumerate(angles):
        t = XX * np.cos(theta) + YY * np.sin(theta)
        t_idx = np.clip(
            np.round(t + n // 2).astype(int), 0, n - 1
        )
        np.add.at(sino[k], t_idx.ravel(), x.ravel())
    return sino


def write_tier(modality: str, tier: str, samples: list,
               spec_data: dict, true_spec_data: dict) -> None:
    """
    samples: list of dicts with keys x_true, y, H_ideal, reconstruction_baseline.
    """
    n = len(samples)
    out_dir = os.path.join(ROOT, modality, tier)
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, f"{modality}_challenge_{tier}.h5")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = modality
        hf.attrs["tier"] = tier
        hf.attrs["version"] = "1.0"
        hf.attrs["n_samples"] = n

        for i, s in enumerate(samples):
            key = f"sample_{i:02d}"
            grp = hf.create_group(key)
            grp.create_dataset("x_true",                   data=s["x_true"].astype(np.float32),                  compression="gzip")
            grp.create_dataset("y",                        data=s["y"].astype(np.float32),                       compression="gzip")
            grp.create_dataset("H_ideal",                  data=s["H_ideal"].astype(np.float32),                 compression="gzip")
            grp.create_dataset("reconstruction_baseline",  data=s["reconstruction_baseline"].astype(np.float32), compression="gzip")
            if "mismatch_params" in s:
                grp.attrs["mismatch_params"] = json.dumps(s["mismatch_params"])

            # Save preview PNGs
            save_png(s["x_true"],                 os.path.join(img_dir, f"{key}_x_true.png"))
            save_png(s["y"],                      os.path.join(img_dir, f"{key}_y.png"))
            save_png(s["reconstruction_baseline"], os.path.join(img_dir, f"{key}_recon.png"))

    # spec.json
    spec_out = dict(spec_data)
    spec_out["tier"] = tier
    spec_out["n_samples"] = n
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec_out, f, indent=2)

    # true_spec.json
    with open(os.path.join(out_dir, "true_spec.json"), "w") as f:
        json.dump(true_spec_data, f, indent=2)

    sz = os.path.getsize(h5_path) / 1e6
    print(f"  [{tier:6s}] {n:2d} samples -> {h5_path}  ({sz:.1f} MB)")


# ===========================================================================
# 1. MINFLUX
# ===========================================================================
def gen_minflux(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: sparse fluorophore positions (128x128)
        x_true = np.zeros((128, 128), dtype=np.float32)
        n_fluoro = int(rng.integers(30, 80))
        ys_f = rng.integers(4, 124, n_fluoro)
        xs_f = rng.integers(4, 124, n_fluoro)
        intensities = rng.uniform(0.3, 1.0, n_fluoro).astype(np.float32)
        x_true[ys_f, xs_f] = intensities

        # Mismatch: position noise in nm (nominal 0, range 0-5)
        if tier == "hidden":
            pos_noise_nm = float(rng.uniform(0.0, 5.0))
            photon_count = float(rng.uniform(50, 2000))
        else:
            pos_noise_nm = 0.5
            photon_count = 500.0

        # y: localization density with position noise + Gaussian broadening
        noise_px = pos_noise_nm / 6.5  # 6.5 nm/pixel
        y = np.zeros((128, 128), dtype=np.float32)
        ys_noisy = np.clip(
            (ys_f + rng.normal(0, noise_px + 0.2, n_fluoro)).astype(int), 0, 127
        )
        xs_noisy = np.clip(
            (xs_f + rng.normal(0, noise_px + 0.2, n_fluoro)).astype(int), 0, 127
        )
        y[ys_noisy, xs_noisy] = intensities
        y = ndimage.gaussian_filter(y, sigma=1.2)
        y += rng.normal(0, 0.01, y.shape).astype(np.float32)
        y = np.clip(y, 0, None)

        # H_ideal: noise-free broadened ground truth
        H_ideal = ndimage.gaussian_filter(x_true.copy(), sigma=1.0)

        # Reconstruction: Gaussian smoothing of y
        recon = ndimage.gaussian_filter(y, sigma=0.8)

        true_specs[f"sample_{i:02d}"] = {
            "n_fluorophores": int(n_fluoro),
            "pos_noise_nm": pos_noise_nm,
            "photon_count": photon_count,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"pos_noise_nm": pos_noise_nm, "photon_count": photon_count},
        })

    spec = {
        "modality": "minflux",
        "workflow": {"dag": "C --> D", "description": "MINFLUX nanoscopy localization"},
        "geometry": {"image_size": 128, "pixel_size_nm": 6.5},
        "mismatch_ranges": {
            "pos_noise_nm": {"min": 0.0, "max": 5.0, "unit": "nm"},
            "photon_count": {"min": 50, "max": 2000, "unit": "photons"},
        },
        "noise_model": {"type": "position_noise_plus_gaussian_readout"},
        "reconstruction_hint": "Localize fluorophore positions from density map y.",
    }
    write_tier("minflux", tier, samples, spec, true_specs)


# ===========================================================================
# 2. MUON_TOMO
# ===========================================================================
def gen_muon_tomo(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: density map (rock/building) 128x128
        x_true = np.zeros((128, 128), dtype=np.float32)
        # Background rock
        x_true[:] = rng.uniform(0.3, 0.6)
        # Add cavities (low density) and dense inclusions
        for _ in range(rng.integers(2, 6)):
            cy, cx = rng.integers(20, 108, 2)
            ry, rx = rng.integers(5, 25, 2)
            val = rng.uniform(-0.25, -0.1)
            yy, xx = np.ogrid[:128, :128]
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            x_true[mask] = np.clip(x_true[mask] + val, 0, 1)
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.integers(20, 108, 2)
            ry, rx = rng.integers(3, 15, 2)
            val = rng.uniform(0.1, 0.4)
            yy, xx = np.ogrid[:128, :128]
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            x_true[mask] = np.clip(x_true[mask] + val, 0, 1)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            flux_var = float(rng.uniform(-0.1, 0.1))
            angle_err = float(rng.uniform(-2.0, 2.0))
        else:
            flux_var = 0.0
            angle_err = 0.0

        # y: muon transmission projections (90 angles, 128 detectors)
        sino = radon_simple(x_true, n_angles=90)
        # Convert to transmission (muons have Beer-Lambert attenuation)
        transmission = np.exp(-sino * (1.0 + flux_var))
        # Add Poisson-like noise
        noise_scale = 0.01 + abs(flux_var) * 0.05
        y = transmission + rng.normal(0, noise_scale, transmission.shape).astype(np.float32)
        y = np.clip(y, 0, 1).astype(np.float32)

        H_ideal = np.exp(-sino).astype(np.float32)

        # Reconstruction: FBP on -log(transmission)
        attenuation = -np.log(np.clip(y, 1e-6, 1.0)).astype(np.float32)
        recon = fbp_simple(attenuation)

        true_specs[f"sample_{i:02d}"] = {
            "flux_variation": flux_var,
            "angle_error_deg": angle_err,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"flux_variation": flux_var, "angle_error_deg": angle_err},
        })

    spec = {
        "modality": "muon_tomo",
        "workflow": {"dag": "Source → P → Σ → D", "description": "Muon transmission tomography"},
        "geometry": {"image_size": 128, "n_angles": 90, "n_detectors": 128},
        "mismatch_ranges": {
            "flux_variation": {"min": -0.1, "max": 0.1, "unit": ""},
            "angle_error_deg": {"min": -2.0, "max": 2.0, "unit": "degrees"},
        },
        "noise_model": {"type": "poisson_transmission"},
        "reconstruction_hint": "FBP on -log(y) transmission sinogram.",
    }
    write_tier("muon_tomo", tier, samples, spec, true_specs)


# ===========================================================================
# 3. NEUTRON_DIFFRACTION
# ===========================================================================
def gen_neutron_diffraction(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: crystal structure (periodic pattern 128x128)
        x_true = np.zeros((128, 128), dtype=np.float32)
        # Lattice parameters
        a = rng.integers(8, 20)
        b = rng.integers(8, 20)
        theta = rng.uniform(0, np.pi / 6)
        yy, xx = np.mgrid[:128, :128]
        # Rotated lattice
        xx_r = xx * np.cos(theta) - yy * np.sin(theta)
        yy_r = xx * np.sin(theta) + yy * np.cos(theta)
        lattice = (np.sin(2 * np.pi * xx_r / a) * np.sin(2 * np.pi * yy_r / b))
        x_true = ((lattice + 1) / 2).astype(np.float32)
        # Add basis atoms
        for _ in range(rng.integers(1, 4)):
            ax, ay = rng.integers(0, 128, 2)
            x_true = np.clip(x_true + 0.3 * np.exp(
                -((xx - ax) ** 2 + (yy - ay) ** 2) / (2 * 4 ** 2)
            ), 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            wavelength_err = float(rng.uniform(-0.05, 0.05))
            background = float(rng.uniform(0.0, 0.2))
        else:
            wavelength_err = 0.0
            background = 0.02

        # y: neutron diffraction pattern (|FFT|^2) + Poisson noise
        ft = np.fft.fftshift(np.fft.fft2(x_true))
        H_ideal = (np.abs(ft) ** 2).astype(np.float32)
        H_ideal = H_ideal / (H_ideal.max() + 1e-8)
        # Scale + Poisson
        scale = 1000.0
        counts = rng.poisson(np.maximum(H_ideal * scale * (1 + wavelength_err), 1e-6))
        y = (counts.astype(np.float32) / scale + background).astype(np.float32)
        y = np.clip(y, 0, None)

        # Reconstruction: inverse FFT magnitude of sqrt(y - background)
        y_corrected = np.maximum(y - background, 0)
        amp = np.sqrt(np.maximum(y_corrected, 0))
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(amp))).astype(np.float32)
        mn, mx = recon.min(), recon.max()
        if mx > mn:
            recon = (recon - mn) / (mx - mn)

        true_specs[f"sample_{i:02d}"] = {
            "lattice_a": int(a), "lattice_b": int(b),
            "wavelength_err": wavelength_err, "background": background,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"wavelength_err": wavelength_err, "background": background},
        })

    spec = {
        "modality": "neutron_diffraction",
        "workflow": {"dag": "C → F → D", "description": "Neutron powder diffraction"},
        "geometry": {"image_size": 128},
        "mismatch_ranges": {
            "wavelength_err": {"min": -0.05, "max": 0.05, "unit": "fraction"},
            "background": {"min": 0.0, "max": 0.2, "unit": "counts/pixel"},
        },
        "noise_model": {"type": "poisson"},
        "reconstruction_hint": "Inverse FFT magnitude of background-corrected diffraction pattern.",
    }
    write_tier("neutron_diffraction", tier, samples, spec, true_specs)


# ===========================================================================
# 4. NEUTRON_TOMO
# ===========================================================================
def gen_neutron_tomo(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: neutron attenuation slice 128x128
        x_true = np.zeros((128, 128), dtype=np.float32)
        # Background material
        x_true[:] = rng.uniform(0.1, 0.4)
        # Internal structure
        for _ in range(rng.integers(3, 8)):
            cy, cx = rng.integers(15, 113, 2)
            ry, rx = rng.integers(4, 30, 2)
            val = rng.uniform(-0.15, 0.3)
            yy, xx = np.ogrid[:128, :128]
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            x_true[mask] = np.clip(x_true[mask] + val, 0, 1)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            beam_harden = float(rng.uniform(0.0, 0.08))
            detector_eff = float(rng.uniform(0.85, 1.0))
        else:
            beam_harden = 0.0
            detector_eff = 1.0

        # y: transmission projections (90 angles, 128 detectors)
        sino = radon_simple(x_true, n_angles=90)
        # Beer-Lambert with beam hardening
        proj_eff = sino * (1.0 + beam_harden * sino)
        transmission = detector_eff * np.exp(-proj_eff)
        # Poisson noise (I0=5000)
        I0 = 5000.0
        I_noisy = rng.poisson(np.maximum(I0 * transmission, 1e-6)).astype(np.float32)
        y = -np.log(np.maximum(I_noisy / I0, 1e-6)).astype(np.float32)
        y = np.clip(y, 0, None)

        H_ideal = sino.astype(np.float32)

        # Reconstruction: FBP
        recon = fbp_simple(y)

        true_specs[f"sample_{i:02d}"] = {
            "beam_hardening": beam_harden,
            "detector_efficiency": detector_eff,
            "I0": I0,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"beam_hardening": beam_harden, "detector_efficiency": detector_eff},
        })

    spec = {
        "modality": "neutron_tomo",
        "workflow": {"dag": "Source → P → Σ → D", "description": "Neutron transmission tomography"},
        "geometry": {"image_size": 128, "n_angles": 90, "n_detectors": 128, "I0": 5000},
        "mismatch_ranges": {
            "beam_hardening": {"min": 0.0, "max": 0.08, "unit": ""},
            "detector_efficiency": {"min": 0.85, "max": 1.0, "unit": ""},
        },
        "noise_model": {"type": "poisson_beer_lambert", "I0": 5000},
        "reconstruction_hint": "FBP on log-attenuation sinogram y.",
    }
    write_tier("neutron_tomo", tier, samples, spec, true_specs)


# ===========================================================================
# 5. NIRS_BRAIN
# ===========================================================================
def gen_nirs_brain(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: brain hemodynamics map (64x64)
        x_true = np.zeros((64, 64), dtype=np.float32)
        # Brain boundary (ellipse)
        yy, xx = np.ogrid[:64, :64]
        brain_mask = ((yy - 32) / 28) ** 2 + ((xx - 32) / 24) ** 2 <= 1
        x_true[brain_mask] = rng.uniform(0.3, 0.6)
        # Activation regions
        for _ in range(rng.integers(2, 5)):
            cy, cx = rng.integers(10, 54, 2)
            r = rng.integers(3, 10)
            val = rng.uniform(0.1, 0.4)
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            x_true[mask & brain_mask] = np.clip(
                x_true[mask & brain_mask] + val, 0, 1
            )
        x_true = x_true.astype(np.float32)

        # Mismatch
        if tier == "hidden":
            sd_coupling = float(rng.uniform(0.5, 1.5))
            motion_art = float(rng.uniform(0.0, 0.3))
        else:
            sd_coupling = 1.0
            motion_art = 0.0

        # y: 32x32 downsampled NIRS measurements + noise
        # Simulate optical diffusion: blur x_true then downsample
        blurred = ndimage.gaussian_filter(x_true * sd_coupling, sigma=2.0)
        y_full = blurred + motion_art * rng.normal(0, 0.05, blurred.shape).astype(np.float32)
        # Downsample 64->32
        y = y_full[::2, ::2].astype(np.float32)
        y += rng.normal(0, 0.02, y.shape).astype(np.float32)
        y = np.clip(y, 0, None)

        # H_ideal: noise-free blurred + downsampled
        H_ideal = blurred[::2, ::2].astype(np.float32)

        # Reconstruction: bicubic upsample from 32->64
        from PIL import Image as PILImage
        y_pil = PILImage.fromarray(
            ((y - y.min()) / (y.max() - y.min() + 1e-8) * 255).astype(np.uint8)
        )
        recon_pil = y_pil.resize((64, 64), PILImage.BICUBIC)
        recon = np.array(recon_pil, dtype=np.float32) / 255.0

        true_specs[f"sample_{i:02d}"] = {
            "sd_coupling": sd_coupling,
            "motion_artifact": motion_art,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"sd_coupling": sd_coupling, "motion_artifact": motion_art},
        })

    spec = {
        "modality": "nirs_brain",
        "workflow": {"dag": "M → R,P → D", "description": "fNIRS brain hemodynamics imaging"},
        "geometry": {"x_size": 64, "y_size": 32},
        "mismatch_ranges": {
            "sd_coupling": {"min": 0.5, "max": 1.5, "unit": ""},
            "motion_artifact": {"min": 0.0, "max": 0.3, "unit": ""},
        },
        "noise_model": {"type": "gaussian", "description": "Gaussian readout noise"},
        "reconstruction_hint": "Bicubic upsample from 32x32 NIRS measurements to 64x64.",
    }
    write_tier("nirs_brain", tier, samples, spec, true_specs)


# ===========================================================================
# 6. NSOM
# ===========================================================================
def gen_nsom(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: near-field optical image with nanoscale features 128x128
        x_true = np.zeros((128, 128), dtype=np.float32)
        yy, xx = np.mgrid[:128, :128]
        # Background
        x_true[:] = rng.uniform(0.1, 0.3)
        # Nanoscale features (sharp edges, small structures)
        for _ in range(rng.integers(5, 15)):
            cy, cx = rng.integers(10, 118, 2)
            r = rng.integers(2, 8)
            val = rng.uniform(0.2, 0.7)
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            x_true[mask] = val
        # Thin lines
        for _ in range(rng.integers(2, 5)):
            x0, x1 = rng.integers(0, 128, 2)
            y0, y1 = rng.integers(0, 128, 2)
            rr, cc = _draw_line(y0, x0, y1, x1, 128)
            x_true[rr, cc] = rng.uniform(0.5, 0.9)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            tip_radius = float(rng.uniform(1.0, 5.0))
            feedback_noise = float(rng.uniform(0.0, 0.05))
        else:
            tip_radius = 2.0
            feedback_noise = 0.01

        # Tip PSF: Gaussian
        psf_sigma = tip_radius / 2.0
        H_ideal = ndimage.gaussian_filter(x_true, sigma=psf_sigma).astype(np.float32)

        # y: convolved with tip PSF + noise
        y = H_ideal + rng.normal(0, 0.02 + feedback_noise, H_ideal.shape).astype(np.float32)
        y = np.clip(y, 0, 1).astype(np.float32)

        # Reconstruction: Wiener deconvolution
        try:
            recon = wiener(y, mysize=5, noise=0.01).astype(np.float32)
        except Exception:
            recon = ndimage.gaussian_filter(y, sigma=0.5).astype(np.float32)
        mn, mx = recon.min(), recon.max()
        if mx > mn:
            recon = (recon - mn) / (mx - mn)
        recon = recon.astype(np.float32)

        true_specs[f"sample_{i:02d}"] = {
            "tip_radius_px": tip_radius,
            "feedback_noise": feedback_noise,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"tip_radius_px": tip_radius, "feedback_noise": feedback_noise},
        })

    spec = {
        "modality": "nsom",
        "workflow": {"dag": "C → PSF → D", "description": "Near-field scanning optical microscopy"},
        "geometry": {"image_size": 128},
        "mismatch_ranges": {
            "tip_radius_px": {"min": 1.0, "max": 5.0, "unit": "pixels"},
            "feedback_noise": {"min": 0.0, "max": 0.05, "unit": ""},
        },
        "noise_model": {"type": "gaussian_plus_tip_convolution"},
        "reconstruction_hint": "Wiener deconvolution of tip-blurred near-field image.",
    }
    write_tier("nsom", tier, samples, spec, true_specs)


def _draw_line(r0, c0, r1, c1, size):
    """Bresenham line, clipped to [0, size)."""
    rr, cc = [], []
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    for _ in range(max(dr, dc) + 1):
        if 0 <= r < size and 0 <= c < size:
            rr.append(r); cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return np.array(rr, dtype=int), np.array(cc, dtype=int)


# ===========================================================================
# 7. OCEAN_ACOUSTIC_TOMO
# ===========================================================================
def gen_ocean_acoustic_tomo(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: ocean sound speed map (64x64), ~1480-1520 m/s normalized to [0,1]
        x_true = np.zeros((64, 64), dtype=np.float32)
        # Base sound speed gradient (depth stratification)
        for row in range(64):
            x_true[row, :] = rng.uniform(0.3, 0.5) + row / 64 * rng.uniform(0.2, 0.4)
        # Thermal/eddy anomalies
        yy, xx = np.ogrid[:64, :64]
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.integers(10, 54, 2)
            ry, rx = rng.integers(4, 18, 2)
            val = rng.uniform(-0.15, 0.2)
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            x_true[mask] = np.clip(x_true[mask] + val, 0, 1)
        x_true = ndimage.gaussian_filter(x_true, sigma=1.5).astype(np.float32)
        x_true = np.clip(x_true, 0, 1)

        # Mismatch
        if tier == "hidden":
            travel_time_err = float(rng.uniform(-0.02, 0.02))
            multipath_noise = float(rng.uniform(0.0, 0.05))
        else:
            travel_time_err = 0.0
            multipath_noise = 0.01

        # y: travel time measurements (32 source-receiver pairs x 64 range bins)
        # Simplified: project sound slowness along horizontal paths at 32 depths
        slowness = 1.0 / (1480 + x_true * 40)  # actual slowness
        slowness_norm = x_true  # normalized for measurement
        # Integral projections at 32 depths
        y = np.zeros((32, 64), dtype=np.float32)
        depths = np.linspace(0, 63, 32).astype(int)
        for k, d in enumerate(depths):
            y[k, :] = slowness_norm[d, :]
        y += travel_time_err + rng.normal(0, 0.01 + multipath_noise, y.shape).astype(np.float32)

        H_ideal = np.zeros((32, 64), dtype=np.float32)
        for k, d in enumerate(depths):
            H_ideal[k, :] = slowness_norm[d, :]

        # Reconstruction: back-project (simple transpose/smear)
        recon = np.zeros((64, 64), dtype=np.float32)
        for k, d in enumerate(depths):
            recon[d, :] = y[k, :]
        # Interpolate missing rows
        for row in range(64):
            if recon[row, :].sum() == 0:
                neighbors = [r for r in range(max(0, row-2), min(64, row+3)) if r != row and recon[r, :].sum() != 0]
                if neighbors:
                    recon[row, :] = np.mean([recon[r, :] for r in neighbors], axis=0)
        recon = ndimage.gaussian_filter(recon, sigma=1.0).astype(np.float32)
        mn, mx = recon.min(), recon.max()
        if mx > mn:
            recon = (recon - mn) / (mx - mn)

        true_specs[f"sample_{i:02d}"] = {
            "travel_time_err": travel_time_err,
            "multipath_noise": multipath_noise,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"travel_time_err": travel_time_err, "multipath_noise": multipath_noise},
        })

    spec = {
        "modality": "ocean_acoustic_tomo",
        "workflow": {"dag": "Source → P → D", "description": "Ocean acoustic tomography"},
        "geometry": {"x_size": 64, "y_size_32x64": "32 depths x 64 range"},
        "mismatch_ranges": {
            "travel_time_err": {"min": -0.02, "max": 0.02, "unit": "s"},
            "multipath_noise": {"min": 0.0, "max": 0.05, "unit": ""},
        },
        "noise_model": {"type": "gaussian_travel_time"},
        "reconstruction_hint": "Back-project travel-time measurements to 2D sound-speed map.",
    }
    write_tier("ocean_acoustic_tomo", tier, samples, spec, true_specs)


# ===========================================================================
# 8. OCEAN_COLOR
# ===========================================================================
def gen_ocean_color(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: 4-band ocean color (128x128x4) — blue, green, red, NIR
        x_true = np.zeros((128, 128, 4), dtype=np.float32)
        # Band-dependent ocean signals
        wavelengths = [443, 550, 665, 865]  # nm
        water_base = [0.6, 0.4, 0.1, 0.02]  # blue absorbs less, NIR absorbs most

        for b, (wl, wb) in enumerate(zip(wavelengths, water_base)):
            # Base ocean radiance
            band = np.full((128, 128), wb, dtype=np.float32)
            # Chlorophyll blooms
            yy, xx = np.ogrid[:128, :128]
            for _ in range(rng.integers(1, 4)):
                cy, cx = rng.integers(20, 108, 2)
                r = rng.integers(10, 35)
                bloom = rng.uniform(0.05, 0.3) * (b == 1 or b == 0)  # green/blue bloom
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                band[mask] += bloom
            # Sediment plume
            cx_sed = rng.integers(20, 108)
            band[:, max(0, cx_sed-10):min(128, cx_sed+10)] += rng.uniform(0.02, 0.1) * (b >= 1)
            x_true[:, :, b] = np.clip(ndimage.gaussian_filter(band, sigma=2), 0, 1)

        # Mismatch
        if tier == "hidden":
            haze_strength = float(rng.uniform(0.05, 0.3))
            sun_glint = float(rng.uniform(0.0, 0.2))
        else:
            haze_strength = 0.1
            sun_glint = 0.0

        # y: top-of-atmosphere radiance with atmospheric haze
        # Atmospheric contribution (Rayleigh + aerosol)
        atm_base = np.array([0.15, 0.08, 0.04, 0.02], dtype=np.float32)  # per band
        y = np.zeros_like(x_true)
        for b in range(4):
            haze = haze_strength * atm_base[b]
            y[:, :, b] = x_true[:, :, b] + haze + sun_glint * rng.uniform(0, 0.5) * (b >= 2)
            y[:, :, b] += rng.normal(0, 0.005, (128, 128)).astype(np.float32)
        y = np.clip(y, 0, 1).astype(np.float32)

        H_ideal = x_true.copy()

        # Reconstruction: atmospheric correction (subtract haze estimate)
        recon = np.zeros_like(y)
        for b in range(4):
            haze_est = haze_strength * atm_base[b] * 0.9  # slightly underestimate
            recon[:, :, b] = np.clip(y[:, :, b] - haze_est, 0, 1)

        true_specs[f"sample_{i:02d}"] = {
            "haze_strength": haze_strength,
            "sun_glint": sun_glint,
            "wavelengths_nm": wavelengths,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"haze_strength": haze_strength, "sun_glint": sun_glint},
        })

    spec = {
        "modality": "ocean_color",
        "workflow": {"dag": "M → Sigma → D", "description": "Ocean color remote sensing"},
        "geometry": {"image_size": 128, "n_bands": 4, "wavelengths_nm": [443, 550, 665, 865]},
        "mismatch_ranges": {
            "haze_strength": {"min": 0.05, "max": 0.3, "unit": ""},
            "sun_glint": {"min": 0.0, "max": 0.2, "unit": ""},
        },
        "noise_model": {"type": "atmospheric_additive_plus_gaussian"},
        "reconstruction_hint": "Subtract atmospheric haze estimate per band.",
    }
    write_tier("ocean_color", tier, samples, spec, true_specs)


# ===========================================================================
# 9. OCTA
# ===========================================================================
def gen_octa(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: retinal vasculature (128x128) — vessel map
        x_true = np.zeros((128, 128), dtype=np.float32)
        # Main vessels radiating from optic disc
        oc_y, oc_x = 64, 64
        for branch in range(rng.integers(4, 8)):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(25, 55)
            width = rng.integers(1, 4)
            ys_path, xs_path = [], []
            y_curr, x_curr = oc_y, oc_x
            for step in range(length):
                angle += rng.normal(0, 0.15)
                y_curr = int(np.clip(y_curr + np.sin(angle), 0, 127))
                x_curr = int(np.clip(x_curr + np.cos(angle), 0, 127))
                ys_path.append(y_curr); xs_path.append(x_curr)
            for y_v, x_v in zip(ys_path, xs_path):
                yy, xx = np.ogrid[:128, :128]
                mask = (yy - y_v) ** 2 + (xx - x_v) ** 2 <= width ** 2
                x_true[mask] = rng.uniform(0.7, 1.0)
        # Capillary network (fine vessels)
        cap_map = rng.uniform(0, 1, (128, 128)) > 0.92
        x_true[cap_map] = rng.uniform(0.3, 0.6)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            speckle_k = float(rng.uniform(1.0, 3.0))
            background_level = float(rng.uniform(0.05, 0.2))
        else:
            speckle_k = 1.5
            background_level = 0.08

        # y: x_true with speckle noise + background (simulate OCT angiogram)
        # Speckle: multiplicative Rayleigh noise
        speckle = rng.rayleigh(scale=speckle_k / 4.0, size=(128, 128)).astype(np.float32)
        speckle = speckle / (speckle.mean() + 1e-8)
        y = x_true * speckle + background_level + rng.normal(0, 0.02, (128, 128)).astype(np.float32)
        y = np.clip(y, 0, None).astype(np.float32)

        # H_ideal: clean signal without background
        H_ideal = (x_true * (speckle.mean())).astype(np.float32)

        # Reconstruction: speckle variance map (threshold + median filter)
        y_med = ndimage.median_filter(y, size=3)
        recon = np.clip(y_med - background_level, 0, None)
        # Threshold to enhance vessels
        thresh = np.percentile(recon, 60)
        recon = np.where(recon > thresh, recon, recon * 0.3)
        mn, mx = recon.min(), recon.max()
        if mx > mn:
            recon = (recon - mn) / (mx - mn)
        recon = recon.astype(np.float32)

        true_specs[f"sample_{i:02d}"] = {
            "speckle_k": speckle_k,
            "background_level": background_level,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"speckle_k": speckle_k, "background_level": background_level},
        })

    spec = {
        "modality": "octa",
        "workflow": {"dag": "C → PSF → D", "description": "Optical coherence tomography angiography"},
        "geometry": {"image_size": 128},
        "mismatch_ranges": {
            "speckle_k": {"min": 1.0, "max": 3.0, "unit": ""},
            "background_level": {"min": 0.05, "max": 0.2, "unit": ""},
        },
        "noise_model": {"type": "rayleigh_speckle_plus_gaussian"},
        "reconstruction_hint": "Median filter and threshold to extract vessel map from speckle-corrupted angiogram.",
    }
    write_tier("octa", tier, samples, spec, true_specs)


# ===========================================================================
# 10. PANORAMA
# ===========================================================================
def gen_panorama(tier: str) -> None:
    cfg = TIERS[tier]
    samples, true_specs = [], {}

    for i in range(cfg["n"]):
        rng = np.random.default_rng(seed_for(tier, i))

        # x_true: wide panoramic image (256x512)
        x_true = np.zeros((256, 512), dtype=np.float32)
        # Sky gradient (top half)
        for row in range(128):
            x_true[row, :] = rng.uniform(0.5, 0.9) - row / 256.0 * 0.3
        # Ground (bottom half)
        for row in range(128, 256):
            x_true[row, :] = rng.uniform(0.2, 0.5) + (row - 128) / 128.0 * rng.uniform(0, 0.2)
        # Scene elements (buildings/trees)
        yy, xx = np.mgrid[:256, :512]
        for _ in range(rng.integers(5, 15)):
            cx = rng.integers(20, 492)
            width = rng.integers(10, 50)
            height = rng.integers(30, 120)
            cy = 256 - height // 2
            val = rng.uniform(0.2, 0.7)
            mask = (np.abs(xx - cx) <= width // 2) & (yy >= 256 - height)
            x_true[mask] = val
        # Texture
        x_true += rng.normal(0, 0.02, x_true.shape).astype(np.float32)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Mismatch
        if tier == "hidden":
            max_misalign = float(rng.uniform(1.0, 5.0))
            exposure_var = float(rng.uniform(0.0, 0.2))
        else:
            max_misalign = 2.0
            exposure_var = 0.05

        # y: 8 overlapping patches (128x128) with slight misalignment
        # Patch positions (overlapping tiles across 512 width)
        n_patches = 8
        patch_h, patch_w = 128, 128
        overlap = (n_patches * patch_w - 512) // (n_patches - 1)  # ~18px
        patch_x_starts = [int(k * (patch_w - overlap)) for k in range(n_patches)]
        # Clamp last start
        patch_x_starts[-1] = 512 - patch_w

        y = np.zeros((n_patches, patch_h, patch_w), dtype=np.float32)
        patch_offsets = []
        for k, x_start in enumerate(patch_x_starts):
            dy = int(rng.integers(-int(max_misalign), int(max_misalign) + 1))
            dx = int(rng.integers(-int(max_misalign), int(max_misalign) + 1))
            y_start = 64 + dy  # center row +/- misalignment
            x_start_d = x_start + dx  # horizontal misalignment
            # Source region in panorama (clamp to valid range)
            y_s = max(0, y_start)
            y_e = min(256, y_start + patch_h)
            x_s = max(0, x_start_d)
            x_e = min(512, x_start_d + patch_w)
            # Destination region in patch
            dst_y0 = max(0, -y_start)
            dst_x0 = max(0, -x_start_d)
            ph = min(y_e - y_s, patch_h - dst_y0)
            pw = min(x_e - x_s, patch_w - dst_x0)
            patch = np.zeros((patch_h, patch_w), dtype=np.float32)
            if ph > 0 and pw > 0:
                patch[dst_y0:dst_y0+ph, dst_x0:dst_x0+pw] = x_true[y_s:y_s+ph, x_s:x_s+pw]
            # Exposure variation
            exp_factor = 1.0 + exposure_var * rng.uniform(-1, 1)
            patch = np.clip(patch * exp_factor, 0, 1)
            patch += rng.normal(0, 0.01, patch.shape).astype(np.float32)
            y[k] = np.clip(patch, 0, 1).astype(np.float32)
            patch_offsets.append({"dy": dy, "dx": dx, "x_start": x_start})

        # H_ideal: patches without misalignment or exposure variation
        H_ideal = np.zeros((n_patches, patch_h, patch_w), dtype=np.float32)
        for k, x_start in enumerate(patch_x_starts):
            H_ideal[k] = x_true[64:192, x_start:x_start + patch_w]

        # Reconstruction: average of patches projected back to panorama
        recon = np.zeros((256, 512), dtype=np.float32)
        counts = np.zeros((256, 512), dtype=np.float32)
        for k, x_start in enumerate(patch_x_starts):
            recon[64:192, x_start:x_start + patch_w] += y[k]
            counts[64:192, x_start:x_start + patch_w] += 1
        valid = counts > 0
        recon[valid] /= counts[valid]
        # Fill top/bottom from x_true mean
        recon[:64, :] = x_true[:64, :].mean()
        recon[192:, :] = x_true[192:, :].mean()
        mn, mx = recon.min(), recon.max()
        if mx > mn:
            recon = (recon - mn) / (mx - mn)
        recon = recon.astype(np.float32)

        true_specs[f"sample_{i:02d}"] = {
            "max_misalign_px": max_misalign,
            "exposure_variation": exposure_var,
            "patch_offsets": patch_offsets,
            "note": "True forward model parameters.",
        }
        samples.append({
            "x_true": x_true, "y": y, "H_ideal": H_ideal,
            "reconstruction_baseline": recon,
            "mismatch_params": {"max_misalign_px": max_misalign, "exposure_variation": exposure_var},
        })

    spec = {
        "modality": "panorama",
        "workflow": {"dag": "C → Warp → D", "description": "Panoramic image stitching"},
        "geometry": {
            "panorama_shape": [256, 512],
            "n_patches": 8,
            "patch_shape": [128, 128],
        },
        "mismatch_ranges": {
            "max_misalign_px": {"min": 1.0, "max": 5.0, "unit": "pixels"},
            "exposure_variation": {"min": 0.0, "max": 0.2, "unit": ""},
        },
        "noise_model": {"type": "gaussian_plus_misalignment"},
        "reconstruction_hint": "Average overlapping patches projected to full panorama.",
    }
    write_tier("panorama", tier, samples, spec, true_specs)


# ===========================================================================
# Main
# ===========================================================================
GENERATORS = {
    "minflux":              gen_minflux,
    "muon_tomo":            gen_muon_tomo,
    "neutron_diffraction":  gen_neutron_diffraction,
    "neutron_tomo":         gen_neutron_tomo,
    "nirs_brain":           gen_nirs_brain,
    "nsom":                 gen_nsom,
    "ocean_acoustic_tomo":  gen_ocean_acoustic_tomo,
    "ocean_color":          gen_ocean_color,
    "octa":                 gen_octa,
    "panorama":             gen_panorama,
}

if __name__ == "__main__":
    print(f"Output root: {ROOT}")
    os.makedirs(ROOT, exist_ok=True)

    for modality, gen_fn in GENERATORS.items():
        print(f"\n{'='*60}")
        print(f"Generating: {modality}")
        print(f"{'='*60}")
        for tier in ["public", "dev", "hidden"]:
            gen_fn(tier)

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
