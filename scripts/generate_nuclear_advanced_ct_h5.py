#!/usr/bin/env python3
"""
Generate benchmark datasets for 5 nuclear and advanced CT modalities:
  1. pet_ct        — combined PET/CT
  2. pet_mr        — combined PET/MRI
  3. spect_ct      — SPECT-CT
  4. spectral_ct   — dual-energy spectral CT
  5. industrial_ct — industrial CT (metal + polymer phantom)

Output layout (per modality):
  datasets/benchmark/{modality}/{tier}/{modality}_challenge_{tier}.h5
  datasets/benchmark/{modality}/{tier}/spec.json
  datasets/benchmark/{modality}/{tier}/true_spec.json
  datasets/benchmark/{modality}/{tier}/images/sample_NN/*.png

HDF5 schema per sample group:
  x_true                — float32 ground truth
  y                     — float32 forward measurement
  H_ideal               — float32 ideal operator (angles or sinogram)
  reconstruction_baseline — float32 FBP / back-projection result

Tiers: public=12, dev=20, hidden=20 samples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

try:
    from skimage.transform import radon, iradon
    from skimage.data import shepp_logan_phantom
    HAS_SKIMAGE = True
except ImportError:
    raise ImportError("pip install scikit-image")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "datasets" / "benchmark"

TIERS = {
    "public": 12,
    "dev":    20,
    "hidden": 20,
}

RNG_SEED = 2024

# ---------------------------------------------------------------------------
# Shared phantom generators
# ---------------------------------------------------------------------------

def make_shepp_logan(size: int = 256, rng: np.random.Generator = None) -> np.ndarray:
    """Shepp-Logan phantom randomised with extra ellipses, normalised [0,1]."""
    if rng is None:
        rng = np.random.default_rng()
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    base = shepp_logan_phantom()
    x = resize(base, (size, size), anti_aliasing=True).astype(np.float32)
    n_extra = rng.integers(2, 6)
    Y, X = np.ogrid[:size, :size]
    cx = cy = size // 2
    for _ in range(n_extra):
        ey = rng.integers(8, 50)
        ex = rng.integers(8, 50)
        y0 = int(rng.integers(cy - 70, cy + 70))
        x0 = int(rng.integers(cx - 70, cx + 70))
        val = float(rng.uniform(0.05, 0.35))
        mask = ((X - x0) / ex) ** 2 + ((Y - y0) / ey) ** 2 <= 1
        x[mask] += val
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def make_pet_activity(size: int = 256, rng: np.random.Generator = None) -> np.ndarray:
    """PET activity map: smooth blobs representing tracer uptake."""
    if rng is None:
        rng = np.random.default_rng()
    img = np.zeros((size, size), dtype=np.float32)
    n_blobs = rng.integers(4, 10)
    Y, X = np.ogrid[:size, :size]
    for _ in range(n_blobs):
        cx = int(rng.integers(30, size - 30))
        cy = int(rng.integers(30, size - 30))
        r = rng.uniform(8.0, 40.0)
        amp = rng.uniform(0.2, 1.0)
        blob = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * r**2))
        img += blob
    img = np.clip(img, 0.0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def make_industrial_phantom(size: int = 256, rng: np.random.Generator = None) -> np.ndarray:
    """Industrial object: polymer matrix (low attenuation) with metal inclusions (high)."""
    if rng is None:
        rng = np.random.default_rng()
    img = np.zeros((size, size), dtype=np.float32)
    Y, X = np.ogrid[:size, :size]
    cx = cy = size // 2
    # Outer polymer shell
    r_out = int(rng.integers(90, 115))
    r_in  = int(rng.integers(60, 85))
    poly_val = float(rng.uniform(0.1, 0.25))
    ring = (((X - cx)**2 + (Y - cy)**2) <= r_out**2) & \
           (((X - cx)**2 + (Y - cy)**2) >= r_in**2)
    img[ring] = poly_val
    # Inner air / low-density fill
    air_val = float(rng.uniform(0.0, 0.05))
    img[((X - cx)**2 + (Y - cy)**2) < r_in**2] = air_val
    # Metal inclusions (high attenuation)
    n_metals = rng.integers(3, 8)
    for _ in range(n_metals):
        mx = int(rng.integers(cx - r_in + 10, cx + r_in - 10))
        my = int(rng.integers(cy - r_in + 10, cy + r_in - 10))
        mr = rng.uniform(3.0, 12.0)
        metal_val = float(rng.uniform(0.6, 1.0))
        mask = ((X - mx)**2 + (Y - my)**2) <= mr**2
        img[mask] = metal_val
    # Add sharp cracks / features
    n_cracks = rng.integers(0, 3)
    for _ in range(n_cracks):
        x0 = int(rng.integers(cx - 60, cx + 60))
        y0 = int(rng.integers(cy - 60, cy + 60))
        length = rng.integers(10, 40)
        angle = rng.uniform(0, np.pi)
        for t in np.linspace(0, length, 80):
            xi = int(x0 + t * np.cos(angle))
            yi = int(y0 + t * np.sin(angle))
            if 0 <= xi < size and 0 <= yi < size:
                img[yi, xi] = float(rng.uniform(0.0, 0.02))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model helpers
# ---------------------------------------------------------------------------

def radon_transform(x: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Radon transform -> (n_angles, n_detectors) float32."""
    sino = radon(x, theta=angles_deg, circle=True)  # (n_det, n_angles)
    return sino.T.astype(np.float32)                 # (n_angles, n_det)


def add_poisson_noise(sino: np.ndarray, photon_count: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Apply Beer-Lambert Poisson noise to a sinogram."""
    I = photon_count * np.exp(-sino.astype(np.float64))
    I_noisy = rng.poisson(np.maximum(I, 1e-9)).astype(np.float64)
    y = -np.log(np.maximum(I_noisy, 1.0) / photon_count)
    return y.astype(np.float32)


def fbp_reconstruct(sino: np.ndarray, angles_deg: np.ndarray,
                    size: int = 256) -> np.ndarray:
    """FBP reconstruction via iradon (Ram-Lak filter)."""
    # iradon expects (n_det, n_angles)
    recon = iradon(sino.T, theta=angles_deg, output_size=size,
                   filter_name='ramp', circle=True)
    return recon.astype(np.float32)


def save_png(arr: np.ndarray, path: Path, label: str = "") -> None:
    """Normalise array to uint8 and save as PNG."""
    if not HAS_PIL:
        return
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    img_u8 = (a * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(str(path))


# ---------------------------------------------------------------------------
# Spec ranges (mismatch parameters shown in spec.json, hidden in true_spec.json)
# ---------------------------------------------------------------------------

SPEC_RANGES = {
    "pet_ct": {
        "ct_dose_gy":         (0.001, 0.05),
        "pet_activity_MBq":   (100.0, 400.0),
        "axial_fov_cm":       (15.0,  25.0),
    },
    "pet_mr": {
        "B0_Tesla":           (3.0,   7.0),
        "pet_sensitivity_pct":(3.0,   20.0),
        "mri_bandwidth_hz":   (125e3, 250e3),
    },
    "spect_ct": {
        "energy_keV":         (140.0, 511.0),
        "window_percent":     (10.0,  20.0),
        "rotation_radius_cm": (15.0,  25.0),
    },
    "spectral_ct": {
        "kVp_low":            (70.0,  100.0),
        "kVp_high":           (120.0, 150.0),
        "dose_split_percent": (50.0,  70.0),
    },
    "industrial_ct": {
        "voltage_kV":         (100.0, 450.0),
        "current_uA":         (50.0,  500.0),
        "detector_pitch_mm":  (0.1,   1.0),
    },
}


def sample_spec(modality: str, rng: np.random.Generator) -> dict:
    out = {}
    for k, (lo, hi) in SPEC_RANGES[modality].items():
        out[k] = float(rng.uniform(lo, hi))
    return out


# ===========================================================================
# Modality generators
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. PET-CT
# ---------------------------------------------------------------------------

def generate_pet_ct(out_dir: Path, rng: np.random.Generator) -> None:
    """
    x_true:  (2, 256, 256) float32  ch0=CT attenuation, ch1=PET activity
    y:       (188, 256)    float32  concatenated sinograms [CT(60), PET(128)]
    H_ideal: (188,)        float32  angle array
    recon:   (256, 256)    float32  average of FBP(CT) and FBP(PET)
    """
    modality = "pet_ct"
    angles_ct  = np.linspace(0, 180, 60,  endpoint=False)
    angles_pet = np.linspace(0, 180, 128, endpoint=False)
    angles_all = np.concatenate([angles_ct, angles_pet]).astype(np.float32)

    photon_ct  = 1e4
    photon_pet = 5e3

    for tier, n_samples in TIERS.items():
        tier_dir = out_dir / modality / tier
        img_dir  = tier_dir / "images"
        tier_dir.mkdir(parents=True, exist_ok=True)

        true_spec_list = []
        spec_list      = []

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({"variant": modality, "version": "1.0", "tier": tier,
                             "n_ct_angles": 60, "n_pet_angles": 128,
                             "img_size": 256})

            for i in range(n_samples):
                grp = hf.create_group(f"sample_{i:02d}")

                # Ground truth
                ct_map  = make_shepp_logan(256, rng)
                pet_map = make_pet_activity(256, rng)
                x_true  = np.stack([ct_map, pet_map], axis=0)  # (2,256,256)

                # Forward: CT sinogram
                sino_ct_ideal = radon_transform(ct_map,  angles_ct)   # (60, 256)
                sino_pet_ideal= radon_transform(pet_map, angles_pet)  # (128, 256)

                sino_ct_noisy  = add_poisson_noise(sino_ct_ideal,  photon_ct,  rng)
                sino_pet_noisy = add_poisson_noise(sino_pet_ideal, photon_pet, rng)

                y = np.concatenate([sino_ct_noisy, sino_pet_noisy], axis=0)  # (188,256)

                # Baseline: FBP each part, average
                recon_ct  = fbp_reconstruct(sino_ct_noisy,  angles_ct,  256)
                recon_pet = fbp_reconstruct(sino_pet_noisy, angles_pet, 256)
                recon_baseline = np.stack([recon_ct, recon_pet]).mean(axis=0)  # (256,256)

                # Write HDF5
                grp.create_dataset("x_true",                 data=x_true,          compression="gzip")
                grp.create_dataset("y",                      data=y,               compression="gzip")
                grp.create_dataset("H_ideal",                data=angles_all,      compression="gzip")
                grp.create_dataset("reconstruction_baseline",data=recon_baseline,  compression="gzip")

                # Spec
                sp = sample_spec(modality, rng)
                true_spec_list.append({"sample": f"sample_{i:02d}", **sp})
                spec_list.append({"sample": f"sample_{i:02d}",
                                  **{k: None for k in sp}})
                grp.attrs["spec"] = json.dumps(sp)

                # Images
                sdir = img_dir / f"sample_{i:02d}"
                sdir.mkdir(parents=True, exist_ok=True)
                save_png(x_true[0],       sdir / "x_true_ch0_ct.png")
                save_png(x_true[1],       sdir / "x_true_ch1_pet.png")
                save_png(y,               sdir / "y_sinogram.png")
                save_png(recon_baseline,  sdir / "reconstruction_baseline.png")

                if (i + 1) % 5 == 0 or i == n_samples - 1:
                    print(f"  [{modality}][{tier}] {i+1}/{n_samples}")

        # Write JSON specs
        (tier_dir / "spec.json").write_text(json.dumps(spec_list, indent=2))
        (tier_dir / "true_spec.json").write_text(json.dumps(true_spec_list, indent=2))
        print(f"  -> {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2. PET-MR
# ---------------------------------------------------------------------------

def generate_pet_mr(out_dir: Path, rng: np.random.Generator) -> None:
    """
    x_true:  (256, 256)  float32  PET activity map
    y:       (128, 256)  float32  PET sinogram (Poisson-noisy)
    H_ideal: (128,)      float32  projection angles
    recon:   (256, 256)  float32  FBP
    """
    modality  = "pet_mr"
    angles    = np.linspace(0, 180, 128, endpoint=False).astype(np.float32)
    photon_count = 5e3

    for tier, n_samples in TIERS.items():
        tier_dir = out_dir / modality / tier
        img_dir  = tier_dir / "images"
        tier_dir.mkdir(parents=True, exist_ok=True)

        true_spec_list = []
        spec_list      = []

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({"variant": modality, "version": "1.0", "tier": tier,
                             "n_angles": 128, "img_size": 256})

            for i in range(n_samples):
                grp = hf.create_group(f"sample_{i:02d}")

                x_true = make_pet_activity(256, rng)             # (256,256)
                sino_ideal = radon_transform(x_true, angles)     # (128,256)
                y = add_poisson_noise(sino_ideal, photon_count, rng)
                recon_baseline = fbp_reconstruct(y, angles, 256) # (256,256)

                grp.create_dataset("x_true",                 data=x_true,         compression="gzip")
                grp.create_dataset("y",                      data=y,              compression="gzip")
                grp.create_dataset("H_ideal",                data=angles,         compression="gzip")
                grp.create_dataset("reconstruction_baseline",data=recon_baseline, compression="gzip")

                sp = sample_spec(modality, rng)
                true_spec_list.append({"sample": f"sample_{i:02d}", **sp})
                spec_list.append({"sample": f"sample_{i:02d}",
                                  **{k: None for k in sp}})
                grp.attrs["spec"] = json.dumps(sp)

                sdir = img_dir / f"sample_{i:02d}"
                sdir.mkdir(parents=True, exist_ok=True)
                save_png(x_true,         sdir / "x_true.png")
                save_png(y,              sdir / "y_sinogram.png")
                save_png(recon_baseline, sdir / "reconstruction_baseline.png")

                if (i + 1) % 5 == 0 or i == n_samples - 1:
                    print(f"  [{modality}][{tier}] {i+1}/{n_samples}")

        (tier_dir / "spec.json").write_text(json.dumps(spec_list, indent=2))
        (tier_dir / "true_spec.json").write_text(json.dumps(true_spec_list, indent=2))
        print(f"  -> {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 3. SPECT-CT
# ---------------------------------------------------------------------------

def generate_spect_ct(out_dir: Path, rng: np.random.Generator) -> None:
    """
    x_true:  (256, 256)  float32  SPECT activity map
    y:       (256, 256)  float32  attenuated sinogram with Poisson noise
    H_ideal: (256,)      float32  projection angles
    recon:   (256, 256)  float32  FBP
    """
    modality = "spect_ct"
    angles   = np.linspace(0, 360, 256, endpoint=False).astype(np.float32)
    photon_count = 2e3

    for tier, n_samples in TIERS.items():
        tier_dir = out_dir / modality / tier
        img_dir  = tier_dir / "images"
        tier_dir.mkdir(parents=True, exist_ok=True)

        true_spec_list = []
        spec_list      = []

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({"variant": modality, "version": "1.0", "tier": tier,
                             "n_angles": 256, "img_size": 256})

            for i in range(n_samples):
                grp = hf.create_group(f"sample_{i:02d}")

                # SPECT activity + CT-based attenuation map
                activity = make_pet_activity(256, rng)
                mu_map   = make_shepp_logan(256, rng) * 0.15  # attenuation coefficients

                # Attenuated Radon: apply depth-dependent attenuation weight
                # Simplified: multiply activity by exp(-mean_mu * half-width)
                mean_mu = float(mu_map.mean())
                half_width = 128.0  # pixels
                atten_factor = float(np.exp(-mean_mu * half_width * 0.5))
                act_attenuated = (activity * atten_factor).astype(np.float32)

                sino_ideal = radon_transform(act_attenuated, angles)   # (256,256)
                y = add_poisson_noise(sino_ideal, photon_count, rng)
                recon_baseline = fbp_reconstruct(y, angles, 256)

                grp.create_dataset("x_true",                 data=activity,       compression="gzip")
                grp.create_dataset("y",                      data=y,              compression="gzip")
                grp.create_dataset("H_ideal",                data=angles,         compression="gzip")
                grp.create_dataset("reconstruction_baseline",data=recon_baseline, compression="gzip")

                sp = sample_spec(modality, rng)
                true_spec_list.append({"sample": f"sample_{i:02d}", **sp})
                spec_list.append({"sample": f"sample_{i:02d}",
                                  **{k: None for k in sp}})
                grp.attrs["spec"] = json.dumps(sp)

                sdir = img_dir / f"sample_{i:02d}"
                sdir.mkdir(parents=True, exist_ok=True)
                save_png(activity,       sdir / "x_true.png")
                save_png(y,              sdir / "y_sinogram.png")
                save_png(recon_baseline, sdir / "reconstruction_baseline.png")

                if (i + 1) % 5 == 0 or i == n_samples - 1:
                    print(f"  [{modality}][{tier}] {i+1}/{n_samples}")

        (tier_dir / "spec.json").write_text(json.dumps(spec_list, indent=2))
        (tier_dir / "true_spec.json").write_text(json.dumps(true_spec_list, indent=2))
        print(f"  -> {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 4. Spectral CT (dual-energy)
# ---------------------------------------------------------------------------

def generate_spectral_ct(out_dir: Path, rng: np.random.Generator) -> None:
    """
    x_true:  (2, 256, 256)  float32  [soft_tissue_density, bone_density]
    y:       (2, 60, 256)   float32  dual-energy sinograms
    H_ideal: (60,)          float32  projection angles
    recon:   (256, 256)     float32  FBP of averaged sinogram
    """
    modality = "spectral_ct"
    angles   = np.linspace(0, 180, 60, endpoint=False).astype(np.float32)
    photon_count = 5e4

    for tier, n_samples in TIERS.items():
        tier_dir = out_dir / modality / tier
        img_dir  = tier_dir / "images"
        tier_dir.mkdir(parents=True, exist_ok=True)

        true_spec_list = []
        spec_list      = []

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({"variant": modality, "version": "1.0", "tier": tier,
                             "n_angles": 60, "img_size": 256,
                             "energy_bins": 2})

            for i in range(n_samples):
                grp = hf.create_group(f"sample_{i:02d}")

                soft_tissue = make_shepp_logan(256, rng)
                bone        = make_shepp_logan(256, rng) * 0.6  # bone is denser

                x_true = np.stack([soft_tissue, bone], axis=0)  # (2,256,256)

                # Two-material mixture per energy bin
                eff_high = 0.8 * soft_tissue + 0.3 * bone
                eff_low  = 0.5 * soft_tissue + 0.9 * bone

                sino_high_ideal = radon_transform(eff_high, angles)  # (60,256)
                sino_low_ideal  = radon_transform(eff_low,  angles)  # (60,256)

                sino_high = add_poisson_noise(sino_high_ideal, photon_count, rng)
                sino_low  = add_poisson_noise(sino_low_ideal,  photon_count, rng)

                y = np.stack([sino_high, sino_low], axis=0)  # (2,60,256)

                # Baseline: FBP of averaged sinogram
                sino_avg = (sino_high + sino_low) / 2.0
                recon_baseline = fbp_reconstruct(sino_avg, angles, 256)

                grp.create_dataset("x_true",                 data=x_true,         compression="gzip")
                grp.create_dataset("y",                      data=y,              compression="gzip")
                grp.create_dataset("H_ideal",                data=angles,         compression="gzip")
                grp.create_dataset("reconstruction_baseline",data=recon_baseline, compression="gzip")

                sp = sample_spec(modality, rng)
                true_spec_list.append({"sample": f"sample_{i:02d}", **sp})
                spec_list.append({"sample": f"sample_{i:02d}",
                                  **{k: None for k in sp}})
                grp.attrs["spec"] = json.dumps(sp)

                sdir = img_dir / f"sample_{i:02d}"
                sdir.mkdir(parents=True, exist_ok=True)
                save_png(x_true[0],      sdir / "x_true_ch0_soft_tissue.png")
                save_png(x_true[1],      sdir / "x_true_ch1_bone.png")
                save_png(y[0],           sdir / "y_sino_high_energy.png")
                save_png(y[1],           sdir / "y_sino_low_energy.png")
                save_png(recon_baseline, sdir / "reconstruction_baseline.png")

                if (i + 1) % 5 == 0 or i == n_samples - 1:
                    print(f"  [{modality}][{tier}] {i+1}/{n_samples}")

        (tier_dir / "spec.json").write_text(json.dumps(spec_list, indent=2))
        (tier_dir / "true_spec.json").write_text(json.dumps(true_spec_list, indent=2))
        print(f"  -> {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 5. Industrial CT
# ---------------------------------------------------------------------------

def generate_industrial_ct(out_dir: Path, rng: np.random.Generator) -> None:
    """
    x_true:  (256, 256)   float32  industrial object (metal + polymer)
    y:       (360, 512)   float32  log-corrected sinogram (360 angles, 512 det)
    H_ideal: (360,)       float32  projection angles
    recon:   (256, 256)   float32  FBP
    """
    modality  = "industrial_ct"
    n_angles  = 360
    n_det     = 512
    angles    = np.linspace(0, 360, n_angles, endpoint=False).astype(np.float32)
    photon_count = 1e5

    for tier, n_samples in TIERS.items():
        tier_dir = out_dir / modality / tier
        img_dir  = tier_dir / "images"
        tier_dir.mkdir(parents=True, exist_ok=True)

        true_spec_list = []
        spec_list      = []

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({"variant": modality, "version": "1.0", "tier": tier,
                             "n_angles": n_angles, "n_detectors": n_det,
                             "img_size": 256})

            for i in range(n_samples):
                grp = hf.create_group(f"sample_{i:02d}")

                x_true = make_industrial_phantom(256, rng)

                # Radon -> (n_angles, n_det_native)
                sino_ideal = radon_transform(x_true, angles)  # (360, 256)

                # Pad/crop to 512 detectors
                n_native = sino_ideal.shape[1]
                if n_native < n_det:
                    pad = n_det - n_native
                    sino_ideal = np.pad(sino_ideal, ((0,0),(pad//2, pad - pad//2)))
                else:
                    trim = (n_native - n_det) // 2
                    sino_ideal = sino_ideal[:, trim:trim+n_det]

                # Beer-Lambert Poisson noise + log correction
                I = photon_count * np.exp(-sino_ideal.astype(np.float64))
                I_noisy = rng.poisson(np.maximum(I, 1e-9)).astype(np.float64)
                y = -np.log(np.maximum(I_noisy, 1.0) / photon_count).astype(np.float32)

                # FBP baseline (trim back to 256 for recon)
                sino_for_recon = y[:, (n_det - 256)//2 : (n_det - 256)//2 + 256]
                recon_baseline = fbp_reconstruct(sino_for_recon, angles, 256)

                grp.create_dataset("x_true",                 data=x_true,         compression="gzip")
                grp.create_dataset("y",                      data=y,              compression="gzip")
                grp.create_dataset("H_ideal",                data=angles,         compression="gzip")
                grp.create_dataset("reconstruction_baseline",data=recon_baseline, compression="gzip")

                sp = sample_spec(modality, rng)
                true_spec_list.append({"sample": f"sample_{i:02d}", **sp})
                spec_list.append({"sample": f"sample_{i:02d}",
                                  **{k: None for k in sp}})
                grp.attrs["spec"] = json.dumps(sp)

                sdir = img_dir / f"sample_{i:02d}"
                sdir.mkdir(parents=True, exist_ok=True)
                save_png(x_true,         sdir / "x_true.png")
                save_png(y,              sdir / "y_sinogram.png")
                save_png(recon_baseline, sdir / "reconstruction_baseline.png")

                if (i + 1) % 5 == 0 or i == n_samples - 1:
                    print(f"  [{modality}][{tier}] {i+1}/{n_samples}")

        (tier_dir / "spec.json").write_text(json.dumps(spec_list, indent=2))
        (tier_dir / "true_spec.json").write_text(json.dumps(true_spec_list, indent=2))
        print(f"  -> {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    rng = np.random.default_rng(RNG_SEED)
    out_dir = OUT_BASE

    print("=" * 60)
    print("Generating nuclear + advanced CT benchmark datasets")
    print(f"Output: {out_dir}")
    print("=" * 60)

    print("\n[1/5] PET-CT")
    generate_pet_ct(out_dir, rng)

    print("\n[2/5] PET-MR")
    generate_pet_mr(out_dir, rng)

    print("\n[3/5] SPECT-CT")
    generate_spect_ct(out_dir, rng)

    print("\n[4/5] Spectral CT (dual-energy)")
    generate_spectral_ct(out_dir, rng)

    print("\n[5/5] Industrial CT")
    generate_industrial_ct(out_dir, rng)

    print("\n" + "=" * 60)
    print("All datasets generated. Verifying shapes...")
    verify_shapes(out_dir)


def verify_shapes(out_dir: Path) -> None:
    expected = {
        "pet_ct":        {"x_true": (2,256,256), "y": (188,256),  "H_ideal": (188,), "reconstruction_baseline": (256,256)},
        "pet_mr":        {"x_true": (256,256),   "y": (128,256),  "H_ideal": (128,), "reconstruction_baseline": (256,256)},
        "spect_ct":      {"x_true": (256,256),   "y": (256,256),  "H_ideal": (256,), "reconstruction_baseline": (256,256)},
        "spectral_ct":   {"x_true": (2,256,256), "y": (2,60,256), "H_ideal": (60,),  "reconstruction_baseline": (256,256)},
        "industrial_ct": {"x_true": (256,256),   "y": (360,512),  "H_ideal": (360,), "reconstruction_baseline": (256,256)},
    }
    all_ok = True
    for modality, shapes in expected.items():
        for tier, n_samples in TIERS.items():
            h5_path = out_dir / modality / tier / f"{modality}_challenge_{tier}.h5"
            if not h5_path.exists():
                print(f"  MISSING: {h5_path}")
                all_ok = False
                continue
            with h5py.File(h5_path, "r") as hf:
                grp = hf["sample_00"]
                errors = []
                for ds, exp_shape in shapes.items():
                    actual = tuple(grp[ds].shape)
                    if actual != exp_shape:
                        errors.append(f"{ds}: expected {exp_shape}, got {actual}")
                if errors:
                    print(f"  SHAPE MISMATCH [{modality}][{tier}]: {errors}")
                    all_ok = False
                else:
                    n_grps = len([k for k in hf.keys() if k.startswith("sample_")])
                    print(f"  OK  [{modality:15s}][{tier:6s}]  {n_grps} samples  "
                          f"x_true={shapes['x_true']}  y={shapes['y']}")
    if all_ok:
        print("\nAll shape checks passed.")
    else:
        print("\nSome shape checks FAILED — review output above.")


if __name__ == "__main__":
    main()
