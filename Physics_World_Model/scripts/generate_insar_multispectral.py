#!/usr/bin/env python3
"""Generate InSAR and Multispectral Satellite datasets (all 3 tiers)."""

import h5py
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "datasets" / "benchmark"


def save_png(arr, path):
    try:
        from PIL import Image
        a = arr.copy()
        if np.iscomplexobj(a):
            a = np.abs(a)
        a = a.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        Image.fromarray((a * 255).astype(np.uint8)).save(path)
    except Exception:
        pass


# ─────────────────────────── InSAR ───────────────────────────

def generate_insar_sample(seed, img_size=256):
    rng = np.random.RandomState(seed)
    # Phase: smooth terrain + atmospheric noise
    x_coords = np.linspace(0, 4 * np.pi, img_size)
    y_coords = np.linspace(0, 4 * np.pi, img_size)
    X, Y = np.meshgrid(x_coords, y_coords)

    # True deformation phase (displacement field)
    deformation = (rng.uniform(0.5, 2.0) * np.sin(X * rng.uniform(0.3, 1.0)) +
                   rng.uniform(0.3, 1.5) * np.cos(Y * rng.uniform(0.3, 1.0)) +
                   rng.randn(img_size, img_size) * 0.1)
    deformation = deformation.astype(np.float32)

    # Amplitude (coherence-weighted)
    coherence = rng.uniform(0.6, 0.95, (img_size, img_size)).astype(np.float32)
    coherence = np.clip(coherence + 0.2 * np.sin(X * 0.5), 0.3, 1.0).astype(np.float32)

    # Wrapped interferogram (measurement) with noise
    noise_std = rng.uniform(0.05, 0.3)
    noise = rng.randn(img_size, img_size).astype(np.float32) * noise_std
    interferogram_wrapped = np.angle(np.exp(1j * (deformation + noise))).astype(np.float32)

    # Baseline unwrapped phase (groundtruth)
    x_true = deformation  # unwrapped phase

    # Forward operator: just wrapping
    H_ideal = np.array([noise_std, float(coherence.mean())], dtype=np.float32)

    return {
        "x_true": x_true,
        "y": interferogram_wrapped,
        "coherence": coherence,
        "H_ideal": H_ideal,
    }


def generate_insar_true_spec(seed):
    rng = np.random.RandomState(seed + 500)
    return {
        "baseline_length_m": float(rng.uniform(50, 500)),
        "perpendicular_baseline_m": float(rng.uniform(20, 300)),
        "temporal_baseline_days": int(rng.randint(6, 180)),
        "noise_std": float(rng.uniform(0.05, 0.3)),
        "coherence_mean": float(rng.uniform(0.5, 0.95)),
        "atmospheric_delay_m": float(rng.uniform(0.001, 0.05)),
    }


def generate_insar(n_public=12, n_dev=20, n_hidden=20):
    out_dir = BENCH / "insar"
    print("Generating InSAR dataset...")

    for tier, n_samples, seed_offset in [
        ("public", n_public, 1000),
        ("dev", n_dev, 7000),
        ("hidden", n_hidden, 9000),
    ]:
        tier_dir = out_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir = tier_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        h5_path = tier_dir / f"insar_challenge_{tier}.h5"
        true_specs = {}

        with h5py.File(h5_path, "w") as hf:
            for i in range(n_samples):
                seed = seed_offset + i * 17
                sample = generate_insar_sample(seed)
                ts = generate_insar_true_spec(seed)
                key = f"sample_{i:02d}"
                true_specs[key] = ts

                grp = hf.create_group(key)
                for k, v in sample.items():
                    grp.create_dataset(k, data=v, compression="gzip")

                save_png(sample["y"], imgs_dir / f"{key}_measurement.png")
                save_png(sample["x_true"], imgs_dir / f"{key}_groundtruth.png")

        # spec.json
        spec = {
            "modality": "insar",
            "tier": tier,
            "n_samples": n_samples,
            "image_size": [256, 256],
            "measurement_key": "y",
            "groundtruth_key": "x_true",
            "forward_model": "phase_wrapping",
            "description": "Wrapped interferogram -> unwrapped deformation phase",
            "reconstruction_task": "phase_unwrapping",
        }
        with open(tier_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        with open(tier_dir / "true_spec.json", "w") as f:
            json.dump(true_specs, f, indent=2)

        print(f"  insar/{tier}: {n_samples} samples -> {h5_path.name}")

    print("InSAR done.")


# ─────────────────── Multispectral Satellite ───────────────────

def generate_multispectral_sample(seed, img_size=128, n_bands=8):
    rng = np.random.RandomState(seed)

    # Simulate a multispectral scene (n_bands x H x W)
    # Different land cover types with spectral signatures
    n_classes = rng.randint(3, 7)
    segmap = rng.randint(0, n_classes, (img_size, img_size))

    # Spectral signatures per class
    signatures = rng.rand(n_classes, n_bands).astype(np.float32)
    # Ensure bands follow rough spectral profile
    for c in range(n_classes):
        signatures[c] = np.sort(rng.rand(n_bands)) if rng.rand() > 0.5 else rng.rand(n_bands)

    # Ground truth spectral image
    x_true = signatures[segmap].transpose(2, 0, 1).astype(np.float32)  # (n_bands, H, W)

    # Measurement: high-res pan + low-res multispectral (pansharpening setup)
    # Pan band: average of visible bands (0-3)
    pan_hr = x_true[:4].mean(axis=0) + rng.randn(img_size, img_size).astype(np.float32) * 0.01
    pan_hr = np.clip(pan_hr, 0, 1).astype(np.float32)

    # Low-res multispectral (4x downsampled)
    from scipy.ndimage import zoom
    ms_lr = np.stack([
        zoom(x_true[b], 0.25, order=1).astype(np.float32) for b in range(n_bands)
    ])

    # Add noise
    noise_std = rng.uniform(0.01, 0.05)
    ms_lr_noisy = (ms_lr + rng.randn(*ms_lr.shape).astype(np.float32) * noise_std).astype(np.float32)
    ms_lr_noisy = np.clip(ms_lr_noisy, 0, None)

    # H_ideal: sensor parameters
    H_ideal = np.array([n_bands, img_size, 4.0, noise_std], dtype=np.float32)  # [n_bands, size, scale_factor, noise]

    # Measurement = concatenate pan_hr and ms_lr (stored separately in h5)
    return {
        "x_true": x_true,
        "pan_hr": pan_hr,
        "ms_lr": ms_lr_noisy,
        "y": ms_lr_noisy,  # primary measurement key
        "H_ideal": H_ideal,
    }


def generate_multispectral_true_spec(seed):
    rng = np.random.RandomState(seed + 600)
    return {
        "scale_factor": 4,
        "n_bands": 8,
        "sensor_noise_std": float(rng.uniform(0.01, 0.05)),
        "atmospheric_opacity": float(rng.uniform(0.05, 0.3)),
        "sun_elevation_deg": float(rng.uniform(30, 80)),
        "cross_calibration_error": float(rng.uniform(0.001, 0.02)),
    }


def generate_multispectral(n_public=12, n_dev=20, n_hidden=20):
    out_dir = BENCH / "multispectral_sat"
    print("Generating Multispectral Satellite dataset...")

    for tier, n_samples, seed_offset in [
        ("public", n_public, 2000),
        ("dev", n_dev, 8000),
        ("hidden", n_hidden, 9500),
    ]:
        tier_dir = out_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir = tier_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        h5_path = tier_dir / f"multispectral_sat_challenge_{tier}.h5"
        true_specs = {}

        with h5py.File(h5_path, "w") as hf:
            for i in range(n_samples):
                seed = seed_offset + i * 13
                sample = generate_multispectral_sample(seed)
                ts = generate_multispectral_true_spec(seed)
                key = f"sample_{i:02d}"
                true_specs[key] = ts

                grp = hf.create_group(key)
                for k, v in sample.items():
                    grp.create_dataset(k, data=v, compression="gzip")

                # Save RGB composite (bands 0,1,2 as RGB)
                rgb = sample["x_true"][:3].transpose(1, 2, 0)
                save_png(rgb.mean(axis=2), imgs_dir / f"{key}_groundtruth.png")
                save_png(sample["pan_hr"], imgs_dir / f"{key}_pan.png")

        spec = {
            "modality": "multispectral_sat",
            "tier": tier,
            "n_samples": n_samples,
            "image_size": [8, 128, 128],
            "measurement_key": "y",
            "groundtruth_key": "x_true",
            "forward_model": "pansharpening",
            "description": "Low-res multispectral + high-res pan -> full multispectral",
            "reconstruction_task": "pansharpening_fusion",
        }
        with open(tier_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        with open(tier_dir / "true_spec.json", "w") as f:
            json.dump(true_specs, f, indent=2)

        print(f"  multispectral_sat/{tier}: {n_samples} samples -> {h5_path.name}")

    print("Multispectral satellite done.")


if __name__ == "__main__":
    generate_insar()
    generate_multispectral()
    print("\nAll done.")
