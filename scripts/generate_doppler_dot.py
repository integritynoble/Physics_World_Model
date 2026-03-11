#!/usr/bin/env python3
"""Generate Doppler Ultrasound and DOT benchmark datasets."""
import h5py
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "datasets" / "benchmark"


def save_png(arr, path):
    try:
        from PIL import Image
        a = arr.astype(np.float32)
        if np.iscomplexobj(a):
            a = np.abs(a)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        Image.fromarray((a * 255).astype(np.uint8)).save(path)
    except Exception:
        pass


# ─────────────────────────── Doppler Ultrasound ───────────────────────────

def generate_doppler_sample(seed, img_size=256):
    rng = np.random.RandomState(seed)

    # Ground truth velocity field (2D, m/s) — simulates blood flow
    x_coords = np.linspace(0, 2 * np.pi, img_size)
    y_coords = np.linspace(0, 2 * np.pi, img_size)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Parabolic flow profile (vessel-like)
    vessel_cx = rng.uniform(0.3, 0.7) * img_size
    vessel_cy = rng.uniform(0.3, 0.7) * img_size
    vessel_r = rng.uniform(0.1, 0.2) * img_size
    grid_x = np.arange(img_size)
    grid_y = np.arange(img_size)
    GX, GY = np.meshgrid(grid_x, grid_y)

    dist = np.sqrt((GX - vessel_cx)**2 + (GY - vessel_cy)**2)
    # Velocity profile: parabolic inside vessel
    v_max = rng.uniform(0.3, 1.5)  # m/s
    v_field = np.where(dist < vessel_r, v_max * (1 - (dist / vessel_r)**2), 0.0).astype(np.float32)

    # Measurement: autocorrelation-based I/Q data (simplified)
    # Doppler shift -> phase shift between pulses
    prf = rng.uniform(5000, 15000)  # pulse repetition frequency Hz
    c_sound = 1540.0  # m/s
    f0 = 5e6  # MHz transducer
    depth_samples = 128
    n_pulses = 512

    # Simplified I/Q signal: phase shift per pulse proportional to velocity
    doppler_phase = 4 * np.pi * f0 * v_field / c_sound / prf  # radians per pulse

    # Build 2D sinogram-like measurement (slow-time fast-time)
    # Just use first row of v_field for 1D measurement
    v_axial = v_field.mean(axis=1)  # (img_size,)
    phase_shifts = 4 * np.pi * f0 * v_axial / c_sound / prf

    # Generate I/Q data with noise
    t = np.arange(n_pulses)[:, np.newaxis]
    noise_std = rng.uniform(0.05, 0.2)

    iq_real = np.cos(phase_shifts[np.newaxis, :64] * t) + rng.randn(n_pulses, 64).astype(np.float32) * noise_std
    iq_imag = np.sin(phase_shifts[np.newaxis, :64] * t) + rng.randn(n_pulses, 64).astype(np.float32) * noise_std

    # Measurement: concatenated I/Q (n_pulses, 2*n_gates) -> reshape to (128, 512)
    y = np.concatenate([iq_real, iq_imag], axis=1).astype(np.float32)  # (512, 128)
    y = y.T  # (128, 512)

    # Ground truth: velocity field (256, 256)
    x_true = v_field.astype(np.float32)

    # H_ideal: system parameters
    H_ideal = np.array([f0 / 1e6, prf / 1000, c_sound, noise_std], dtype=np.float32)

    # Baseline: autocorrelation estimator velocity
    # Simple: take angle of correlation between consecutive pulses
    iq_complex = iq_real.astype(complex) + 1j * iq_imag
    corr = (iq_complex[1:] * np.conj(iq_complex[:-1])).mean(axis=0)
    v_est_1d = np.angle(corr) * c_sound * prf / (4 * np.pi * f0)
    # Expand to 2D
    v_est_2d = np.tile(v_est_1d[:, np.newaxis], (4, img_size)).T[:img_size, :img_size]
    recon_baseline = v_est_2d.astype(np.float32)

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon_baseline,
    }


def generate_doppler_true_spec(seed):
    rng = np.random.RandomState(seed + 400)
    return {
        "center_frequency_mhz": float(rng.uniform(3, 10)),
        "prf_hz": float(rng.uniform(5000, 15000)),
        "sound_speed_ms": 1540.0,
        "max_velocity_ms": float(rng.uniform(0.3, 1.5)),
        "noise_std": float(rng.uniform(0.05, 0.2)),
    }


def generate_doppler_ultrasound(n_public=12, n_dev=20, n_hidden=20):
    out_dir = BENCH / "doppler_ultrasound"
    print("Generating Doppler Ultrasound dataset...")

    for tier, n_samples, seed_offset in [
        ("public", n_public, 1100),
        ("dev", n_dev, 7100),
        ("hidden", n_hidden, 9100),
    ]:
        tier_dir = out_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir = tier_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        h5_path = tier_dir / f"doppler_ultrasound_challenge_{tier}.h5"
        true_specs = {}

        with h5py.File(h5_path, "w") as hf:
            for i in range(n_samples):
                seed = seed_offset + i * 19
                sample = generate_doppler_sample(seed)
                ts = generate_doppler_true_spec(seed)
                key = f"sample_{i:02d}"
                true_specs[key] = ts

                grp = hf.create_group(key)
                for k, v in sample.items():
                    grp.create_dataset(k, data=v, compression="gzip")

                save_png(sample["x_true"], imgs_dir / f"{key}_velocity_gt.png")
                save_png(sample["y"], imgs_dir / f"{key}_iq_measurement.png")

        spec = {
            "modality": "doppler_ultrasound",
            "tier": tier,
            "n_samples": n_samples,
            "image_size": [256, 256],
            "measurement_key": "y",
            "groundtruth_key": "x_true",
            "forward_model": "doppler_iq",
            "description": "I/Q Doppler signals -> velocity field",
            "reconstruction_task": "velocity_estimation",
        }
        with open(tier_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2)
        with open(tier_dir / "true_spec.json", "w") as f:
            json.dump(true_specs, f, indent=2)

        print(f"  doppler_ultrasound/{tier}: {n_samples} samples -> {h5_path.name}")

    print("Doppler Ultrasound done.")


# ─────────────────────────── DOT ───────────────────────────

def generate_dot_sample(seed, vol_size=64):
    rng = np.random.RandomState(seed)

    # Ground truth: 3D absorption map (cm^-1)
    x_true_3d = rng.uniform(0.01, 0.05, (vol_size, vol_size, vol_size)).astype(np.float32)

    # Add inclusions (tumors)
    n_inclusions = rng.randint(1, 4)
    for _ in range(n_inclusions):
        cx = rng.randint(vol_size // 4, 3 * vol_size // 4)
        cy = rng.randint(vol_size // 4, 3 * vol_size // 4)
        cz = rng.randint(vol_size // 4, 3 * vol_size // 4)
        r = rng.randint(3, 10)
        zz, yy, xx = np.ogrid[:vol_size, :vol_size, :vol_size]
        mask = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 < r**2
        x_true_3d[mask] += rng.uniform(0.1, 0.3)

    x_true_3d = x_true_3d.flatten().astype(np.float32)  # (64^3,) = 262144 -> for storage

    # Forward model: Born approximation
    # y = A * x where A is the DOT sensitivity matrix (simplified)
    # Use smaller measurement vector (256 source-detector pairs)
    n_meas = 256
    n_src = 16
    n_det = 16
    n_vox = min(1024, vol_size**3)  # Use reduced spatial sampling

    # Simplified sensitivity matrix (Born approx, exponential decay)
    rng2 = np.random.RandomState(seed + 200)
    x_small = x_true_3d[:n_vox]  # first n_vox voxels
    A = rng2.randn(n_meas, n_vox).astype(np.float32) * 0.01  # random sensitivity
    A = np.abs(A)  # Positive sensitivity (forward model)

    # Measurement
    noise_std = rng.uniform(0.001, 0.01)
    y = (A @ x_small + rng.randn(n_meas).astype(np.float32) * noise_std).astype(np.float32)

    # Baseline reconstruction: Born approximation (A^T y, thresholded)
    x_hat = A.T @ y
    x_hat = x_hat / (x_hat.max() + 1e-8) * x_small.max()
    recon_baseline = x_hat.astype(np.float32)

    # x_true: 3D volume reshaped to 2D for display
    x_true_2d = x_true_3d[:vol_size*vol_size].reshape(vol_size, vol_size).astype(np.float32)

    # H_ideal: wavelength, scatter coeff
    H_ideal = np.array([785.0, 1.0, noise_std, n_meas], dtype=np.float32)  # wavelength, mu_s, noise, n_meas

    return {
        "x_true": x_true_2d,  # 2D slice for benchmarking
        "x_true_3d": x_true_3d[:n_vox],  # reduced 3D data
        "y": y,  # measurement vector
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon_baseline,
    }


def generate_dot_true_spec(seed):
    rng = np.random.RandomState(seed + 500)
    return {
        "wavelength_nm": float(rng.choice([785, 830, 690])),
        "source_detector_pairs": int(rng.randint(128, 512)),
        "background_absorption_cm_inv": float(rng.uniform(0.01, 0.05)),
        "noise_std": float(rng.uniform(0.001, 0.01)),
        "inclusion_count": int(rng.randint(1, 4)),
    }


def generate_dot(n_public=12, n_dev=20, n_hidden=20):
    out_dir = BENCH / "dot"
    print("Generating DOT dataset...")

    for tier, n_samples, seed_offset in [
        ("public", n_public, 1200),
        ("dev", n_dev, 7200),
        ("hidden", n_hidden, 9200),
    ]:
        tier_dir = out_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir = tier_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        h5_path = tier_dir / f"dot_challenge_{tier}.h5"
        true_specs = {}

        with h5py.File(h5_path, "w") as hf:
            for i in range(n_samples):
                seed = seed_offset + i * 23
                sample = generate_dot_sample(seed)
                ts = generate_dot_true_spec(seed)
                key = f"sample_{i:02d}"
                true_specs[key] = ts

                grp = hf.create_group(key)
                for k, v in sample.items():
                    grp.create_dataset(k, data=v, compression="gzip")

                save_png(sample["x_true"], imgs_dir / f"{key}_absorption_gt.png")
                save_png(np.abs(sample["y"]).reshape(16, 16), imgs_dir / f"{key}_measurement.png")

        spec = {
            "modality": "dot",
            "tier": tier,
            "n_samples": n_samples,
            "image_size": [64, 64],
            "measurement_key": "y",
            "groundtruth_key": "x_true",
            "forward_model": "born_approximation",
            "description": "Boundary flux measurements -> 3D absorption map",
            "reconstruction_task": "tomographic_reconstruction",
        }
        with open(tier_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2)
        with open(tier_dir / "true_spec.json", "w") as f:
            json.dump(true_specs, f, indent=2)

        print(f"  dot/{tier}: {n_samples} samples -> {h5_path.name}")

    print("DOT done.")


if __name__ == "__main__":
    generate_doppler_ultrasound()
    generate_dot()
    print("\nAll done.")
