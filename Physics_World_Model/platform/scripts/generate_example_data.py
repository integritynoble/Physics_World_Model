#!/usr/bin/env python3
"""Generate small example HDF5 files for CT and MRI modalities.

Creates synthetic challenge datasets with tiny dimensions (32x32) to serve
as downloadable examples that demonstrate the exact HDF5 format used in
the Blind Reconstruction Challenge.

For each modality (CT, MRI), generates:
  - {mod}_example_public.h5   — includes x_true + true_spec (development)
  - {mod}_example_dev.h5      — no x_true, no true_spec (blind evaluation)
  - {mod}_example_hidden.h5   — includes x_true + true_spec (server format ref)
  - {mod}_example_submission.h5 — reconstruction output with x_hat + corrected_spec

Output directory: pwm_platform/static/examples/

Usage:
    cd platform
    python scripts/generate_example_data.py
"""

import json
import os
import sys

import numpy as np
import h5py

# ── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE = 32
N_SAMPLES = 3
SEED = 2024

CT_SPEC_RANGES = [
    {"name": "center_offset", "min": -2.0, "max": 2.0, "unit": "px"},
    {"name": "angle_error", "min": -3.0, "max": 3.0, "unit": "deg"},
    {"name": "beam_hardening", "min": 0.0, "max": 0.5, "unit": "a.u."},
    {"name": "detector_tilt", "min": -1.0, "max": 1.0, "unit": "deg"},
]

CT_TRUE_SPEC = {
    "center_offset": 0.7,
    "angle_error": -1.2,
    "beam_hardening": 0.15,
    "detector_tilt": 0.3,
}

MRI_SPEC_RANGES = [
    {"name": "B0_inhomog", "min": 0.0, "max": 0.5, "unit": "a.u."},
    {"name": "gradient_nonlin", "min": 0.0, "max": 0.3, "unit": "a.u."},
    {"name": "coil_sensitivity", "min": 0.0, "max": 0.8, "unit": "a.u."},
    {"name": "k_trajectory", "min": -1.0, "max": 1.0, "unit": "px"},
]

MRI_TRUE_SPEC = {
    "B0_inhomog": 0.2,
    "gradient_nonlin": 0.1,
    "coil_sensitivity": 0.35,
    "k_trajectory": 0.4,
}


# ── Phantom generators ──────────────────────────────────────────────────────


def make_shepp_logan(size, seed=0):
    """Simple Shepp-Logan phantom with slight random variation."""
    rng = np.random.RandomState(seed)
    yy = np.linspace(-1, 1, size)
    xx = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((size, size), dtype=np.float64)
    # Outer ellipse
    arr[(X / 0.85) ** 2 + (Y / 0.95) ** 2 < 1] = 0.15
    # Inner structures
    arr[((X - 0.2) / 0.25) ** 2 + ((Y + 0.1) / 0.35) ** 2 < 1] = 0.6
    arr[((X + 0.25) / 0.20) ** 2 + ((Y + 0.05) / 0.30) ** 2 < 1] = 0.45
    arr[((X + 0.05) / 0.15) ** 2 + ((Y - 0.35) / 0.20) ** 2 < 1] = 0.7
    # Add slight noise
    arr += 0.02 * rng.randn(size, size)
    return np.clip(arr, 0, 1)


def make_brain_phantom(size, seed=0):
    """Brain-like phantom with concentric regions."""
    rng = np.random.RandomState(seed)
    yy = np.linspace(-1, 1, size)
    xx = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((size, size), dtype=np.float64)
    # Skull
    arr[(X ** 2 + Y ** 2) < 0.8] = 0.2
    # Gray matter
    arr[((X ** 2 + Y ** 2) < 0.6) & ((X ** 2 + Y ** 2) > 0.35)] = 0.8
    # White matter
    arr[(X ** 2 + Y ** 2) < 0.35] = 0.4
    # Ventricles
    arr[((X / 0.08) ** 2 + ((Y + 0.05) / 0.15) ** 2) < 1] = 0.05
    arr[((X / 0.08) ** 2 + ((Y - 0.05) / 0.15) ** 2) < 1] = 0.05
    arr += 0.02 * rng.randn(size, size)
    return np.clip(arr, 0, 1)


# ── Forward models ───────────────────────────────────────────────────────────


def forward_radon(x, n_angles=32):
    """Simplified Radon transform (sinogram) for small images."""
    from scipy.ndimage import rotate as ndrotate

    size = x.shape[0]
    diag = int(np.ceil(np.sqrt(2) * size))
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    pad = (diag - size) // 2
    padded = np.pad(x, ((pad, diag - size - pad), (pad, diag - size - pad)))

    sinogram = np.zeros((n_angles, diag), dtype=np.float64)
    for i, angle in enumerate(angles):
        rotated = ndrotate(padded, -angle, reshape=False, order=1, mode="constant")
        sinogram[i] = rotated.sum(axis=0)

    return sinogram, angles


def forward_kspace(x, acceleration=4):
    """K-space undersampling for MRI."""
    H, W = x.shape
    kspace = np.fft.fftshift(np.fft.fft2(x))

    # Cartesian undersampling mask with ACS lines
    mask = np.zeros((H, W), dtype=np.float64)
    acs_lines = max(4, H // 8)
    center = H // 2
    mask[center - acs_lines // 2 : center + acs_lines // 2, :] = 1.0
    # Uniform undersampling
    step = max(1, acceleration)
    mask[::step, :] = 1.0

    undersampled = kspace * mask
    y = np.log1p(np.abs(undersampled))
    return y, mask


def apply_mismatch_ct(y, true_spec, rng):
    """Apply CT-specific mismatch to sinogram."""
    n_angles, n_det = y.shape

    # Center offset: shift sinogram columns
    offset = true_spec.get("center_offset", 0)
    offset_int = int(round(offset))
    if offset_int != 0:
        y = np.roll(y, offset_int, axis=1)

    # Angle error: slight global rotation of the sinogram
    angle_err = true_spec.get("angle_error", 0)
    if abs(angle_err) > 0.01:
        # Shift sinogram rows (equivalent to angle perturbation)
        shift = int(round(angle_err * n_angles / 180.0))
        y = np.roll(y, shift, axis=0)

    # Beam hardening: nonlinear intensity transformation
    bh = true_spec.get("beam_hardening", 0)
    if bh > 0:
        y = y + bh * y ** 2

    # Add noise
    y += 0.01 * rng.randn(*y.shape)
    return y


def apply_mismatch_mri(y, mask, true_spec, rng):
    """Apply MRI-specific mismatch to k-space data."""
    H, W = y.shape

    # B0 inhomogeneity: spatial phase variation
    b0 = true_spec.get("B0_inhomog", 0)
    if b0 > 0.01:
        yy = np.linspace(-1, 1, H)
        xx = np.linspace(-1, 1, W)
        X, Y = np.meshgrid(xx, yy)
        phase_map = b0 * (X ** 2 + Y ** 2)
        y = y * (1 + 0.3 * phase_map)

    # Coil sensitivity: spatial intensity modulation
    cs = true_spec.get("coil_sensitivity", 0)
    if cs > 0.01:
        yy = np.linspace(-1, 1, H)
        xx = np.linspace(-1, 1, W)
        X, Y = np.meshgrid(xx, yy)
        sensitivity = 1.0 - cs * np.sqrt(X ** 2 + Y ** 2)
        y = y * np.clip(sensitivity, 0.3, 1.0)

    # Add noise
    y += 0.01 * rng.randn(*y.shape)
    return y


# ── HDF5 file generation ────────────────────────────────────────────────────


def generate_ct_file(output_path, tier, n_samples=N_SAMPLES):
    """Generate a CT example HDF5 file."""
    rng = np.random.RandomState(SEED + hash(tier) % 10000)

    with h5py.File(output_path, "w") as f:
        f.attrs["variant"] = "ct"
        f.attrs["tier"] = tier
        f.attrs["version"] = "1.0"
        f.attrs["runner_type"] = "radon"

        for i in range(n_samples):
            x_true = make_shepp_logan(IMG_SIZE, seed=SEED + i)
            y, angles = forward_radon(x_true, n_angles=IMG_SIZE)
            y = apply_mismatch_ct(y, CT_TRUE_SPEC, rng)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=angles, compression="gzip")
            grp.attrs["spec_ranges"] = json.dumps(CT_SPEC_RANGES)
            grp.attrs["metadata"] = json.dumps({
                "scene": f"shepp_logan_{i}",
                "shape": [IMG_SIZE, IMG_SIZE],
                "noise_model": "gaussian",
            })

            if tier in ("public", "hidden"):
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.attrs["true_spec"] = json.dumps(CT_TRUE_SPEC)


def generate_mri_file(output_path, tier, n_samples=N_SAMPLES):
    """Generate an MRI example HDF5 file."""
    rng = np.random.RandomState(SEED + hash(tier) % 10000 + 42)

    with h5py.File(output_path, "w") as f:
        f.attrs["variant"] = "mri"
        f.attrs["tier"] = tier
        f.attrs["version"] = "1.0"
        f.attrs["runner_type"] = "kspace"

        for i in range(n_samples):
            x_true = make_brain_phantom(IMG_SIZE, seed=SEED + i + 100)
            y, mask = forward_kspace(x_true, acceleration=4)
            y = apply_mismatch_mri(y, mask, MRI_TRUE_SPEC, rng)

            grp = f.create_group(f"sample_{i:02d}")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=mask, compression="gzip")
            grp.attrs["spec_ranges"] = json.dumps(MRI_SPEC_RANGES)
            grp.attrs["metadata"] = json.dumps({
                "scene": f"brain_phantom_{i}",
                "shape": [IMG_SIZE, IMG_SIZE],
                "noise_model": "gaussian",
            })

            if tier in ("public", "hidden"):
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.attrs["true_spec"] = json.dumps(MRI_TRUE_SPEC)


def generate_ct_submission(output_path, input_path):
    """Generate an example CT submission HDF5 from a dev file."""
    rng = np.random.RandomState(SEED + 999)

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        fout.attrs["variant"] = "ct"
        fout.attrs["tier"] = "dev"
        fout.attrs["submission_type"] = "reconstruction"

        for key in sorted(fin.keys()):
            if not key.startswith("sample_"):
                continue
            y = fin[key]["y"][:]
            n_det = y.shape[1]
            # Dummy reconstruction (just a placeholder image)
            x_hat = rng.rand(IMG_SIZE, IMG_SIZE).astype(np.float64) * 0.5
            corrected_spec = {p["name"]: (p["min"] + p["max"]) / 2
                              for p in CT_SPEC_RANGES}

            grp = fout.create_group(key)
            grp.create_dataset("x_hat", data=x_hat, compression="gzip")
            grp.attrs["corrected_spec"] = json.dumps(corrected_spec)


def generate_mri_submission(output_path, input_path):
    """Generate an example MRI submission HDF5 from a dev file."""
    rng = np.random.RandomState(SEED + 998)

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        fout.attrs["variant"] = "mri"
        fout.attrs["tier"] = "dev"
        fout.attrs["submission_type"] = "reconstruction"

        for key in sorted(fin.keys()):
            if not key.startswith("sample_"):
                continue
            x_hat = rng.rand(IMG_SIZE, IMG_SIZE).astype(np.float64) * 0.5
            corrected_spec = {p["name"]: (p["min"] + p["max"]) / 2
                              for p in MRI_SPEC_RANGES}

            grp = fout.create_group(key)
            grp.create_dataset("x_hat", data=x_hat, compression="gzip")
            grp.attrs["corrected_spec"] = json.dumps(corrected_spec)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "pwm_platform", "static", "examples")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating example HDF5 files in {output_dir}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}, Samples per file: {N_SAMPLES}")
    print()

    # CT files
    for tier in ("public", "dev", "hidden"):
        path = os.path.join(output_dir, f"ct_example_{tier}.h5")
        generate_ct_file(path, tier)
        size_kb = os.path.getsize(path) / 1024
        print(f"  CT {tier:7s} -> {os.path.basename(path)} ({size_kb:.1f} KB)")

    # MRI files
    for tier in ("public", "dev", "hidden"):
        path = os.path.join(output_dir, f"mri_example_{tier}.h5")
        generate_mri_file(path, tier)
        size_kb = os.path.getsize(path) / 1024
        print(f"  MRI {tier:7s} -> {os.path.basename(path)} ({size_kb:.1f} KB)")

    # Submission examples
    ct_dev = os.path.join(output_dir, "ct_example_dev.h5")
    ct_sub = os.path.join(output_dir, "ct_example_submission.h5")
    generate_ct_submission(ct_sub, ct_dev)
    size_kb = os.path.getsize(ct_sub) / 1024
    print(f"  CT submission -> {os.path.basename(ct_sub)} ({size_kb:.1f} KB)")

    mri_dev = os.path.join(output_dir, "mri_example_dev.h5")
    mri_sub = os.path.join(output_dir, "mri_example_submission.h5")
    generate_mri_submission(mri_sub, mri_dev)
    size_kb = os.path.getsize(mri_sub) / 1024
    print(f"  MRI submission -> {os.path.basename(mri_sub)} ({size_kb:.1f} KB)")

    print()
    print("Done! Generated 8 example HDF5 files.")


if __name__ == "__main__":
    main()
