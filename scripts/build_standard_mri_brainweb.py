#!/usr/bin/env python3
"""Build MRI standard dataset from BrainWeb brain phantoms.

Replaces synthetic Shepp-Logan phantoms with realistic BrainWeb brain anatomy.
20 subjects × 1 informative axial slice each → 20 standard scenes at 320×320.

Forward model: single-coil Cartesian MRI with variable-density 4× undersampling
(8% center fraction), matching the fastMRI benchmark protocol.

References:
  - BrainWeb: Collins et al., IEEE TMI 1998 (>5000 citations)
  - fastMRI protocol: Zbontar et al., arXiv 2018
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import json
import brainweb
from skimage.transform import resize
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")

# Parameters matching fastMRI benchmark
IMAGE_SIZE = 320        # fastMRI standard
ACCELERATION = 4        # 4× undersampling
CENTER_FRACTION = 0.08  # 8% center lines always sampled
N_SCENES = 20
SEED = 42


def create_variable_density_mask(shape, acceleration, center_fraction, rng):
    """Create fastMRI-style variable-density Cartesian undersampling mask.

    Keeps center_fraction of k-space lines in the center,
    randomly samples remaining lines to achieve target acceleration.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Always sample center lines (low-frequency)
    n_center = int(round(W * center_fraction))
    center_start = (W - n_center) // 2
    mask[:, center_start:center_start + n_center] = 1.0

    # Randomly sample additional lines to reach target acceleration
    n_total_lines = int(round(W / acceleration))
    n_extra = max(0, n_total_lines - n_center)

    # Available outer lines (not in center)
    outer_indices = list(range(0, center_start)) + list(range(center_start + n_center, W))
    chosen = rng.choice(outer_indices, size=min(n_extra, len(outer_indices)), replace=False)
    for idx in chosen:
        mask[:, idx] = 1.0

    return mask


def build_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # Download all 20 BrainWeb subjects
    print("Downloading BrainWeb subjects...")
    files = brainweb.get_files()
    print(f"Got {len(files)} subjects")

    scenes = []
    psnr_list = []

    for i in range(N_SCENES):
        print(f"\n--- Scene {i:02d} (subject {i}/{len(files)}) ---")

        # Generate T1-weighted image (no noise = clean ground truth)
        mmr = brainweb.get_mmr_fromfile(
            files[i % len(files)],
            t1Noise=0, t2Noise=0, petNoise=0
        )
        t1_vol = mmr['T1']  # shape: (127, 344, 344)

        # Pick the most informative axial slice (maximum tissue content)
        # Vary slice selection across subjects for diversity
        best_slice_idx = -1
        best_score = -1
        search_range = range(40, 90)  # brain region
        for z in search_range:
            sl = t1_vol[z]
            score = np.sum(sl > 10) * np.std(sl)  # tissue coverage × contrast
            if score > best_score:
                best_score = score
                best_slice_idx = z

        # Offset slice selection for diversity (±5 slices around optimal)
        offset = (i % 5) - 2  # -2, -1, 0, 1, 2
        slice_idx = np.clip(best_slice_idx + offset, 40, 89)

        raw_slice = t1_vol[int(slice_idx)]  # (344, 344)
        print(f"  Selected slice {slice_idx}, raw shape: {raw_slice.shape}")

        # Center-crop to 320×320
        h, w = raw_slice.shape
        ch = (h - IMAGE_SIZE) // 2
        cw = (w - IMAGE_SIZE) // 2
        x_true = raw_slice[ch:ch+IMAGE_SIZE, cw:cw+IMAGE_SIZE].astype(np.float32)

        # Normalize to [0, 1]
        x_min, x_max = x_true.min(), x_true.max()
        if x_max > x_min:
            x_true = (x_true - x_min) / (x_max - x_min)

        print(f"  x_true: shape={x_true.shape}, range=[{x_true.min():.3f}, {x_true.max():.3f}]")

        # Simulate fully-sampled k-space via 2D FFT
        kspace_full = np.fft.fftshift(np.fft.fft2(x_true))

        # Create variable-density undersampling mask (different per scene)
        mask = create_variable_density_mask(
            (IMAGE_SIZE, IMAGE_SIZE), ACCELERATION, CENTER_FRACTION, rng
        )
        actual_rate = mask.sum() / mask.size
        print(f"  Mask: sampling rate={actual_rate:.3f} ({1/actual_rate:.1f}× acceleration)")

        # Apply mask to get undersampled k-space
        kspace_under = kspace_full * mask

        # Store as (H, W, 2) real+imag format (matches existing convention)
        y_ideal = np.stack([kspace_under.real, kspace_under.imag], axis=-1).astype(np.float32)

        # Compute zero-filled reconstruction for baseline PSNR
        zf_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under))).astype(np.float32)
        mse = np.mean((x_true - zf_recon) ** 2)
        psnr_zf = 10 * np.log10(1.0 / max(mse, 1e-10))
        psnr_list.append(psnr_zf)
        print(f"  Zero-filled PSNR: {psnr_zf:.2f} dB")

        # Save h5 file
        h5_path = os.path.join(OUT_DIR, f"standard_mri_{i:02d}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('x_true', data=x_true, dtype=np.float32)
            hf.create_dataset('y_ideal', data=y_ideal, dtype=np.float32)
            hf.create_dataset('sampling_mask', data=mask, dtype=np.float32)

            # H_params group (forward model parameters)
            hp = hf.create_group('H_params')
            hp.attrs['modality'] = 'mri'
            hp.attrs['forward_model'] = 'single_coil_cartesian_fft'
            hp.attrs['acceleration'] = ACCELERATION
            hp.attrs['center_fraction'] = CENTER_FRACTION
            hp.attrs['image_size'] = IMAGE_SIZE

            # Metadata
            meta = hf.create_group('metadata')
            meta.attrs['source'] = f'BrainWeb subject {i:02d}'
            meta.attrs['slice_index'] = int(slice_idx)
            meta.attrs['contrast'] = 'T1-weighted'
            meta.attrs['zero_filled_psnr'] = float(psnr_zf)

        scenes.append(h5_path)
        print(f"  Saved: {h5_path}")

    # Update metadata.json
    metadata = {
        "modality": "mri",
        "canonical_dataset": "BrainWeb T1-weighted brain MRI (Collins et al., IEEE TMI 1998)",
        "benchmark_protocol": "fastMRI-style single-coil Cartesian 4× undersampling",
        "n_samples": N_SCENES,
        "x_true_shape": [IMAGE_SIZE, IMAGE_SIZE],
        "y_ideal_shape": [IMAGE_SIZE, IMAGE_SIZE, 2],
        "y_normalization": "none",
        "image_size": IMAGE_SIZE,
        "acceleration": ACCELERATION,
        "center_fraction": CENTER_FRACTION,
        "undersampling": "variable-density random Cartesian (fastMRI-style)",
        "note": (
            "x_true: 320×320 T1-weighted brain MRI from BrainWeb (20 subjects). "
            "y_ideal: undersampled k-space (real+imag). "
            "mask stored as sampling_mask. "
            "Forward model: 2D FFT with Cartesian undersampling. "
            "Reference: Collins et al., 'Design and construction of a realistic digital "
            "brain phantom', IEEE TMI 1998. "
            "BrainWeb is one of the most widely used MRI phantoms in reconstruction "
            "literature (>5000 citations). "
            "For comparison with fastMRI leaderboard numbers, use real fastMRI knee data "
            "(requires registration at https://fastmri.med.nyu.edu/)."
        ),
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "files": [f"standard_mri_{i:02d}.h5" for i in range(N_SCENES)]
    }

    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update spec.json
    spec = {
        "modality": "mri",
        "operator": "fourier_encoding",
        "image_size": IMAGE_SIZE,
        "acceleration": ACCELERATION,
        "center_fraction": CENTER_FRACTION,
        "kspace_format": "(H, W, 2) real+imag",
        "noise": "none",
        "y_normalization": "none",
        "source": "BrainWeb T1 brain phantom (20 subjects)",
        "reference": "Collins et al., IEEE TMI 1998; fastMRI protocol (Zbontar et al., 2018)"
    }

    spec_path = os.path.join(OUT_DIR, "spec.json")
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Built {N_SCENES} MRI standard scenes from BrainWeb brain phantoms")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Acceleration: {ACCELERATION}×")
    print(f"Mean zero-filled PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    build_dataset()
