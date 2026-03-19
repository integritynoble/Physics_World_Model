#!/usr/bin/env python3
"""CT Baseline Algorithm — Filtered Back-Projection with Mismatch Estimation.

Loads a PWM CT challenge HDF5 file (dev tier), reconstructs each sample via
Filtered Back-Projection (FBP), estimates mismatch parameters by minimizing
the measurement residual, and saves the submission HDF5.

Usage:
    python ct_baseline_algorithm.py <challenge.h5> <submission.h5>

Dependencies:
    numpy, h5py, scipy (standard scientific Python)

This is a baseline — better results are possible with iterative methods,
learned priors, or more sophisticated mismatch estimation.
"""

import sys
import json
import numpy as np
import h5py
from scipy.ndimage import rotate as ndrotate


def fbp_reconstruct(sinogram, angles, img_size=None):
    """Filtered Back-Projection reconstruction.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detectors)
        Sinogram measurement data.
    angles : ndarray, shape (n_angles,)
        Projection angles in degrees.
    img_size : int, optional
        Output image size. Defaults to n_detectors.

    Returns
    -------
    recon : ndarray, shape (img_size, img_size)
        Reconstructed image.
    """
    n_angles, n_det = sinogram.shape
    if img_size is None:
        img_size = n_det

    # Step 1: Apply ramp filter in frequency domain to each projection
    freqs = np.fft.fftfreq(n_det)
    ramp = np.abs(freqs)
    filtered = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj_fft = np.fft.fft(sinogram[i])
        filtered[i] = np.real(np.fft.ifft(proj_fft * ramp))

    # Step 2: Back-project filtered projections
    recon = np.zeros((img_size, img_size), dtype=np.float64)
    pad_size = int(np.ceil(np.sqrt(2) * img_size))
    pad_offset = (pad_size - img_size) // 2

    for i, angle in enumerate(angles):
        # Create a 2D image by replicating the filtered projection row
        proj_img = np.zeros((pad_size, pad_size), dtype=np.float64)
        # Center the projection in the padded image
        if n_det <= pad_size:
            start = (pad_size - n_det) // 2
            for row in range(pad_size):
                proj_img[row, start:start + n_det] = filtered[i]
        else:
            center = n_det // 2
            half = pad_size // 2
            for row in range(pad_size):
                proj_img[row] = filtered[i, center - half:center - half + pad_size]

        # Rotate and accumulate
        rotated = ndrotate(proj_img, angle, reshape=False, order=1, mode="constant")
        recon += rotated[pad_offset:pad_offset + img_size,
                         pad_offset:pad_offset + img_size]

    recon *= np.pi / (2.0 * n_angles)
    return np.clip(recon, 0, None)


def estimate_center_offset(sinogram):
    """Estimate center offset from sinogram asymmetry.

    Compares the sinogram at 0 and 180 degrees (or the closest pair).
    A non-zero center offset causes left-right asymmetry.
    """
    n_angles, n_det = sinogram.shape
    # Compare first and last (approximately 0 and ~180 degrees)
    proj_0 = sinogram[0]
    proj_180 = sinogram[n_angles // 2] if n_angles > 1 else proj_0
    flipped = proj_180[::-1]

    best_offset = 0.0
    best_score = np.inf
    for offset_px in range(-3, 4):
        shifted = np.roll(proj_0, offset_px)
        score = np.sum((shifted - flipped) ** 2)
        if score < best_score:
            best_score = score
            best_offset = float(offset_px) * 0.5
    return best_offset


def estimate_angle_error(sinogram, angles, x_hat):
    """Estimate global angle error by testing small rotations."""
    n_angles, n_det = sinogram.shape
    best_err = 0.0
    best_resid = np.inf

    for err in np.linspace(-2.0, 2.0, 9):
        corrected_angles = angles + err
        # Quick forward: compute a few projections and compare
        resid = 0.0
        for idx in [0, n_angles // 4, n_angles // 2]:
            if idx >= n_angles:
                continue
            angle = corrected_angles[idx]
            pad_size = int(np.ceil(np.sqrt(2) * x_hat.shape[0]))
            pad_h = (pad_size - x_hat.shape[0]) // 2
            pad_w = (pad_size - x_hat.shape[1]) // 2
            padded = np.pad(x_hat, ((pad_h, pad_size - x_hat.shape[0] - pad_h),
                                     (pad_w, pad_size - x_hat.shape[1] - pad_w)))
            rotated = ndrotate(padded, -angle, reshape=False, order=1, mode="constant")
            proj = rotated.sum(axis=0)
            # Compare with measured sinogram row
            if n_det <= pad_size:
                start = (pad_size - n_det) // 2
                proj_crop = proj[start:start + n_det]
            else:
                proj_crop = proj
            measured = sinogram[idx, :len(proj_crop)]
            resid += np.sum((proj_crop - measured) ** 2)

        if resid < best_resid:
            best_resid = resid
            best_err = err

    return best_err


def process_sample(grp):
    """Reconstruct one sample from its HDF5 group.

    Returns (x_hat, corrected_spec).
    """
    y = grp["y"][:]            # sinogram, shape (n_angles, n_detectors)
    H_ideal = grp["H_ideal"][:] # projection angles, shape (n_angles,)
    spec_ranges = json.loads(grp.attrs["spec_ranges"])

    # Determine image size from sinogram
    n_angles, n_det = y.shape
    img_size = n_det  # typically n_detectors == image width

    # Initial FBP reconstruction
    x_hat = fbp_reconstruct(y, H_ideal, img_size=img_size)

    # Estimate mismatch parameters
    center_offset = estimate_center_offset(y)
    angle_error = estimate_angle_error(y, H_ideal, x_hat)

    # Refine reconstruction with corrected angles
    corrected_angles = H_ideal + angle_error
    x_hat = fbp_reconstruct(y, corrected_angles, img_size=img_size)

    # Build corrected_spec from spec_ranges
    corrected_spec = {}
    for param in spec_ranges:
        name = param["name"]
        mid = (param["min"] + param["max"]) / 2.0
        if "center" in name or "offset" in name:
            corrected_spec[name] = center_offset
        elif "angle" in name or "tilt" in name:
            corrected_spec[name] = angle_error
        else:
            corrected_spec[name] = mid  # default: midpoint of range
    return x_hat, corrected_spec


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        samples = sorted([k for k in fin.keys() if k.startswith("sample_")])
        print(f"Processing {len(samples)} samples from {input_path}")

        for sample_key in samples:
            print(f"  {sample_key}...", end=" ", flush=True)
            x_hat, corrected_spec = process_sample(fin[sample_key])

            grp = fout.create_group(sample_key)
            grp.create_dataset("x_hat", data=x_hat, compression="gzip")
            grp.attrs["corrected_spec"] = json.dumps(corrected_spec)
            print(f"done (shape={x_hat.shape}, "
                  f"spec={list(corrected_spec.keys())})")

        # Copy metadata
        fout.attrs["variant"] = fin.attrs.get("variant", "ct")
        fout.attrs["tier"] = fin.attrs.get("tier", "dev")
        fout.attrs["submission_type"] = "reconstruction"

    print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    main()
