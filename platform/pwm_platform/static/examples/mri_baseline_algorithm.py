#!/usr/bin/env python3
"""MRI Baseline Algorithm — Zero-Filled IFFT with Iterative Soft-Thresholding.

Loads a PWM MRI challenge HDF5 file (dev tier), reconstructs each sample via
zero-filled inverse FFT followed by iterative soft-thresholding (compressed
sensing style), estimates mismatch parameters, and saves the submission HDF5.

Usage:
    python mri_baseline_algorithm.py <challenge.h5> <submission.h5>

Dependencies:
    numpy, h5py, scipy (standard scientific Python)

This is a baseline — better results are possible with learned unrolled
networks, dictionary learning, or physics-informed priors.
"""

import sys
import json
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter


def zero_filled_ifft(y_kspace, mask):
    """Zero-filled inverse FFT reconstruction.

    Parameters
    ----------
    y_kspace : ndarray, shape (H, W)
        Log-magnitude k-space measurements (undersampled).
    mask : ndarray, shape (H, W)
        Binary sampling mask (1 = sampled, 0 = not sampled).

    Returns
    -------
    x0 : ndarray, shape (H, W)
        Initial reconstruction (magnitude image).
    """
    # Convert from log-magnitude back to complex k-space
    # The challenge stores y = log(1 + |k-space|), so invert:
    kspace_mag = np.expm1(y_kspace)  # expm1(x) = exp(x) - 1

    # Use zero phase (we only have magnitude)
    kspace_complex = kspace_mag * mask

    # Inverse FFT
    x0 = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_complex)))
    return x0


def soft_threshold(x, lam):
    """Soft-thresholding operator for sparse regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def iterative_soft_threshold(y_kspace, mask, n_iter=30, lam=0.01):
    """Iterative soft-thresholding with data consistency.

    Alternates between:
    1. Enforcing data consistency in k-space (replace sampled locations)
    2. Soft-thresholding in image domain (promote sparsity)

    Parameters
    ----------
    y_kspace : ndarray, shape (H, W)
        Log-magnitude k-space measurements.
    mask : ndarray, shape (H, W)
        Binary sampling mask.
    n_iter : int
        Number of iterations.
    lam : float
        Soft-thresholding parameter.

    Returns
    -------
    x : ndarray, shape (H, W)
        Reconstructed image.
    """
    # Convert to k-space magnitude
    kspace_mag = np.expm1(y_kspace) * mask

    # Initial estimate
    x = zero_filled_ifft(y_kspace, mask)

    for _ in range(n_iter):
        # Forward: go to k-space
        kx = np.fft.fftshift(np.fft.fft2(x))

        # Data consistency: replace sampled locations with measured data
        kx_dc = kx * (1 - mask) + kspace_mag * mask

        # Back to image domain
        x_new = np.abs(np.fft.ifft2(np.fft.ifftshift(kx_dc)))

        # Soft-threshold (image-domain sparsity)
        x_new = soft_threshold(x_new, lam)

        # Light smoothing for stability
        x = gaussian_filter(x_new, sigma=0.3)

    return np.clip(x, 0, None)


def estimate_b0_inhomog(x_hat):
    """Estimate B0 inhomogeneity from spatial intensity variation.

    A uniform B0 field produces uniform intensity in magnitude images.
    Spatial intensity variation suggests B0 inhomogeneity.
    """
    H, W = x_hat.shape
    # Compute intensity profile along rows and columns
    row_mean = np.mean(x_hat, axis=1)
    col_mean = np.mean(x_hat, axis=0)
    # Variation relative to mean
    mean_val = max(np.mean(x_hat), 1e-8)
    row_var = np.std(row_mean) / mean_val
    col_var = np.std(col_mean) / mean_val
    return float(np.clip((row_var + col_var) / 2.0, 0, 1))


def estimate_coil_sensitivity(x_hat):
    """Estimate coil sensitivity non-uniformity.

    Low-frequency intensity modulation suggests coil sensitivity variation.
    """
    smoothed = gaussian_filter(x_hat, sigma=max(x_hat.shape) / 8)
    mean_val = max(np.mean(smoothed), 1e-8)
    non_uniformity = np.std(smoothed) / mean_val
    return float(np.clip(non_uniformity, 0, 1))


def process_sample(grp):
    """Reconstruct one sample from its HDF5 group.

    Returns (x_hat, corrected_spec).
    """
    y = grp["y"][:]              # k-space magnitude, shape (H, W)
    H_ideal = grp["H_ideal"][:]  # sampling mask, shape (H, W)
    spec_ranges = json.loads(grp.attrs["spec_ranges"])

    # Reconstruct via iterative soft-thresholding
    x_hat = iterative_soft_threshold(y, H_ideal, n_iter=30, lam=0.005)

    # Estimate mismatch parameters from reconstruction
    b0_est = estimate_b0_inhomog(x_hat)
    coil_est = estimate_coil_sensitivity(x_hat)

    # Build corrected_spec from spec_ranges
    corrected_spec = {}
    for param in spec_ranges:
        name = param["name"]
        mid = (param["min"] + param["max"]) / 2.0
        if "b0" in name.lower() or "inhomog" in name.lower():
            # Scale to parameter range
            pmin, pmax = param["min"], param["max"]
            corrected_spec[name] = pmin + b0_est * (pmax - pmin)
        elif "coil" in name.lower() or "sensitivity" in name.lower():
            pmin, pmax = param["min"], param["max"]
            corrected_spec[name] = pmin + coil_est * (pmax - pmin)
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
        fout.attrs["variant"] = fin.attrs.get("variant", "mri")
        fout.attrs["tier"] = fin.attrs.get("tier", "dev")
        fout.attrs["submission_type"] = "reconstruction"

    print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    main()
