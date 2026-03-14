"""Standardized data loader for the expert study.

Loads benchmark data from GCS cache. Returns measurements + calibration
metadata only — NO physics model hints. Experts must derive the forward
model from the one-sentence design brief.
"""
from __future__ import annotations

import sys
import h5py
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.gcs_dataset_helper import ensure_challenge_dataset


def load_data(modality: str, n_samples: int = 10) -> list[dict]:
    """Load benchmark samples for a modality.

    Returns a list of dicts, each containing:
      - 'measurements': the raw measurement array
      - 'calibration': dict of calibration/metadata arrays (angles, mask, etc.)
      - 'x_true': ground truth (for evaluation only — NOT given to experts)
      - 'shape_info': string describing array shapes (given to experts)
    """
    # Prefer canonical path (proper keys) over legacy flat cache
    canonical = Path(f"/tmp/pwm_challenge_cache/datasets/Benchmark/{modality}/public/{modality}_challenge_public.h5")
    if canonical.exists():
        h5_path = canonical
    else:
        h5_path = ensure_challenge_dataset(modality, "public")

    samples = []
    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
        for key in sample_keys[:n_samples]:
            grp = f[key]
            sample = _load_modality_sample(modality, grp)
            samples.append(sample)

    return samples


def _load_modality_sample(modality: str, grp: h5py.Group) -> dict:
    """Load a single sample based on modality."""
    if modality == "ct":
        return _load_ct(grp)
    elif modality == "mri":
        return _load_mri(grp)
    elif modality == "sd_cassi":
        return _load_cassi(grp)
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def _load_ct(grp: h5py.Group) -> dict:
    x_true = np.array(grp["x_true"])
    keys = list(grp.keys())
    # Handle both canonical (sinogram_measured, angles_nominal) and
    # legacy (y, H_ideal) key formats
    if "sinogram_measured" in keys:
        sino = np.array(grp["sinogram_measured"])
    else:
        sino = np.array(grp["y"])
    if "angles_nominal" in keys:
        angles = np.array(grp["angles_nominal"])
    elif "H_ideal" in keys:
        # Legacy: H_ideal stores angles in degrees or radians
        angles = np.array(grp["H_ideal"])
        # Convert to radians if angles look like degrees (max > 2π)
        if angles.max() > 2 * np.pi:
            angles = np.deg2rad(angles)
    else:
        # Generate uniform angles
        n_angles = sino.shape[0]
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    return {
        "measurements": sino,
        "calibration": {"angles_rad": angles},
        "x_true": x_true,
        "shape_info": (
            f"sinogram shape: {sino.shape} (n_angles={sino.shape[0]}, "
            f"n_det={sino.shape[1]}), angles: {angles.shape}, "
            f"image size to reconstruct: {x_true.shape}"
        ),
    }


def _load_mri(grp: h5py.Group) -> dict:
    x_true = np.array(grp["x_true"])
    kspace = np.array(grp["kspace_undersampled"])
    coil_maps = np.array(grp["coil_maps"])
    mask = np.array(grp["mask"])
    return {
        "measurements": kspace,
        "calibration": {"coil_maps": coil_maps, "mask": mask},
        "x_true": x_true,
        "shape_info": (
            f"k-space shape: {kspace.shape} (n_coils={kspace.shape[0]}, "
            f"ny={kspace.shape[1]}, nx={kspace.shape[2]}), dtype=complex, "
            f"coil_maps: {coil_maps.shape} complex, "
            f"undersampling mask: {mask.shape}, "
            f"image size to reconstruct: {x_true.shape}"
        ),
    }


def _load_cassi(grp: h5py.Group) -> dict:
    x_true = np.array(grp["x_true"])
    y = np.array(grp["y"])
    H = np.array(grp["H_ideal"])
    return {
        "measurements": y,
        "calibration": {"coded_aperture_mask": H},
        "x_true": x_true,
        "shape_info": (
            f"measurement shape: {y.shape} (ny={y.shape[0]}, "
            f"nx_compressed={y.shape[1]}), "
            f"coded aperture mask: {H.shape}, "
            f"spectral cube to reconstruct: {x_true.shape} "
            f"(ny, nx, n_bands={x_true.shape[2]})"
        ),
    }


def get_design_brief(modality: str) -> str:
    """Return the one-sentence design brief for a modality."""
    briefs = {
        "ct": (
            "Design a computational imaging system for parallel-beam CT "
            "with Poisson noise. Specify the forward model, select "
            "appropriate reconstruction algorithm, and produce "
            "reconstructed images."
        ),
        "mri": (
            "Design a computational imaging system for multi-coil MRI "
            "with 4× Cartesian undersampling. Specify the forward model, "
            "select appropriate reconstruction algorithm, and produce "
            "reconstructed images."
        ),
        "sd_cassi": (
            "Design a computational imaging system for coded-aperture "
            "snapshot spectral imaging (CASSI). Specify the forward model, "
            "select appropriate reconstruction algorithm, and produce "
            "reconstructed images."
        ),
    }
    return briefs[modality]
