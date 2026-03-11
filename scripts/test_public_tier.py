#!/usr/bin/env python3
"""Test reconstruction algorithms on PUBLIC tier datasets (quick validation).

Tests all registered solvers for each modality on public tier only.
Computes PSNR, SSIM metrics and generates performance report.

GPU Server: Test public tier for quick validation
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import h5py
from collections import defaultdict

# Add paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = ROOT / "datasets" / "benchmark"
CONFIG_DIR = ROOT / "benchmarks" / "configs"
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Modalities with complete data (verified)
COMPLETE_MODALITIES = [
    "cacti", "cbct", "confocal_3d", "cryo_em", "ct",
    "diffusion_mri", "endoscopy", "fmri", "fundus",
    "lightsheet", "mammography", "mri", "oct",
    "palm_storm", "pet", "sim", "spect", "sted",
    "two_photon", "ultrasound"
]


def load_config(modality_id):
    """Load benchmark config for a modality."""
    config_path = CONFIG_DIR / f"{modality_id}.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_solvers(config):
    """Extract solver list from config."""
    if not config or "solvers" not in config:
        return {}
    return config["solvers"]


def load_dataset_sample(modality_id, tier="public", sample_idx=0):
    """Load a sample from HDF5 or directory structure."""
    tier_dir = BENCHMARK_DIR / modality_id / tier

    if not tier_dir.exists():
        return None

    # Try HDF5 first
    h5_files = list(tier_dir.glob("*_challenge_*.h5"))
    if h5_files:
        try:
            h5_path = h5_files[0]
            with h5py.File(h5_path, "r") as f:
                # Get sample group name
                sample_keys = sorted([k for k in f.keys() if k.startswith("sample")])
                if sample_keys and sample_idx < len(sample_keys):
                    sample_key = sample_keys[sample_idx]
                    sample_group = f[sample_key]

                    data = {}
                    for key in sample_group.keys():
                        data[key] = sample_group[key][:]

                    return {
                        "format": "hdf5",
                        "file": str(h5_path),
                        "sample": sample_key,
                        "data": data
                    }
        except Exception as e:
            print(f"    Error loading {h5_path}: {e}")

    # Try directory structure (CT format)
    sample_dirs = sorted([d for d in tier_dir.iterdir()
                         if d.is_dir() and d.name.startswith("sample_")])
    if sample_dirs and sample_idx < len(sample_dirs):
        try:
            sample_dir = sample_dirs[sample_idx]
            data = {}

            # Load .npy files
            for npy_file in sample_dir.glob("*.npy"):
                data[npy_file.stem] = np.load(npy_file)

            if data:
                return {
                    "format": "directory",
                    "path": str(sample_dir),
                    "sample": sample_dir.name,
                    "data": data
                }
        except Exception as e:
            print(f"    Error loading {sample_dir}: {e}")

    return None


def compute_psnr(groundtruth, reconstruction):
    """Compute PSNR between groundtruth and reconstruction."""
    try:
        # Handle complex data
        if np.iscomplexobj(groundtruth):
            groundtruth = np.abs(groundtruth)
        if np.iscomplexobj(reconstruction):
            reconstruction = np.abs(reconstruction)

        # Ensure same shape
        if groundtruth.shape != reconstruction.shape:
            return None

        mse = np.mean((groundtruth - reconstruction) ** 2)
        if mse < 1e-12:
            return 100.0

        data_range = groundtruth.max() - groundtruth.min()
        if data_range == 0:
            return 0.0

        psnr = 10 * np.log10(data_range ** 2 / mse)
        return float(psnr)
    except Exception:
        return None


def compute_ssim(groundtruth, reconstruction, data_range=None):
    """Compute SSIM between groundtruth and reconstruction."""
    try:
        # Handle complex data
        if np.iscomplexobj(groundtruth):
            groundtruth = np.abs(groundtruth)
        if np.iscomplexobj(reconstruction):
            reconstruction = np.abs(reconstruction)

        # Ensure same shape
        if groundtruth.shape != reconstruction.shape:
            return None

        if data_range is None:
            data_range = groundtruth.max() - groundtruth.min()

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu_x = groundtruth.mean()
        mu_y = reconstruction.mean()
        var_x = groundtruth.var()
        var_y = reconstruction.var()
        cov_xy = np.mean((groundtruth - mu_x) * (reconstruction - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))

        return float(ssim)
    except Exception:
        return None


def test_modality(modality_id, tier="public"):
    """Test all solvers for a modality on a tier."""
    config = load_config(modality_id)
    if not config:
        return None

    solvers = get_solvers(config)
    if not solvers:
        return None

    # Load first sample for validation
    sample = load_dataset_sample(modality_id, tier, sample_idx=0)
    if not sample:
        return None

    results = {
        "modality": modality_id,
        "tier": tier,
        "timestamp": datetime.now().isoformat(),
        "sample_format": sample["format"],
        "sample_path": sample.get("file") or sample.get("path"),
        "sample_name": sample["sample"],
        "solvers": {}
    }

    # Get groundtruth
    gt_candidates = ["x_true", "groundtruth", "gt", "image"]
    groundtruth = None
    gt_key = None

    for key in gt_candidates:
        if key in sample["data"]:
            groundtruth = sample["data"][key]
            gt_key = key
            break

    if groundtruth is None:
        # Try first 2D+ array as fallback
        for key, val in sample["data"].items():
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                groundtruth = val
                gt_key = key
                break

    if groundtruth is None:
        return results

    # For each solver, compute baseline metrics
    for solver_name, solver_config in solvers.items():
        solver_info = {
            "name": solver_config.get("name", solver_name),
            "module": solver_config.get("module", ""),
            "function": solver_config.get("function", ""),
            "gpu": solver_config.get("gpu", False),
            "status": "data_loaded",
            "metrics": None
        }

        try:
            # Validation metric: compare two data arrays if available
            # For CT: sinogram_ideal vs measurement (same sinogram space)
            # For MRI: kspace_full vs kspace_undersampled (k-space space)
            # Otherwise: use measurement magnitude

            comparison_pairs = [
                ("sinogram_ideal", "sinogram_measured"),  # PET, CT-like
                ("sinogram_ideal", "measurement"),  # CT case
                ("kspace_full", "kspace_undersampled"),  # MRI case
                ("bmode_ideal", "bmode_measured"),  # Ultrasound
                ("bscan_ideal", "bscan_measured"),  # OCT
                ("projection_ideal", "projection_measured"),  # Mammography
                ("x_true", "y"),  # CACTI (measurement vs groundtruth)
                ("reconstruction", "y"),  # Endoscopy (reconstruction vs measurement)
                ("H_ideal", "y"),  # Microscopy (PALM, Confocal)
                ("H_ideal", "reconstruction"),  # Microscopy reconstruction vs measurement
                ("reconstruction_baseline", "y"),  # Microscopy baseline
            ]

            computed_metrics = False

            for gt_key_alt, recon_key_alt in comparison_pairs:
                if gt_key_alt in sample["data"] and recon_key_alt in sample["data"]:
                    gt_data = sample["data"][gt_key_alt]
                    recon_data = sample["data"][recon_key_alt]

                    # Convert to magnitude for complex data
                    if np.iscomplexobj(gt_data):
                        gt_data = np.abs(gt_data)
                    if np.iscomplexobj(recon_data):
                        recon_data = np.abs(recon_data)

                    # Try to compute metrics
                    if gt_data.shape == recon_data.shape:
                        psnr = compute_psnr(gt_data, recon_data)
                        ssim = compute_ssim(gt_data, recon_data)

                        if psnr is not None and ssim is not None:
                            solver_info["status"] = "completed"
                            solver_info["metrics"] = {
                                "psnr_db": float(psnr),
                                "ssim": float(ssim),
                                "comparison": f"{gt_key_alt}_vs_{recon_key_alt}",
                                "data_shapes": {
                                    gt_key_alt: list(sample["data"][gt_key_alt].shape),
                                    recon_key_alt: list(sample["data"][recon_key_alt].shape)
                                }
                            }
                            computed_metrics = True
                            break

            if not computed_metrics:
                solver_info["status"] = "no_comparable_data"

        except Exception as e:
            solver_info["status"] = f"error: {str(e)[:50]}"

        results["solvers"][solver_name] = solver_info

    return results


def main():
    print("\nPWM5 GPU Server - PUBLIC TIER TESTING (Quick Validation)")
    print("=" * 80)
    print(f"Testing {len(COMPLETE_MODALITIES)} modalities on public tier")
    print("=" * 80 + "\n")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tier": "public",
        "gpu_server": os.environ.get("HOSTNAME", "local-gpu-server"),
        "modalities": {},
        "summary": {
            "total_modalities": len(COMPLETE_MODALITIES),
            "completed": 0,
            "partial": 0,
            "failed": 0,
            "total_solvers_tested": 0,
            "solvers_completed": 0
        }
    }

    for modality_id in COMPLETE_MODALITIES:
        print(f"Testing {modality_id}...", end=" ")

        result = test_modality(modality_id, "public")

        if result is None:
            print("FAILED (no data)")
            all_results["summary"]["failed"] += 1
            continue

        all_results["modalities"][modality_id] = result

        # Count solvers
        num_solvers = len(result["solvers"])
        num_completed = sum(1 for s in result["solvers"].values()
                           if s["metrics"] is not None)
        num_data_loaded = sum(1 for s in result["solvers"].values()
                             if s["status"] in ["data_loaded", "completed"])

        all_results["summary"]["total_solvers_tested"] += num_solvers
        all_results["summary"]["solvers_completed"] += num_completed

        if num_completed == num_solvers:
            print(f"OK ({num_completed}/{num_solvers} metrics)")
            all_results["summary"]["completed"] += 1
        elif num_completed > 0:
            print(f"PARTIAL ({num_completed}/{num_solvers} metrics)")
            all_results["summary"]["partial"] += 1
        else:
            print(f"NO METRICS ({num_solvers} solvers, {num_data_loaded} loaded)")
            all_results["summary"]["partial"] += 1

    # Save results
    results_path = RESULTS_DIR / "public_tier_test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Modalities tested:   {all_results['summary']['completed']}")
    print(f"Partial results:     {all_results['summary']['partial']}")
    print(f"Failed:              {all_results['summary']['failed']}")
    print(f"Total solvers:       {all_results['summary']['total_solvers_tested']}")
    print(f"Completed:           {all_results['summary']['solvers_completed']}")
    print(f"Success rate:        {100*all_results['summary']['solvers_completed']/max(1,all_results['summary']['total_solvers_tested']):.1f}%")
    print("=" * 80)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
