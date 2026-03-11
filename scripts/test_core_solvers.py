#!/usr/bin/env python3
"""Test core solvers on representative modalities.

Focuses on: CT, MRI, PET, Lightsheet
These represent: Tomography, k-space, tomography, microscopy

GPU Server: Baseline solver testing with proper inputs
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import yaml
import numpy as np
import h5py

# Add paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCHMARK_DIR = ROOT / "datasets" / "benchmark"
CONFIG_DIR = ROOT / "benchmarks" / "configs"
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Core modalities to test
CORE_MODALITIES = {
    "ct": {
        "description": "X-ray Computed Tomography (Radon/Tomography)",
        "measurement_key": "measurement",
        "groundtruth_key": "groundtruth",
        "geometry_key": "angles"
    },
    "mri": {
        "description": "Magnetic Resonance Imaging (k-space)",
        "measurement_key": "kspace_undersampled",
        "groundtruth_key": "x_true",
        "geometry_key": "mask"
    },
    "pet": {
        "description": "Positron Emission Tomography (Radon)",
        "measurement_key": "sinogram_measured",
        "groundtruth_key": "x_true",
        "geometry_key": "angles_deg"
    },
    "lightsheet": {
        "description": "Lightsheet Microscopy (PSF deconvolution)",
        "measurement_key": "y",
        "groundtruth_key": "x_true",
        "geometry_key": "H_ideal"
    }
}


def load_config(modality_id: str) -> Optional[Dict]:
    """Load modality config."""
    config_path = CONFIG_DIR / f"{modality_id}.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample(modality_id: str, tier: str = "public", sample_idx: int = 0) -> Optional[Dict]:
    """Load single sample from benchmark dataset."""
    tier_dir = BENCHMARK_DIR / modality_id / tier
    if not tier_dir.exists():
        return None

    # Try HDF5 first
    h5_files = list(tier_dir.glob("*_challenge_*.h5"))
    if h5_files:
        try:
            with h5py.File(h5_files[0], "r") as f:
                sample_keys = sorted([k for k in f.keys() if k.startswith("sample")])
                if sample_keys and sample_idx < len(sample_keys):
                    sample_key = sample_keys[sample_idx]
                    data = {}
                    for key in f[sample_key].keys():
                        data[key] = f[sample_key][key][:]
                    return {
                        "format": "hdf5",
                        "sample": sample_key,
                        "data": data
                    }
        except Exception as e:
            print(f"      Error loading HDF5: {e}")

    # Try directory format
    sample_dirs = sorted([d for d in tier_dir.iterdir()
                         if d.is_dir() and d.name.startswith("sample_")])
    if sample_dirs and sample_idx < len(sample_dirs):
        try:
            sample_dir = sample_dirs[sample_idx]
            data = {}
            for npy_file in sample_dir.glob("*.npy"):
                data[npy_file.stem] = np.load(npy_file)
            if data:
                return {
                    "format": "directory",
                    "sample": sample_dir.name,
                    "data": data
                }
        except Exception as e:
            print(f"      Error loading directory: {e}")

    return None


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> Optional[float]:
    """Compute PSNR."""
    try:
        if np.iscomplexobj(gt):
            gt = np.abs(gt)
        if np.iscomplexobj(recon):
            recon = np.abs(recon)

        if gt.shape != recon.shape:
            return None

        mse = np.mean((gt - recon) ** 2)
        if mse < 1e-12:
            return 100.0

        data_range = gt.max() - gt.min()
        if data_range == 0:
            return 0.0

        return float(10 * np.log10(data_range ** 2 / mse))
    except Exception:
        return None


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> Optional[float]:
    """Compute SSIM."""
    try:
        if np.iscomplexobj(gt):
            gt = np.abs(gt)
        if np.iscomplexobj(recon):
            recon = np.abs(recon)

        if gt.shape != recon.shape:
            return None

        data_range = gt.max() - gt.min()
        if data_range == 0:
            return 0.0

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu_x = gt.mean()
        mu_y = recon.mean()
        var_x = gt.var()
        var_y = recon.var()
        cov_xy = np.mean((gt - mu_x) * (recon - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))

        return float(ssim)
    except Exception:
        return None


def test_ct(sample: Dict, solver_config: Dict) -> Optional[Dict]:
    """Test CT FBP solver."""
    try:
        from pwm_core.recon.ct_solvers import fbp_2d

        measurement = sample["data"].get("measurement")
        angles = sample["data"].get("angles")
        groundtruth = sample["data"].get("groundtruth")

        if measurement is None or angles is None or groundtruth is None:
            return {"status": "missing_data"}

        # Convert angles to radians if needed
        if angles.max() > 2 * np.pi:
            angles_rad = np.deg2rad(angles)
        else:
            angles_rad = angles

        start = time.time()
        reconstruction = fbp_2d(measurement, angles_rad, filter_type="ramlak", output_size=groundtruth.shape[0])
        exec_time = time.time() - start

        psnr = compute_psnr(groundtruth, reconstruction)
        ssim = compute_ssim(groundtruth, reconstruction)

        if psnr is not None and ssim is not None:
            return {
                "status": "completed",
                "psnr_db": psnr,
                "ssim": ssim,
                "exec_time_sec": exec_time,
                "result_shape": list(reconstruction.shape)
            }
        else:
            return {"status": "metric_error"}
    except Exception as e:
        return {"status": f"error: {str(e)[:50]}"}


def test_mri(sample: Dict, solver_config: Dict) -> Optional[Dict]:
    """Test MRI zero-filled reconstruction (baseline)."""
    try:
        from pwm_core.recon.mri_solvers import zero_filled_reconstruction

        kspace = sample["data"].get("kspace_undersampled")
        mask = sample["data"].get("mask")
        groundtruth = sample["data"].get("x_true")

        if kspace is None or groundtruth is None:
            return {"status": "missing_data"}

        start = time.time()
        reconstruction = zero_filled_reconstruction(kspace, mask=mask)
        exec_time = time.time() - start

        # Convert to image domain if needed
        if np.iscomplexobj(reconstruction):
            reconstruction = np.abs(reconstruction)
        if np.iscomplexobj(groundtruth):
            groundtruth = np.abs(groundtruth)

        psnr = compute_psnr(groundtruth, reconstruction)
        ssim = compute_ssim(groundtruth, reconstruction)

        if psnr is not None and ssim is not None:
            return {
                "status": "completed",
                "psnr_db": psnr,
                "ssim": ssim,
                "exec_time_sec": exec_time,
                "result_shape": list(np.abs(reconstruction).shape)
            }
        else:
            return {"status": "metric_error"}
    except Exception as e:
        return {"status": f"error: {str(e)[:50]}"}


def test_pet(sample: Dict, solver_config: Dict) -> Optional[Dict]:
    """Test PET FBP solver."""
    try:
        from pwm_core.recon.ct_solvers import fbp_2d

        sinogram = sample["data"].get("sinogram_measured")
        angles = sample["data"].get("angles_deg")
        groundtruth = sample["data"].get("x_true")

        if sinogram is None or angles is None or groundtruth is None:
            return {"status": "missing_data"}

        # Convert angles to radians if needed
        if angles.max() > 2 * np.pi:
            angles_rad = np.deg2rad(angles)
        else:
            angles_rad = angles

        start = time.time()
        reconstruction = fbp_2d(sinogram, angles_rad, filter_type="ramlak", output_size=groundtruth.shape[0])
        exec_time = time.time() - start

        psnr = compute_psnr(groundtruth, reconstruction)
        ssim = compute_ssim(groundtruth, reconstruction)

        if psnr is not None and ssim is not None:
            return {
                "status": "completed",
                "psnr_db": psnr,
                "ssim": ssim,
                "exec_time_sec": exec_time,
                "result_shape": list(reconstruction.shape)
            }
        else:
            return {"status": "metric_error"}
    except Exception as e:
        return {"status": f"error: {str(e)[:50]}"}


def test_lightsheet(sample: Dict, solver_config: Dict) -> Optional[Dict]:
    """Test Lightsheet destriping solver (Fourier notch baseline)."""
    try:
        from pwm_core.recon.lightsheet_solver import fourier_notch_destripe

        measurement = sample["data"].get("y")
        groundtruth = sample["data"].get("x_true")

        if measurement is None or groundtruth is None:
            return {"status": "missing_data"}

        start = time.time()
        reconstruction = fourier_notch_destripe(measurement, stripe_direction="horizontal")
        exec_time = time.time() - start

        psnr = compute_psnr(groundtruth, reconstruction)
        ssim = compute_ssim(groundtruth, reconstruction)

        if psnr is not None and ssim is not None:
            return {
                "status": "completed",
                "psnr_db": psnr,
                "ssim": ssim,
                "exec_time_sec": exec_time,
                "result_shape": list(reconstruction.shape)
            }
        else:
            return {"status": "metric_error"}
    except Exception as e:
        return {"status": f"error: {str(e)[:50]}"}


def main():
    print("\n" + "="*80)
    print("PWM5 GPU SERVER - CORE SOLVER TESTING")
    print("Testing: CT (Radon), MRI (k-space), PET (Radon), Lightsheet (PSF)")
    print("="*80 + "\n")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tier": "public",
        "gpu_server": os.environ.get("HOSTNAME", "local-gpu-server"),
        "modalities": {},
        "summary": {
            "total_modalities": len(CORE_MODALITIES),
            "completed": 0,
            "failed": 0,
            "total_tests": 0,
            "tests_passed": 0
        }
    }

    # Test functions by modality
    test_functions = {
        "ct": test_ct,
        "mri": test_mri,
        "pet": test_pet,
        "lightsheet": test_lightsheet
    }

    for modality_id, modality_info in CORE_MODALITIES.items():
        print(f"{modality_id:15} - {modality_info['description']}")

        config = load_config(modality_id)
        if not config:
            print(f"  ERROR: Config not found\n")
            all_results["summary"]["failed"] += 1
            continue

        sample = load_sample(modality_id, "public", 0)
        if not sample:
            print(f"  ERROR: Sample not found\n")
            all_results["summary"]["failed"] += 1
            continue

        solvers = config.get("solvers", {})
        mod_result = {
            "description": modality_info["description"],
            "sample": sample["sample"],
            "solvers": {}
        }

        test_fn = test_functions.get(modality_id)
        if not test_fn:
            print(f"  ERROR: No test function\n")
            all_results["summary"]["failed"] += 1
            continue

        # Test traditional_cpu solver
        solver_name = "traditional_cpu"
        if solver_name in solvers:
            print(f"  {solver_name:20} ", end="", flush=True)

            result = test_fn(sample, solvers[solver_name])
            mod_result["solvers"][solver_name] = result

            all_results["summary"]["total_tests"] += 1

            if result.get("psnr_db") is not None:
                print(f"PASS - PSNR={result['psnr_db']:.2f} dB, SSIM={result['ssim']:.4f}, Time={result['exec_time_sec']:.2f}s")
                all_results["summary"]["tests_passed"] += 1
            else:
                print(f"FAIL - {result.get('status', 'unknown')}")

        all_results["modalities"][modality_id] = mod_result
        all_results["summary"]["completed"] += 1
        print()

    # Save results
    results_path = RESULTS_DIR / "core_solvers_test.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Modalities tested:      {all_results['summary']['completed']}")
    print(f"Failed:                 {all_results['summary']['failed']}")
    print(f"Total tests:            {all_results['summary']['total_tests']}")
    print(f"Tests passed:           {all_results['summary']['tests_passed']}")
    if all_results['summary']['total_tests'] > 0:
        success = 100 * all_results['summary']['tests_passed'] / all_results['summary']['total_tests']
        print(f"Success rate:           {success:.1f}%")
    print("="*80)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
