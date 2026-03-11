#!/usr/bin/env python3
"""Execute actual solver algorithms on benchmark public tier.

Runs registered solvers for each modality on public tier datasets.
Computes PSNR, SSIM, and execution time.

GPU Server: Real algorithm testing (not mocks)
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

# Modalities with complete data
COMPLETE_MODALITIES = [
    "cacti", "confocal_3d", "cryo_em", "ct", "diffusion_mri",
    "endoscopy", "fmri", "fundus", "lightsheet", "mammography",
    "mri", "oct", "palm_storm", "pet", "sim", "spect", "sted",
    "two_photon", "ultrasound"
]


def load_config(modality_id: str) -> Optional[Dict]:
    """Load modality config."""
    config_path = CONFIG_DIR / f"{modality_id}.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_solvers(config: Dict) -> Dict:
    """Extract solver dict from config."""
    return config.get("solvers", {})


def load_dataset_sample(modality_id: str, tier: str = "public", sample_idx: int = 0) -> Optional[Dict]:
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
            print(f"    Error loading HDF5: {e}")

    # Try directory format (CT)
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
            print(f"    Error loading directory: {e}")

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


def execute_solver(modality_id: str, solver_name: str, solver_config: Dict, sample: Dict) -> Optional[Dict]:
    """Execute single solver and return metrics."""
    module_name = solver_config.get("module", "")
    function_name = solver_config.get("function", "")
    params = solver_config.get("params", "")

    if not module_name or not function_name:
        return {"status": "no_function"}

    try:
        # Import solver module
        module = __import__(module_name, fromlist=[function_name])
        solver_fn = getattr(module, function_name, None)

        if solver_fn is None:
            return {"status": "function_not_found"}

        # Get groundtruth and measurement
        data = sample["data"]
        gt_candidates = ["x_true", "groundtruth", "gt", "image"]
        gt = None
        for key in gt_candidates:
            if key in data:
                gt = data[key]
                break

        if gt is None:
            return {"status": "no_groundtruth"}

        # Modality-specific solver execution
        result = None
        exec_time = 0.0

        # Try to call solver with appropriate inputs
        if modality_id == "ct":
            if "measurement" in data:
                start = time.time()
                result = solver_fn(data["measurement"], data.get("angles", None))
                exec_time = time.time() - start

        elif modality_id == "mri":
            if "kspace_undersampled" in data:
                start = time.time()
                result = solver_fn(data["kspace_undersampled"], data.get("mask", None))
                exec_time = time.time() - start

        elif modality_id == "pet":
            if "sinogram_measured" in data or "measurement" in data:
                meas = data.get("sinogram_measured") or data.get("measurement")
                start = time.time()
                result = solver_fn(meas, data.get("angles", None))
                exec_time = time.time() - start

        elif modality_id in ["ultrasound", "oct"]:
            # These may use measurement directly
            measurement_keys = ["bmode_measured", "bscan_measured", "measurement", "y"]
            for key in measurement_keys:
                if key in data:
                    start = time.time()
                    result = solver_fn(data[key])
                    exec_time = time.time() - start
                    break

        else:
            # Generic attempt with measurement
            measurement_keys = ["measurement", "y", "sinogram_measured", "bmode_measured", "bscan_measured"]
            for key in measurement_keys:
                if key in data:
                    start = time.time()
                    result = solver_fn(data[key])
                    exec_time = time.time() - start
                    break

        if result is None:
            return {"status": "execution_failed", "exec_time": exec_time}

        # Compute metrics
        psnr = compute_psnr(gt, result)
        ssim = compute_ssim(gt, result)

        if psnr is not None and ssim is not None:
            return {
                "status": "completed",
                "psnr_db": psnr,
                "ssim": ssim,
                "exec_time_sec": exec_time,
                "result_shape": list(result.shape)
            }
        else:
            return {"status": "metric_error", "exec_time": exec_time}

    except ImportError as e:
        return {"status": f"import_error: {str(e)[:40]}"}
    except Exception as e:
        return {"status": f"error: {str(e)[:40]}"}


def main():
    print("\n" + "="*80)
    print("PWM5 GPU SERVER - ACTUAL SOLVER EXECUTION")
    print("Testing real reconstruction algorithms on public tier")
    print("="*80 + "\n")

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
            "total_solvers": 0,
            "solvers_executed": 0
        }
    }

    for modality_id in COMPLETE_MODALITIES:
        config = load_config(modality_id)
        if not config:
            continue

        solvers = get_solvers(config)
        sample = load_dataset_sample(modality_id, "public", 0)

        if not sample:
            print(f"{modality_id:20} FAILED (no data)")
            all_results["summary"]["failed"] += 1
            continue

        print(f"{modality_id:20} ", end="", flush=True)

        mod_result = {
            "config": config.get("display_name", modality_id),
            "solvers": {}
        }

        num_executed = 0
        for solver_name, solver_config in solvers.items():
            result = execute_solver(modality_id, solver_name, solver_config, sample)
            mod_result["solvers"][solver_name] = result

            if result.get("psnr_db") is not None:
                num_executed += 1

        all_results["modalities"][modality_id] = mod_result
        all_results["summary"]["total_solvers"] += len(solvers)
        all_results["summary"]["solvers_executed"] += num_executed

        if num_executed == len(solvers):
            print(f"OK ({num_executed}/{len(solvers)} solvers)")
            all_results["summary"]["completed"] += 1
        elif num_executed > 0:
            print(f"PARTIAL ({num_executed}/{len(solvers)} solvers)")
            all_results["summary"]["partial"] += 1
        else:
            print(f"NO EXECUTION ({len(solvers)} solvers)")
            all_results["summary"]["partial"] += 1

    # Save results
    results_path = RESULTS_DIR / "actual_solver_execution.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Modalities completed:   {all_results['summary']['completed']}")
    print(f"Partial results:        {all_results['summary']['partial']}")
    print(f"Failed:                 {all_results['summary']['failed']}")
    print(f"Total solvers:          {all_results['summary']['total_solvers']}")
    print(f"Executed:               {all_results['summary']['solvers_executed']}")
    success_rate = 100 * all_results['summary']['solvers_executed'] / max(1, all_results['summary']['total_solvers'])
    print(f"Execution rate:         {success_rate:.1f}%")
    print("="*80)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
