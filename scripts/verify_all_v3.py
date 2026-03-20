"""Comprehensive verification of all solvers across 168 modalities (v3).

Tests every solver entry in every YAML config using the standard
(y, physics, cfg) -> (result, info) calling convention.
"""

import sys
import json
import importlib
import traceback
import time
import numpy as np
import yaml
from pathlib import Path

CONFIGS_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmarks\configs")
OUT_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmark_results\solver_verification_v3.json")

# Category mapping from YAML category_module
CATEGORY_MODULES = {
    "microscopy_psf": "microscopy_psf",
    "medical_ct_radon": "medical_ct_radon",
    "compressive_mask": "compressive_mask",
    "medical_mri_kspace": "medical_mri_kspace",
    "remote_sensing_sar": "remote_sensing_sar",
    "electron_ctf": "electron_ctf",
    "scanning_probe": "scanning_probe",
}


class PhysicsOperator:
    """Generic physics operator for testing solvers."""

    def __init__(self, category, n=64, m=32):
        self.category = category
        self.n = n
        self.m = m
        rng = np.random.default_rng(42)

        if category == "medical_mri_kspace":
            self.x_shape = (n, n)
            self.mask = (rng.random((n, n)) > 0.5).astype(np.float32)
        elif category == "compressive_mask":
            self.A = rng.standard_normal((m, n * n)).astype(np.float32) * 0.01
            self.x_shape = (n, n)
        elif category == "medical_ct_radon":
            self.x_shape = (n, n)
            self.n_angles = 90
            self.n_det = n
        else:
            self.x_shape = (n, n)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        if hasattr(self, 'A'):
            return self.A @ x
        return x[:self.n * self.n]

    def adjoint(self, y):
        y = np.asarray(y, dtype=np.float32).ravel()
        if hasattr(self, 'A'):
            return self.A.T @ y
        n2 = self.n * self.n
        if len(y) < n2:
            out = np.zeros(n2, dtype=np.float32)
            out[:len(y)] = y
            return out
        return y[:n2]

    def info(self):
        d = {"x_shape": self.x_shape}
        if hasattr(self, 'mask'):
            d["mask"] = self.mask
        return d


def make_measurement(category, n=64, m=32):
    """Create synthetic measurement for given category."""
    rng = np.random.default_rng(123)
    if category == "medical_mri_kspace":
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64)
    elif category == "compressive_mask":
        A = rng.standard_normal((m, n * n)).astype(np.float32) * 0.01
        x = rng.standard_normal(n * n).astype(np.float32)
        return (A @ x).astype(np.float32)
    else:
        return rng.standard_normal((n, n)).astype(np.float32)


def test_solver_entry(module_name, function_name, gpu_flag, category):
    """Test a single solver entry. Returns (status, message, exec_time)."""
    # Import
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, function_name)
    except Exception as e:
        return "import_error", str(e)[:200], 0

    # Skip GPU-only
    if gpu_flag:
        try:
            import torch
            if not torch.cuda.is_available():
                return "skipped_gpu", "", 0
        except ImportError:
            return "skipped_gpu", "", 0

    # Create test data
    n, m = 64, 32
    op = PhysicsOperator(category, n=n, m=m)
    y = make_measurement(category, n=n, m=m)
    cfg = {}

    # Call
    try:
        t0 = time.time()
        result = fn(y, op, cfg)
        dt = time.time() - t0

        if not isinstance(result, tuple) or len(result) != 2:
            return "bad_return", f"Got {type(result).__name__} len={len(result) if isinstance(result, tuple) else 'N/A'}", dt

        arr, info = result
        if not isinstance(arr, np.ndarray):
            return "bad_return_type", f"Result type: {type(arr).__name__}", dt

        return "verified", f"shape={arr.shape}", dt

    except Exception as e:
        return "error", str(e)[:200], 0


def main():
    results = {}
    stats = {"verified": 0, "skipped_gpu": 0, "error": 0, "import_error": 0,
             "bad_return": 0, "bad_return_type": 0}

    yaml_files = sorted(CONFIGS_DIR.glob("*.yaml"))
    yaml_files = [f for f in yaml_files if not f.name.startswith("_")]

    for yf in yaml_files:
        modality = yf.stem
        with open(yf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "solvers" not in data or not data["solvers"]:
            continue

        category = data.get("category_module", "microscopy_psf")
        mod_results = {}

        for solver_key, solver_cfg in data["solvers"].items():
            if not isinstance(solver_cfg, dict):
                continue

            module_name = solver_cfg.get("module", "")
            function_name = solver_cfg.get("function", "")
            gpu_flag = solver_cfg.get("gpu", False)

            status, msg, dt = test_solver_entry(module_name, function_name, gpu_flag, category)
            stats[status] = stats.get(status, 0) + 1

            mod_results[solver_key] = {
                "status": status,
                "module": module_name,
                "function": function_name,
                "message": msg,
                "exec_time": round(dt, 3),
            }

            icon = {"verified": " OK ", "skipped_gpu": "SKIP", "error": "FAIL",
                    "import_error": "XIMP", "bad_return": "XRET", "bad_return_type": "XTYP"}.get(status, "????")
            if status not in ("verified", "skipped_gpu"):
                print(f"[{icon}] {modality}/{solver_key}: {function_name} -> {msg}")

        results[modality] = mod_results

    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"Total: {total} solver entries across {len(results)} modalities")
    print(f"  Verified: {stats['verified']}")
    print(f"  GPU-skipped: {stats['skipped_gpu']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Import errors: {stats['import_error']}")
    print(f"  Bad return: {stats.get('bad_return', 0) + stats.get('bad_return_type', 0)}")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "local_cpu",
        "stats": stats,
        "modalities": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
