"""Verify that the YAML registration fixes resolved all interface issues.

Tests the previously-broken solvers by importing and calling them with
the standard (y, physics, cfg) interface.
"""

import sys
import json
import importlib
import traceback
import numpy as np
from pathlib import Path

CONFIGS_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmarks\configs")

# Previously broken modality/solver pairs to test
TEST_CASES = [
    # device_cfg_mismatch (MRI zero-filled)
    ("mri", "traditional_cpu"),
    ("diffusion_mri", "traditional_cpu"),
    ("fmri", "traditional_cpu"),
    ("elastography", "traditional_cpu"),
    ("mrs", "traditional_cpu"),
    ("asl_mri", "asl_dl"),
    ("cest_mri", "cest_dl"),
    ("swi", "swi_dl"),
    ("us_mri", "us_mri_dl"),
    ("mra", "mra_dl"),
    ("mr_fingerprinting", "mrf_dl"),
    # device_cfg_mismatch (RED-CNN)
    ("ct", "famous_dl"),
    # device_cfg_mismatch (DeStripe)
    ("lightsheet", "famous_dl"),
    # return_tuple_mismatch
    ("cacti", "traditional_cpu"),
    ("cacti", "best_quality"),
    ("cacti", "famous_dl"),
    ("cassi", "traditional_cpu"),
    ("cassi", "best_quality"),
    ("flim", "best_quality"),
    ("panorama", "famous_dl"),
    ("ptychography", "best_quality"),
    ("sim", "best_quality"),
    ("sim", "famous_dl"),
    ("sim", "small_gpu"),
    # operator_attr errors
    ("dexa", "traditional_cpu"),
    ("ebsd", "traditional_cpu"),
    ("lidar", "traditional_cpu"),
    ("matrix", "traditional_cpu"),
    ("matrix", "famous_dl"),
    ("tem", "traditional_cpu"),
    ("tof_camera", "traditional_cpu"),
    # SPC
    ("spc", "best_quality"),
    ("spc", "famous_dl"),
    # lightsheet vsnr
    ("lightsheet", "best_quality"),
    # MRI cs_mri_wavelet
    ("mri", "best_quality"),
    # MoDL
    ("mri", "famous_dl"),
    # CARE
    ("widefield", "famous_dl"),
    ("confocal_3d", "famous_dl"),
    ("confocal_livecell", "famous_dl"),
]


class SimpleOperator:
    """Minimal physics operator for testing."""
    def __init__(self, n=64, m=32, category="microscopy_psf"):
        self.n = n
        self.m = m
        self.category = category
        rng = np.random.default_rng(42)

        if category == "medical_mri_kspace":
            self.x_shape = (n, n)
            self.mask = rng.random((n, n)) > 0.5
        elif category == "compressive_mask":
            self.A = rng.standard_normal((m, n * n)).astype(np.float32) * 0.01
            self.x_shape = (n, n)
        else:
            self.x_shape = (n, n)
            self._psf = rng.standard_normal((n, n)).astype(np.float32) * 0.01

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        if hasattr(self, 'A'):
            return self.A @ x
        return x

    def adjoint(self, y):
        y = np.asarray(y, dtype=np.float32).ravel()
        if hasattr(self, 'A'):
            return self.A.T @ y
        return y

    def info(self):
        d = {"x_shape": self.x_shape}
        if hasattr(self, 'mask'):
            d["mask"] = self.mask
        return d


# Category mapping for modalities
CATEGORY_MAP = {
    "mri": "medical_mri_kspace",
    "diffusion_mri": "medical_mri_kspace",
    "fmri": "medical_mri_kspace",
    "elastography": "medical_mri_kspace",
    "mrs": "medical_mri_kspace",
    "asl_mri": "medical_mri_kspace",
    "cest_mri": "medical_mri_kspace",
    "swi": "medical_mri_kspace",
    "us_mri": "medical_mri_kspace",
    "mra": "medical_mri_kspace",
    "mr_fingerprinting": "medical_mri_kspace",
    "spc": "compressive_mask",
    "matrix": "compressive_mask",
    "cassi": "compressive_mask",
    "cacti": "compressive_mask",
}


def load_yaml(modality):
    fpath = CONFIGS_DIR / f"{modality}.yaml"
    if not fpath.exists():
        return None
    import yaml
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_solver(modality, solver_key):
    """Test a single solver entry."""
    data = load_yaml(modality)
    if not data or "solvers" not in data:
        return "no_yaml", ""

    solvers = data["solvers"]
    if solver_key not in solvers:
        return "no_solver_key", ""

    solver = solvers[solver_key]
    module_name = solver.get("module", "")
    function_name = solver.get("function", "")
    gpu = solver.get("gpu", False)

    # Import
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, function_name)
    except Exception as e:
        return "import_error", str(e)[:200]

    # Skip GPU-only solvers
    if gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                return "gpu_skipped", ""
        except ImportError:
            return "gpu_skipped", ""

    # Create test data
    category = CATEGORY_MAP.get(modality, "microscopy_psf")
    op = SimpleOperator(n=64, m=32, category=category)

    if category == "medical_mri_kspace":
        y = np.random.randn(64, 64).astype(np.float32) + 1j * np.random.randn(64, 64).astype(np.float32)
        y = y.astype(np.complex64)
    elif category == "compressive_mask":
        y = op.A @ np.random.randn(64 * 64).astype(np.float32)
    else:
        y = np.random.randn(64, 64).astype(np.float32)

    cfg = {}

    # Call solver
    try:
        result = fn(y.astype(np.float32) if not np.iscomplexobj(y) else y, op, cfg)
        if not isinstance(result, tuple) or len(result) != 2:
            return "bad_return", f"Expected (result, info) tuple, got {type(result)}"
        arr, info = result
        if not isinstance(arr, np.ndarray):
            return "bad_return", f"First element not ndarray: {type(arr)}"
        return "ok", f"shape={arr.shape}"
    except Exception as e:
        return "error", str(e)[:200]


def main():
    results = {"ok": 0, "error": 0, "gpu_skipped": 0, "other": 0}
    errors = []

    for modality, solver_key in TEST_CASES:
        status, msg = test_solver(modality, solver_key)
        tag = f"{modality}/{solver_key}"

        if status == "ok":
            results["ok"] += 1
            print(f"  OK  {tag}: {msg}")
        elif status == "gpu_skipped":
            results["gpu_skipped"] += 1
            print(f" SKIP {tag} (GPU)")
        elif status == "error" or status == "import_error":
            results["error"] += 1
            errors.append((tag, status, msg))
            print(f" FAIL {tag}: {msg}")
        else:
            results["other"] += 1
            errors.append((tag, status, msg))
            print(f" {status.upper()} {tag}: {msg}")

    total = sum(results.values())
    print(f"\n{'='*60}")
    print(f"Results: {results['ok']}/{total} OK, {results['error']} errors, "
          f"{results['gpu_skipped']} GPU-skipped, {results['other']} other")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for tag, status, msg in errors:
            print(f"  {tag}: [{status}] {msg}")

    # Save results
    out = {
        "summary": results,
        "errors": [{"tag": t, "status": s, "msg": m} for t, s, m in errors],
    }
    out_path = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmark_results\fix_verification.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
