"""Verify GPU-dependent solvers on local GTX 1660 Ti.

Tests all solver entries marked gpu=true in YAML configs.
"""

import sys
import json
import importlib
import time
import numpy as np
import yaml
import torch
from pathlib import Path

CONFIGS_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmarks\configs")
OUT_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\benchmark_results\gpu_verification.json")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class PhysicsOperator:
    def __init__(self, category, n=64, m=32):
        self.category = category
        self.n = n
        self.x_shape = (n, n)
        rng = np.random.default_rng(42)
        if category == "medical_mri_kspace":
            self.mask = (rng.random((n, n)) > 0.5).astype(np.float32)
        elif category == "compressive_mask":
            self.A = rng.standard_normal((m, n * n)).astype(np.float32) * 0.01

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
        out = np.zeros(n2, dtype=np.float32)
        out[:min(len(y), n2)] = y[:min(len(y), n2)]
        return out

    def info(self):
        d = {"x_shape": self.x_shape}
        if hasattr(self, 'mask'):
            d["mask"] = self.mask
        return d


def make_measurement(category, n=64, m=32):
    rng = np.random.default_rng(123)
    if category == "medical_mri_kspace":
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64)
    elif category == "compressive_mask":
        A = rng.standard_normal((m, n * n)).astype(np.float32) * 0.01
        x = rng.standard_normal(n * n).astype(np.float32)
        return (A @ x).astype(np.float32)
    else:
        return rng.standard_normal((n, n)).astype(np.float32)


def test_gpu_solver(module_name, function_name, category):
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, function_name)
    except Exception as e:
        return "import_error", str(e)[:200], 0

    n, m = 64, 32
    op = PhysicsOperator(category, n=n, m=m)
    y = make_measurement(category, n=n, m=m)
    cfg = {"device": "cuda:0"}

    try:
        torch.cuda.empty_cache()
        t0 = time.time()
        result = fn(y, op, cfg)
        dt = time.time() - t0

        if not isinstance(result, tuple) or len(result) != 2:
            return "bad_return", f"Got {type(result).__name__}", dt

        arr, info = result
        if not isinstance(arr, np.ndarray):
            return "bad_return_type", f"Type: {type(arr).__name__}", dt

        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        return "verified", f"shape={arr.shape}, vram={vram_mb:.0f}MB, time={dt:.2f}s", dt

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "oom", "Out of GPU memory", 0
    except Exception as e:
        torch.cuda.empty_cache()
        return "error", str(e)[:200], 0


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU available. Exiting.")
        return

    results = {}
    stats = {"verified": 0, "error": 0, "import_error": 0, "oom": 0, "bad_return": 0}

    yaml_files = sorted(CONFIGS_DIR.glob("*.yaml"))
    yaml_files = [f for f in yaml_files if not f.name.startswith("_")]

    # Collect all GPU solver entries
    gpu_entries = []
    for yf in yaml_files:
        modality = yf.stem
        with open(yf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "solvers" not in data or not data["solvers"]:
            continue
        category = data.get("category_module", "microscopy_psf")
        for solver_key, solver_cfg in data["solvers"].items():
            if not isinstance(solver_cfg, dict):
                continue
            if solver_cfg.get("gpu", False):
                gpu_entries.append({
                    "modality": modality,
                    "solver_key": solver_key,
                    "module": solver_cfg.get("module", ""),
                    "function": solver_cfg.get("function", ""),
                    "category": category,
                })

    print(f"\nTesting {len(gpu_entries)} GPU solver entries...\n")

    # Deduplicate by (module, function) to avoid redundant tests
    seen = {}
    for entry in gpu_entries:
        key = (entry["module"], entry["function"])
        tag = f"{entry['modality']}/{entry['solver_key']}"

        if key in seen:
            status, msg = seen[key]
            results[tag] = {"status": status, "message": msg, **entry}
            stats[status] = stats.get(status, 0) + 1
            icon = "OK" if status == "verified" else "FAIL"
            print(f"[{icon:4s}] {tag}: (cached) {msg}")
            continue

        status, msg, dt = test_gpu_solver(entry["module"], entry["function"], entry["category"])
        seen[key] = (status, msg)
        stats[status] = stats.get(status, 0) + 1
        results[tag] = {"status": status, "message": msg, "exec_time": round(dt, 3), **entry}

        icon = "OK" if status == "verified" else "FAIL"
        print(f"[{icon:4s}] {tag}: {msg}")

    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"GPU Solvers: {stats.get('verified', 0)}/{total} verified")
    for k, v in stats.items():
        if k != "verified" and v > 0:
            print(f"  {k}: {v}")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "stats": stats,
        "entries": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
