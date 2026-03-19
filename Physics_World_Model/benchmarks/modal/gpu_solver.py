"""Modal GPU solver wrapper for PWM benchmark.

Deploys GPU-requiring solvers to Modal with T4 GPU.
Used by main server (no local GPU) to run GPU algorithms remotely.

Usage:
    # Deploy and run a GPU solver
    modal run benchmarks/modal/gpu_solver.py --modality cassi --solver best_quality

    # From Python (server integration)
    from benchmarks.modal.gpu_solver import run_gpu_solver
    result = run_gpu_solver.remote(modality_id="cassi", solver_key="best_quality",
                                   y_data=y.tobytes(), y_shape=list(y.shape), y_dtype="float32")
"""

from __future__ import annotations

import modal
import numpy as np
import json
import sys
from pathlib import Path

# Modal app definition
app = modal.App("pwm-gpu-solver")

# Image with all dependencies
pwm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "scipy", "h5py", "pyyaml", "scikit-image",
        "torch", "torchvision",  # for DL solvers
    )
    .copy_local_dir(
        local_path="packages/pwm_core",
        remote_path="/root/pwm_core_pkg",
    )
    .copy_local_dir(
        local_path="benchmarks",
        remote_path="/root/benchmarks",
    )
    .run_commands("cd /root/pwm_core_pkg && pip install -e .")
)


@app.function(
    image=pwm_image,
    gpu="T4",
    timeout=300,
    retries=1,
)
def run_gpu_solver(
    modality_id: str,
    solver_key: str,
    y_data: bytes,
    y_shape: list,
    y_dtype: str = "float32",
    operator_data: bytes | None = None,
    operator_shape: list | None = None,
    params: dict | None = None,
) -> dict:
    """Run a GPU solver on Modal T4.

    Args:
        modality_id: Modality identifier (e.g., "cassi")
        solver_key: Solver key from YAML config (e.g., "best_quality")
        y_data: Measurement data as bytes (numpy tobytes())
        y_shape: Shape of measurement array
        y_dtype: Data type string (e.g., "float32")
        operator_data: Optional forward operator matrix as bytes
        operator_shape: Shape of operator matrix
        params: Optional solver parameters override

    Returns:
        dict with "x_hat" (base64-encoded numpy), "shape", "dtype", "psnr", "time"
    """
    import importlib
    import time
    import yaml
    import base64

    sys.path.insert(0, "/root/pwm_core_pkg")
    sys.path.insert(0, "/root")

    # Reconstruct numpy arrays
    y = np.frombuffer(y_data, dtype=y_dtype).reshape(y_shape).copy()

    # Load solver spec from YAML
    config_path = Path(f"/root/benchmarks/configs/{modality_id}.yaml")
    if not config_path.exists():
        return {"error": f"Config not found: {modality_id}"}

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    solvers = cfg.get("solvers", {})
    if solver_key not in solvers:
        return {"error": f"Solver {solver_key} not found in {modality_id}"}

    spec = solvers[solver_key]
    module_name = spec.get("module", "")
    func_name = spec.get("function", "")
    solver_params = params or spec.get("params", {})
    if isinstance(solver_params, str):
        solver_params = {}

    # Build operator object
    class SimpleOperator:
        def __init__(self, A=None):
            self.A = A
            self.shape = y_shape

        def forward(self, x):
            if self.A is not None:
                return self.A @ x.ravel()
            return x

        def adjoint(self, y):
            if self.A is not None:
                return self.A.T @ y.ravel()
            return y

    operator = SimpleOperator()
    if operator_data is not None and operator_shape is not None:
        A = np.frombuffer(operator_data, dtype="float32").reshape(operator_shape).copy()
        operator = SimpleOperator(A)

    # Import and run solver
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
        result = fn(y.astype(np.float32), operator, solver_params)
        if isinstance(result, tuple):
            x_hat = np.asarray(result[0], dtype=np.float32)
        else:
            x_hat = np.asarray(result, dtype=np.float32)
    except Exception as e:
        return {"error": str(e), "module": module_name, "function": func_name}

    elapsed = time.time() - t0

    # Encode result
    x_bytes = base64.b64encode(x_hat.tobytes()).decode("ascii")

    return {
        "x_hat": x_bytes,
        "shape": list(x_hat.shape),
        "dtype": str(x_hat.dtype),
        "time": round(elapsed, 3),
    }


@app.function(image=pwm_image, gpu="T4", timeout=60)
def check_gpu():
    """Verify GPU is available on Modal."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }


@app.local_entrypoint()
def main():
    """Test the GPU solver from command line."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args()

    if args.check_gpu:
        result = check_gpu.remote()
        print(json.dumps(result, indent=2))
        return

    # Create dummy data for testing
    y = np.random.randn(64, 64).astype(np.float32)
    result = run_gpu_solver.remote(
        modality_id=args.modality,
        solver_key=args.solver,
        y_data=y.tobytes(),
        y_shape=list(y.shape),
        y_dtype="float32",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "x_hat"}, indent=2))
