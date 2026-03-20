"""Solvers for 3D Gaussian Splatting (3DGS) (gaussian_splatting).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "gaussian_splatting"
DISPLAY_NAME = "3D Gaussian Splatting (3DGS)"


# Solver registry for gaussian_splatting
SOLVERS = {
    "traditional_cpu": {
        "name": "EWA Splatting",
        "module": "pwm_core.recon.gaussian_splatting_solver",
        "function": "run_gaussian_splatting",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "3DGS (full)",
        "module": "pwm_core.recon.gaussian_splatting_solver",
        "function": "run_gaussian_splatting",
        "gpu": True,
        "reference": "Kerbl et al. SIGGRAPH 2023",
    },
    "famous_dl": {
        "name": "NeRF (baseline comparison)",
        "module": "pwm_core.recon.nerf_solver",
        "function": "run_nerf",
        "gpu": False,
        "reference": "",
    },
    "small_gpu": {
        "name": "3DGS (compact)",
        "module": "pwm_core.recon.gaussian_splatting_solver",
        "function": "run_gaussian_splatting",
        "gpu": False,
        "reference": "",
    },
}


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu']
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        result = np.asarray(result[0], dtype=np.float32)
    else:
        result = np.asarray(result, dtype=np.float32)
    # run_gaussian_splatting returns (H, W, 3) RGB. Convert to grayscale.
    if result.ndim == 3 and result.shape[-1] == 3:
        result = (0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for gaussian_splatting."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EWA Splatting. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """3DGS (full). GPU required.
    Reference: Kerbl et al. SIGGRAPH 2023
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """NeRF (baseline comparison). CPU only.
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """3DGS (compact). CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)
