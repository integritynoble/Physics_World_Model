"""Solvers for Diffuse Optical Tomography (DOT) (dot).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "dot"
DISPLAY_NAME = "Diffuse Optical Tomography (DOT)"


# Solver registry for dot
SOLVERS = {
    "traditional_cpu": {
        "name": "Born Approximation",
        "module": "pwm_core.recon.dot_solver",
        "function": "run_dot",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "L-BFGS-TV [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "dot_dl": {
        "name": "DOT-Net [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'dot_dl']
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
    # run_dot returns a 3D volume (nz, ny, nx). Take max-intensity projection along z.
    if result.ndim == 3:
        result = result.max(axis=0)
    # Resize to 256x256 if result is not the right shape
    if result.ndim == 2 and result.shape != (256, 256):
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    # Handle 1D output (fallback from DOT when no operator provided)
    if result.ndim == 1:
        n = int(result.shape[0] ** 0.5) or 1
        result = result[:n*n].reshape(n, n)
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for dot."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Born Approximation. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """L-BFGS-TV [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_dot_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DOT-Net [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("dot_dl", y, operator, cfg)
