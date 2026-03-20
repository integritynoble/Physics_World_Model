"""Solvers for Brachytherapy Imaging (brachytherapy_img).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "brachytherapy_img"
DISPLAY_NAME = "Brachytherapy Imaging"


# Solver registry for brachytherapy_img
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "best_quality": {
        "name": "DL-Recon [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "brachy_dl": {
        "name": "BrachyNet [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
}



def _fbp_from_sinogram(y: np.ndarray, output_size: int = 256) -> np.ndarray:
    """FBP from sinogram (n_angles, n_det) → image (output_size, output_size)."""
    from skimage.transform import iradon
    n_angles = y.shape[0]
    angles_deg = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    recon = iradon(y.T, theta=angles_deg, circle=False,
                   output_size=output_size, filter_name='ramp')
    return np.clip(recon.astype(np.float32), 0.0, None)


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of ['traditional_cpu', 'best_quality', 'brachy_dl']
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")

    cfg = dict(cfg or {})
    # For sinogram/projection input, apply FBP to get image-domain reconstruction
    if y.ndim == 2 and y.shape[0] != y.shape[1] and operator is None:
        n_angles, n_det = y.shape
        target_size = 256  # standard benchmark output size
        return _fbp_from_sinogram(y.astype(np.float32), output_size=target_size)
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for brachytherapy_img."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DL-Recon [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_brachy_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """BrachyNet [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("brachy_dl", y, operator, cfg)
