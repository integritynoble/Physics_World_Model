"""Solvers for Proton Radiography (proton_radiography).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "proton_radiography"
DISPLAY_NAME = "Proton Radiography"


# Solver registry for proton_radiography
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP (proton radiography)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "ProtonRecon-Net [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "famous_dl": {
        "name": "FBP-Proton [proxy]",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl']
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")

    cfg = dict(cfg or {})
    # For sinogram input (n_angles, n_det), set output_size=256 so FBP returns correct shape
    if y.ndim == 2 and operator is None and "output_size" not in cfg:
        cfg["output_size"] = 256
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for proton_radiography."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP (proton radiography). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ProtonRecon-Net [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP-Proton [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("famous_dl", y, operator, cfg)
