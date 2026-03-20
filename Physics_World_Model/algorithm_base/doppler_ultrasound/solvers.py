"""Solvers for Doppler Ultrasound (doppler_ultrasound).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "doppler_ultrasound"
DISPLAY_NAME = "Doppler Ultrasound"


# Solver registry for doppler_ultrasound
SOLVERS = {
    "traditional_cpu": {
        "name": "Back-Projection (Doppler)",
        "module": "pwm_core.recon.photoacoustic_solver",
        "function": "run_photoacoustic",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "UDoppler-Net [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "famous_dl": {
        "name": "Doppler CFAR [proxy]",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl']
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
    # run_photoacoustic defaults to (128,128). Resize to 256x256.
    if result.ndim == 2 and result.shape != (256, 256):
        from scipy.ndimage import zoom
        z = (256 / result.shape[0], 256 / result.shape[1])
        result = zoom(result, z, order=1).astype(np.float32)
    return result


def list_solvers():
    """List all available solvers for doppler_ultrasound."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Back-Projection (Doppler). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """UDoppler-Net [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Doppler CFAR [proxy]. CPU only.
    Reference: Richardson 1972, JOSA
    """
    return run_solver("famous_dl", y, operator, cfg)
