"""Solvers for Cryo-Electron Tomography (Cryo-ET) (cryo_et).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cryo_et"
DISPLAY_NAME = "Cryo-Electron Tomography (Cryo-ET)"


# Solver registry for cryo_et
SOLVERS = {
    "traditional_cpu": {
        "name": "Richardson-Lucy",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "CARE",
        "module": "pwm_core.recon.care_unet",
        "function": "run_care",
        "gpu": True,
        "reference": "Weigert et al. 2018",
    },
    "cryo_et_dl": {
        "name": "CryoCARE",
        "module": "pwm_core.recon.cryoet_solvers",
        "function": "cryocare_recon",
        "gpu": True,
        "reference": "Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol.",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'cryo_et_dl']
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
    """List all available solvers for cryo_et."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CARE. GPU required.
    Reference: Weigert et al. 2018
    """
    return run_solver("best_quality", y, operator, cfg)

def run_cryo_et_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CryoCARE. GPU required.
    Reference: Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol.
    """
    return run_solver("cryo_et_dl", y, operator, cfg)
