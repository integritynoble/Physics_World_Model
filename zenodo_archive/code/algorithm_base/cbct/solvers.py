"""Solvers for Cone-Beam Computed Tomography (CBCT) (cbct).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cbct"
DISPLAY_NAME = "Cone-Beam Computed Tomography (CBCT)"


# Solver registry for cbct
SOLVERS = {
    "traditional_cpu": {
        "name": "FDK / FBP",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "FDK-DL",
        "module": "pwm_core.recon.cbct_solvers",
        "function": "fdk_dl_recon",
        "gpu": True,
        "reference": "Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI",
    },
    "famous_dl": {
        "name": "CBCT-UNet",
        "module": "pwm_core.recon.cbct_solvers",
        "function": "cbct_unet_recon",
        "gpu": True,
        "reference": "Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP",
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
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for cbct."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK / FBP. CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK-DL. GPU required.
    Reference: Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CBCT-UNet. GPU required.
    Reference: Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP
    """
    return run_solver("famous_dl", y, operator, cfg)
