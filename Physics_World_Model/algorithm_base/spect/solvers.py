"""Solvers for Single Photon Emission CT (SPECT) (spect).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "spect"
DISPLAY_NAME = "Single Photon Emission CT (SPECT)"


# Solver registry for spect
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP (emission tomography)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "SPECT-DL (OSEM+)",
        "module": "pwm_core.recon.spect_solvers",
        "function": "spect_dl_recon",
        "gpu": True,
        "reference": "Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging",
    },
    "famous_dl": {
        "name": "SPECT-UNet",
        "module": "pwm_core.recon.spect_solvers",
        "function": "spect_unet_recon",
        "gpu": True,
        "reference": "Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6)",
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
    """List all available solvers for spect."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP (emission tomography). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SPECT-DL (OSEM+). GPU required.
    Reference: Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SPECT-UNet. GPU required.
    Reference: Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6)
    """
    return run_solver("famous_dl", y, operator, cfg)
