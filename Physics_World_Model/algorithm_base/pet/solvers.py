"""Solvers for Positron Emission Tomography (PET) (pet).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "pet"
DISPLAY_NAME = "Positron Emission Tomography (PET)"


# Solver registry for pet
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP (emission tomography)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "NeuroLF-PET",
        "module": "pwm_core.recon.pet_solvers",
        "function": "neurolF_pet_recon",
        "gpu": True,
        "reference": "Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58",
    },
    "famous_dl": {
        "name": "PET-DL (U-Net)",
        "module": "pwm_core.recon.pet_solvers",
        "function": "pet_unet_recon",
        "gpu": True,
        "reference": "Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9)",
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
    cfg = dict(cfg or {})
    # For sinogram input (n_angles, n_det), set output_size=256 so FBP returns (256,256)
    if y.ndim == 2 and operator is None and "output_size" not in cfg:
        cfg["output_size"] = 256
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for pet."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP (emission tomography). CPU only.
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """NeuroLF-PET. GPU required.
    Reference: Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PET-DL (U-Net). GPU required.
    Reference: Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9)
    """
    return run_solver("famous_dl", y, operator, cfg)
