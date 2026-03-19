"""Solvers for Coded Aperture Compressive Temporal Imaging (CACTI) (cacti).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cacti"
DISPLAY_NAME = "Coded Aperture Compressive Temporal Imaging (CACTI)"


# Solver registry for cacti
SOLVERS = {
    "traditional_cpu": {
        "name": "GAP-TV",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016",
    },
    "best_quality": {
        "name": "EfficientSCI",
        "module": "pwm_core.recon.efficientsci",
        "function": "run_efficientsci",
        "gpu": False,
        "reference": "Wang et al. CVPR 2023",
    },
    "famous_dl": {
        "name": "ELP-Unfolding",
        "module": "pwm_core.recon.elp_unfolding",
        "function": "run_elp_unfolding",
        "gpu": False,
        "reference": "Yang et al. ECCV 2022",
    },
    "small_gpu": {
        "name": "EfficientSCI-T",
        "module": "pwm_core.recon.efficientsci",
        "function": "run_efficientsci",
        "gpu": False,
        "reference": "",
    },
    "pnp_ffdnet": {
        "name": "PnP-FFDNet",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "pnp_ffdnet_cacti",
        "gpu": False,
        "reference": "Yuan et al., CVPR 2020",
    },
    "hisvit9": {
        "name": "HiSViT-9",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "hisvit_cacti",
        "gpu": True,
        "reference": "Chen et al., ICCV 2023",
    },
    "hisvit13": {
        "name": "HiSViT-13",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "hisvit_cacti",
        "gpu": True,
        "reference": "Chen et al., ECCV 2024",
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
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu', 'pnp_ffdnet', 'hisvit9', 'hisvit13']
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
    """List all available solvers for cacti."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV. CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EfficientSCI. CPU only.
    Reference: Wang et al. CVPR 2023
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ELP-Unfolding. CPU only.
    Reference: Yang et al. ECCV 2022
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EfficientSCI-T. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_pnp_ffdnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PnP-FFDNet. CPU only.
    Reference: Yuan et al., CVPR 2020
    """
    return run_solver("pnp_ffdnet", y, operator, cfg)

def run_hisvit9(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HiSViT-9. GPU required.
    Reference: Chen et al., ICCV 2023
    """
    return run_solver("hisvit9", y, operator, cfg)

def run_hisvit13(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HiSViT-13. GPU required.
    Reference: Chen et al., ECCV 2024
    """
    return run_solver("hisvit13", y, operator, cfg)
