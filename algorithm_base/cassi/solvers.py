"""Solvers for Coded Aperture Snapshot Spectral Imaging (CASSI) (cassi).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat

When operator=None, loads mask/n_bands from H5 metadata or creates defaults.
Standard data format: y_ideal (H, W_meas), mask (H, W), wavelength (n_bands,)
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cassi"
DISPLAY_NAME = "Coded Aperture Snapshot Spectral Imaging (CASSI)"


# Solver registry for cassi
# Reference PSNR from InverseNet ECCV benchmark (KAIST 10 scenes, Scenario I):
#   GAP-TV: 24.34 dB (100 iters, Chambolle TV w=0.1)
#   PnP-HSICNN: 25.12 dB (GAP + HSI-SDeCNN hybrid)
#   HDNet: 34.66 dB (pretrained, mask-oblivious)
#   MST-L: 34.81 dB (pretrained, mask-aware transformer)
SOLVERS = {
    "traditional_cpu": {
        "name": "GAP-TV",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016 — 24.34 dB on KAIST",
        "cfg_override": {"iters": 100, "lam": 0.1, "tv_iter": 5},
    },
    "best_quality": {
        "name": "GAP-TV (200 iter)",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016 — ~24.9 dB on KAIST",
        "cfg_override": {"iters": 200, "lam": 0.01, "tv_iter": 5},
    },
    "famous_dl": {
        "name": "MST-L",
        "module": "pwm_core.recon.mst",
        "function": "mst_recon_cassi",
        "gpu": True,
        "reference": "Cai et al., CVPR 2022 — 34.81 dB on KAIST",
    },
    "small_gpu": {
        "name": "GAP-TV (fast)",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016",
        "cfg_override": {"iters": 50, "lam": 0.1, "tv_iter": 5},
    },
    "mst_l": {
        "name": "MST-L",
        "module": "pwm_core.recon.mst",
        "function": "mst_recon_cassi",
        "gpu": True,
        "reference": "Cai et al., CVPR 2022 — 34.81 dB on KAIST",
    },
    "hdnet": {
        "name": "HDNet",
        "module": "pwm_core.recon.hdnet",
        "function": "run_hdnet",
        "gpu": True,
        "reference": "Hu et al., CVPR 2022 — 34.66 dB on KAIST",
    },
    "hsi_sdecnn": {
        "name": "PnP-HSICNN",
        "module": "pwm_core.recon.hsi_sdecnn",
        "function": "run_hsi_sdecnn",
        "gpu": True,
        "reference": "Maffei et al., TGRS 2020 — 25.12 dB on KAIST",
    },
}


class CASSIOperator:
    """CASSI forward/adjoint operator for solvers."""
    def __init__(self, mask: np.ndarray, n_bands: int, step: int = 2):
        self.mask = mask.astype(np.float32)
        self.n_bands = n_bands
        self.step = step
        h, w = mask.shape
        self.h = h
        self.w = w
        self.w_meas = w + (n_bands - 1) * step
        self.x_shape = (h, w, n_bands)

    def forward(self, x):
        x_3d = x.reshape(self.h, self.w, self.n_bands)
        y = np.zeros((self.h, self.w_meas), dtype=np.float32)
        for k in range(self.n_bands):
            y[:, k*self.step : k*self.step + self.w] += self.mask * x_3d[:, :, k]
        return y

    def adjoint(self, y):
        y_2d = y.reshape(self.h, self.w_meas)
        x = np.zeros((self.h, self.w, self.n_bands), dtype=np.float32)
        for k in range(self.n_bands):
            x[:, :, k] = self.mask * y_2d[:, k*self.step : k*self.step + self.w]
        return x

    def info(self):
        return {
            'modality': 'cassi',
            'mask': self.mask,
            'n_bands': self.n_bands,
            'step': self.step,
            'x_shape': self.x_shape,
        }


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def _make_operator_from_h5(y: np.ndarray, cfg: Dict) -> Optional[CASSIOperator]:
    """Try to load mask and params from the standard H5 file."""
    mask = cfg.get('mask', None)
    n_bands = cfg.get('n_bands', None)
    step = cfg.get('step', 2)

    if mask is None or n_bands is None:
        return None
    return CASSIOperator(mask, n_bands, step)


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: Solver key from SOLVERS dict
        y: CASSI measurement (H, W_meas) float32
        operator: CASSI operator (auto-created if None, needs mask/n_bands in cfg)
        cfg: Hyperparameters. For auto-operator: must include 'mask', 'n_bands', 'step'

    Returns:
        x_hat: Reconstructed spectral cube (H, W, n_bands)
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})

    # Apply solver-specific config overrides
    spec = SOLVERS[solver_key]
    if "cfg_override" in spec:
        for k, v in spec["cfg_override"].items():
            if k not in cfg:
                cfg[k] = v

    # Auto-create CASSI operator when none provided
    if operator is None:
        operator = _make_operator_from_h5(y, cfg)

    fn = _load_fn(solver_key)

    # Special dispatch for solvers with non-standard interfaces
    if spec["function"] == "mst_recon_cassi":
        # MST takes (measurement, mask_2d, nC, step, ...)
        mask = operator.mask if operator else cfg.get('mask')
        n_bands = operator.n_bands if operator else cfg.get('n_bands', 28)
        step = operator.step if operator else cfg.get('step', 2)
        if mask is None:
            raise ValueError("MST requires mask (via operator or cfg['mask'])")
        result = fn(y.astype(np.float32), mask, nC=n_bands, step=step,
                     device=cfg.get('device'), variant=cfg.get('variant', 'mst_l'))
    elif spec["function"] == "run_hsi_sdecnn":
        # HSI-SDeCNN: pass through standard interface
        result = fn(y.astype(np.float32), operator, cfg)
    else:
        # Standard (y, physics, cfg) interface
        result = fn(y.astype(np.float32), operator, cfg)

    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for cassi."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV. CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (guided). CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (fast). CPU only."""
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV (small). CPU only."""
    return run_solver("small_gpu", y, operator, cfg)

def run_mst_l(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MST-L. CPU only.
    Reference: Cai et al., CVPR 2022
    """
    return run_solver("mst_l", y, operator, cfg)

def run_hdnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HDNet. CPU only.
    Reference: Hu et al., CVPR 2022
    """
    return run_solver("hdnet", y, operator, cfg)

def run_hsi_sdecnn(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HSI-SDeCNN. CPU only.
    Reference: Maffei et al., TGRS 2020
    """
    return run_solver("hsi_sdecnn", y, operator, cfg)
