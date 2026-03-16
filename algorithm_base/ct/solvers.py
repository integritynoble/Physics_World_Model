"""Solvers for X-ray Computed Tomography (CT) (ct).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat

When no operator is provided, a Radon/FBP operator is created automatically
from the sinogram dimensions.
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "ct"
DISPLAY_NAME = "X-ray Computed Tomography (CT)"


# Solver registry for ct
SOLVERS = {
    "traditional_cpu": {
        "name": "FBP",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "PnP-HQS + NLM",
        "module": "pwm_core.recon.pnp",
        "function": "run_pnp",
        "gpu": False,
        "reference": "",
    },
    "famous_dl": {
        "name": "RED-CNN",
        "module": "pwm_core.recon.redcnn",
        "function": "run_redcnn",
        "gpu": False,
        "reference": "Chen et al. 2017, IEEE TMI",
    },
    "small_gpu": {
        "name": "RED-CNN",
        "module": "pwm_core.recon.redcnn",
        "function": "run_redcnn",
        "gpu": False,
        "reference": "",
    },
}


class CTOperator:
    """Lightweight CT forward/adjoint operator for solvers that need one.

    Uses skimage radon/iradon with parallel-beam geometry.
    """
    def __init__(self, n_angles: int, n_detectors: int, output_size: int = 256):
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.output_size = output_size
        self.angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        self.angles = np.deg2rad(self.angles_deg)
        self.x_shape = (output_size, output_size)

    def forward(self, x):
        from skimage.transform import radon
        x_2d = x.reshape(self.output_size, self.output_size)
        sino = radon(x_2d, theta=self.angles_deg, circle=True)
        return sino.T.astype(np.float32)  # (n_angles, n_det)

    def adjoint(self, y):
        from skimage.transform import iradon
        y_2d = y.reshape(self.n_angles, self.n_detectors)
        recon = iradon(y_2d.T, theta=self.angles_deg, circle=True,
                       output_size=self.output_size, filter_name=None)
        return recon.astype(np.float32)

    def info(self):
        return {
            'modality': 'ct',
            'angles': self.angles,
            'n_angles': self.n_angles,
            'x_shape': self.x_shape,
        }


def _make_ct_operator(y: np.ndarray, cfg: Dict) -> CTOperator:
    """Create a CT operator from sinogram shape."""
    n_angles, n_det = y.shape[:2]
    output_size = cfg.get('output_size', n_det)
    return CTOperator(n_angles, n_det, output_size)


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of ['traditional_cpu', 'best_quality', 'famous_dl', 'small_gpu']
        y: Sinogram (n_angles, n_detectors) float32
        operator: Forward operator (auto-created if None)
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed image
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})

    # Auto-create CT operator when none provided
    if operator is None and y.ndim == 2:
        operator = _make_ct_operator(y, cfg)

    # For image-domain solvers (PnP, RED-CNN), pre-reconstruct with FBP
    spec = SOLVERS[solver_key]
    needs_image_input = spec["module"] in ("pwm_core.recon.pnp", "pwm_core.recon.redcnn")

    if needs_image_input and y.ndim == 2 and y.shape[0] != y.shape[1]:
        # y is sinogram, not image — run FBP first
        fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
        fbp_result = fbp_mod.run_fbp(y.astype(np.float32), operator, cfg)
        if isinstance(fbp_result, tuple):
            y_img = fbp_result[0]
        else:
            y_img = fbp_result
        # For PnP: pass FBP result as y with the CT operator for refinement
        if spec["module"] == "pwm_core.recon.pnp":
            fn = _load_fn(solver_key)
            result = fn(y.astype(np.float32), operator, cfg)
        else:
            # For RED-CNN: pass FBP result as image input
            fn = _load_fn(solver_key)
            result = fn(y_img.astype(np.float32), operator, cfg)
    else:
        fn = _load_fn(solver_key)
        result = fn(y.astype(np.float32), operator, cfg)

    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for ct."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP. CPU only."""
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PnP-HQS + NLM. CPU only."""
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RED-CNN. CPU only.
    Reference: Chen et al. 2017, IEEE TMI
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RED-CNN. CPU only."""
    return run_solver("small_gpu", y, operator, cfg)
