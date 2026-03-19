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
        "name": "FBP + NLM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_nlm",
        "gpu": False,
        "reference": "Buades et al. 2005 + PnP framework",
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
    def __init__(self, n_angles: int, n_detectors: int, output_size: int = 362):
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.output_size = output_size
        self.angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        self.angles = np.deg2rad(self.angles_deg)
        self.x_shape = (output_size, output_size)

    def forward(self, x):
        from skimage.transform import radon
        x_2d = x.reshape(self.output_size, self.output_size)
        sino = radon(x_2d, theta=self.angles_deg, circle=False)
        return sino.T.astype(np.float32)  # (n_angles, n_det)

    def adjoint(self, y):
        from skimage.transform import iradon
        y_2d = y.reshape(self.n_angles, self.n_detectors)
        recon = iradon(y_2d.T, theta=self.angles_deg, circle=False,
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

    # Apply solver-specific config overrides
    spec = SOLVERS[solver_key]
    if "cfg_override" in spec:
        for k, v in spec["cfg_override"].items():
            if k not in cfg:
                cfg[k] = v

    # Auto-create CT operator when none provided
    if operator is None and y.ndim == 2:
        operator = _make_ct_operator(y, cfg)

    is_sinogram = y.ndim == 2 and y.shape[0] != y.shape[1]

    # For self-referencing solvers (e.g. run_fbp_nlm), call directly
    if spec["module"] == "algorithm_base.ct.solvers":
        fn = globals()[spec["function"]]
        result = fn(y.astype(np.float32), operator, cfg)
    elif spec["module"] == "pwm_core.recon.redcnn" and is_sinogram:
        # RED-CNN needs image input — do FBP first
        fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
        fbp_result = fbp_mod.run_fbp(y.astype(np.float32), operator, cfg)
        fbp_img = fbp_result[0] if isinstance(fbp_result, tuple) else fbp_result
        fn = _load_fn(solver_key)
        result = fn(fbp_img.astype(np.float32), operator, cfg)
    else:
        fn = _load_fn(solver_key)
        result = fn(y.astype(np.float32), operator, cfg)

    if isinstance(result, tuple):
        x_hat = np.asarray(result[0], dtype=np.float32)
    else:
        x_hat = np.asarray(result, dtype=np.float32)

    # Safety: if solver returned wrong shape, fall back to FBP
    expected_shape = operator.x_shape if operator else None
    if expected_shape and x_hat.shape != expected_shape:
        fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
        fbp_result = fbp_mod.run_fbp(y.astype(np.float32), operator, cfg)
        x_hat = np.asarray(
            fbp_result[0] if isinstance(fbp_result, tuple) else fbp_result,
            dtype=np.float32)

    return x_hat


def run_fbp_nlm(y: np.ndarray, physics: Any, cfg: Dict[str, Any] = None) -> tuple:
    """FBP + Non-Local Means denoising for CT.

    Steps: 1) FBP reconstruction, 2) NLM denoising for artifact reduction.
    """
    from skimage.restoration import denoise_nl_means
    cfg = cfg or {}

    # Step 1: FBP reconstruction
    if physics is None:
        physics = _make_ct_operator(y, cfg)

    fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
    fbp_result = fbp_mod.run_fbp(y.astype(np.float32), physics, cfg)
    if isinstance(fbp_result, tuple):
        img = fbp_result[0]
    else:
        img = fbp_result

    # Step 2: NLM denoising
    img_norm = img.astype(np.float64)
    img_min, img_max = img_norm.min(), img_norm.max()
    if img_max - img_min > 1e-8:
        img_norm = (img_norm - img_min) / (img_max - img_min)

    sigma_est = cfg.get("sigma", 0.02)
    denoised = denoise_nl_means(
        img_norm, patch_size=5, patch_distance=6,
        h=0.8 * sigma_est, fast_mode=True, sigma=sigma_est,
    )

    if img_max - img_min > 1e-8:
        denoised = denoised * (img_max - img_min) + img_min

    return denoised.astype(np.float32), {"solver": "fbp_nlm"}


def list_solvers():
    """List all available solvers for ct."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP. CPU only."""
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """FBP + NLM. CPU only."""
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RED-CNN. CPU only.
    Reference: Chen et al. 2017, IEEE TMI
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RED-CNN. CPU only."""
    return run_solver("small_gpu", y, operator, cfg)
