"""Solvers for Magnetic Resonance Imaging (MRI) (mri).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat

When no operator is provided, an FFT-based MRI operator is created automatically.
k-space data in (H, W, 2) real+imag format is converted to complex automatically.
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "mri"
DISPLAY_NAME = "Magnetic Resonance Imaging (MRI)"


# Solver registry for mri
SOLVERS = {
    "traditional_cpu": {
        "name": "Zero-Filled IFFT",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_zero_filled",
        "gpu": False,
        "reference": "",
    },
    "best_quality": {
        "name": "CS-MRI (Wavelet)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_cs_mri",
        "gpu": False,
        "reference": "Lustig et al. 2007, MRM",
    },
    "famous_dl": {
        "name": "MoDL",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": False,
        "reference": "Aggarwal et al. 2019, IEEE TMI",
    },
    "small_gpu": {
        "name": "MoDL (5 unrolls)",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": False,
        "reference": "",
    },
    "sense": {
        "name": "SENSE",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_sense",
        "gpu": False,
        "reference": "Pruessmann et al., MRM 1999",
    },
}


class MRIOperator:
    """Lightweight MRI forward/adjoint operator for solvers that need one.

    Single-coil Cartesian MRI with undersampling mask.
    """
    def __init__(self, mask: np.ndarray, image_size: int = 256):
        self.mask = mask.astype(np.float32)
        self.image_size = image_size
        self.x_shape = (image_size, image_size)

    def forward(self, x):
        x_2d = x.reshape(self.image_size, self.image_size)
        kspace = np.fft.fftshift(np.fft.fft2(x_2d))
        return (kspace * self.mask).astype(np.complex64)

    def adjoint(self, y):
        return np.fft.ifft2(np.fft.ifftshift(y)).astype(np.complex64)

    def info(self):
        return {
            'modality': 'mri',
            'mask': self.mask,
            'x_shape': self.x_shape,
        }


def _to_complex(y: np.ndarray) -> np.ndarray:
    """Convert real+imag format to complex if needed."""
    if y.ndim >= 2 and y.shape[-1] == 2 and not np.iscomplexobj(y):
        return (y[..., 0] + 1j * y[..., 1]).astype(np.complex64)
    return y


def _infer_mask(kspace: np.ndarray) -> np.ndarray:
    """Infer sampling mask from k-space (sampled locations have nonzero values)."""
    return (np.abs(kspace) > 1e-10).astype(np.float32)


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of the registered solver keys
        y: k-space data. Accepts (H, W, 2) real+imag or (H, W) complex
        operator: MRI operator (auto-created if None)
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed image (magnitude)
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})

    # Convert real+imag to complex
    y_complex = _to_complex(y)

    # Auto-create MRI operator when none provided
    if operator is None and y_complex.ndim == 2:
        mask = _infer_mask(y_complex)
        operator = MRIOperator(mask, y_complex.shape[0])

    spec = SOLVERS[solver_key]
    needs_image_input = spec["module"] in ("pwm_core.recon.modl",)

    if needs_image_input:
        # MoDL needs complex k-space but also the operator
        # Pre-reconstruct with zero-filled for initialization
        zf_mod = importlib.import_module("pwm_core.recon.mri_solvers")
        zf_result = zf_mod.run_zero_filled(y_complex.astype(np.complex64 if np.iscomplexobj(y_complex) else np.float32), operator, cfg)
        if isinstance(zf_result, tuple):
            y_img = zf_result[0]
        else:
            y_img = zf_result
        fn = _load_fn(solver_key)
        result = fn(y_img.astype(np.float32), operator, cfg)
    else:
        fn = _load_fn(solver_key)
        result = fn(y_complex.astype(np.complex64 if np.iscomplexobj(y_complex) else np.float32), operator, cfg)

    if isinstance(result, tuple):
        x_hat = np.asarray(result[0], dtype=np.float32)
    else:
        x_hat = np.asarray(result, dtype=np.float32)

    # Return magnitude for complex results
    if np.iscomplexobj(x_hat):
        x_hat = np.abs(x_hat).astype(np.float32)

    return x_hat


def list_solvers():
    """List all available solvers for mri."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Zero-Filled IFFT. CPU only."""
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CS-MRI (Wavelet). CPU only.
    Reference: Lustig et al. 2007, MRM
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MoDL. CPU only.
    Reference: Aggarwal et al. 2019, IEEE TMI
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MoDL (5 unrolls). CPU only."""
    return run_solver("small_gpu", y, operator, cfg)

def run_sense(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """SENSE. CPU only.
    Reference: Pruessmann et al., MRM 1999
    """
    return run_solver("sense", y, operator, cfg)
