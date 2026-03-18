"""Solvers for Magnetic Resonance Imaging (MRI) (mri).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat

When no operator is provided, an FFT-based MRI operator is created automatically.
k-space data in (H, W, 2) real+imag format is converted to complex automatically.
"""

from __future__ import annotations
import sys, os
# Ensure PWM5 packages are imported, not PWM4
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'packages', 'pwm_core'))

import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "mri"
DISPLAY_NAME = "Magnetic Resonance Imaging (MRI)"


# Solver registry for mri
SOLVERS = {
    # ====== Classical / CPU solvers ======
    "traditional_cpu": {
        "name": "Zero-Filled IFFT",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_zero_filled",
        "gpu": False,
        "reference": "Lauterbur, Nature 1973",
    },
    "best_quality": {
        "name": "CS-MRI (Wavelet)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_cs_mri",
        "gpu": False,
        "reference": "Lustig et al., MRM 2007 — 33.0 dB on fastMRI knee 4x",
    },
    "sense": {
        "name": "SENSE",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_sense",
        "gpu": False,
        "reference": "Pruessmann et al., MRM 1999",
    },
    "espirit": {
        "name": "ESPIRiT",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_espirit_recon",
        "gpu": False,
        "reference": "Uecker et al., MRM 2014 — 34.2 dB on fastMRI knee 4x",
    },
    "cs_tv": {
        "name": "CS-MRI (TV)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_tv_mri",
        "gpu": False,
        "reference": "Block et al., MRM 2007",
    },
    "pocs": {
        "name": "POCS",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_pocs",
        "gpu": False,
        "reference": "Haacke et al., MRM 1991",
    },
    "admm_mri": {
        "name": "ADMM",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_admm_mri",
        "gpu": False,
        "reference": "Yang et al., MRM 2010",
    },
    "conjugate_gradient": {
        "name": "Conjugate Gradient",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_conjugate_gradient",
        "gpu": False,
        "reference": "Pruessmann et al., MRM 2001",
    },
    "truncated_ifft": {
        "name": "Truncated IFFT",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_truncated_ifft",
        "gpu": False,
        "reference": "Classic Fourier MRI, Lauterbur 1973",
    },
    "gradient_descent": {
        "name": "Gradient Descent",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_gradient_descent",
        "gpu": False,
        "reference": "Fessler, IEEE SPM 2010",
    },
    "split_bregman": {
        "name": "Split Bregman",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_split_bregman",
        "gpu": False,
        "reference": "Goldstein & Osher, SIAM J Imaging Sci 2009",
    },
    "pnp_admm": {
        "name": "PnP-ADMM",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_pnp_mri",
        "gpu": False,
        "reference": "Ahmad et al., IEEE SPM 2020",
    },
    "low_rank": {
        "name": "Low-Rank",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_low_rank",
        "gpu": False,
        "reference": "Haldar, IEEE TMI 2014 — LORAKS",
    },
    "ista_mri": {
        "name": "ISTA",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_ista_mri",
        "gpu": False,
        "reference": "Daubechies et al., 2004; Beck & Teboulle, 2009",
    },
    "grappa_like": {
        "name": "GRAPPA-like",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_grappa_like",
        "gpu": False,
        "reference": "Griswold et al., MRM 2002",
    },
    # ====== Deep Learning solvers ======
    "famous_dl": {
        "name": "MoDL",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": True,
        "reference": "Aggarwal et al., IEEE TMI 2019 — 36.0 dB on fastMRI knee 4x",
    },
    "small_gpu": {
        "name": "MoDL (5 unrolls)",
        "module": "pwm_core.recon.modl",
        "function": "run_modl",
        "gpu": True,
        "reference": "Aggarwal et al., IEEE TMI 2019",
        "cfg_override": {"n_iter": 5},
    },
    "varnet": {
        "name": "E2E-VarNet",
        "module": "pwm_core.recon.varnet",
        "function": "run_varnet",
        "gpu": True,
        "reference": "Sriram et al., MICCAI 2020 — 40.5 dB on fastMRI knee 4x",
    },
    # ====== Additional classical solvers (1951–2017) ======
    "fista_mri": {
        "name": "FISTA",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_fista_mri",
        "gpu": False,
        "reference": "Beck & Teboulle, SIAM J Imaging Sci 2009",
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber, Amer J Math 1951",
    },
    "tikhonov": {
        "name": "Tikhonov Regularization",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov, Soviet Math Dokl 1963",
    },
    "homodyne": {
        "name": "Homodyne Detection",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_homodyne",
        "gpu": False,
        "reference": "Noll, Nishimura, Macovski, IEEE TMI 1991",
    },
    "nuclear_norm": {
        "name": "Nuclear Norm (SVT)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_nuclear_norm",
        "gpu": False,
        "reference": "Cai, Candes, Shen, SIAM J Optim 2010; Shin et al., MRM 2014 (SAKE)",
    },
    "proximal_gradient": {
        "name": "Proximal Gradient Descent",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_proximal_gradient",
        "gpu": False,
        "reference": "Combettes & Wajs, Multiscale Model Simul 2005",
    },
    "bm3d_mri": {
        "name": "BM3D-MRI",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_bm3d_mri",
        "gpu": False,
        "reference": "Eksioglu, IEEE SPL 2016",
    },
    "spirit_like": {
        "name": "SPIRiT-like",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_spirit_like",
        "gpu": False,
        "reference": "Lustig & Pauly, MRM 2010",
    },
    "red_mri": {
        "name": "RED (Regularization by Denoising)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_red_mri",
        "gpu": False,
        "reference": "Romano, Elad, Milanfar, SIAM J Imaging Sci 2017",
    },
    "dictionary_learning": {
        "name": "Dictionary Learning MRI",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_dictionary_learning",
        "gpu": False,
        "reference": "Ravishankar & Bresler, IEEE TMI 2011",
    },
    "aloha": {
        "name": "ALOHA (Hankel Low-Rank)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_aloha",
        "gpu": False,
        "reference": "Jin & Ye, IEEE TIP 2015",
    },
    # ====== Additional DL solvers (random init) ======
    "unet_mri": {
        "name": "U-Net (fastMRI)",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_unet_mri",
        "gpu": True,
        "reference": "Zbontar et al., arXiv 2018; Ronneberger et al., MICCAI 2015",
    },
    "dccnn": {
        "name": "DC-CNN",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_dccnn",
        "gpu": True,
        "reference": "Schlemper et al., IEEE TMI 2018",
    },
    "deep_admm_net": {
        "name": "Deep ADMM-Net",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_deep_admm_net",
        "gpu": True,
        "reference": "Sun, Li, Xu, NeurIPS 2016",
    },
    "ista_net_plus": {
        "name": "ISTA-Net+",
        "module": "pwm_core.recon.mri_solvers",
        "function": "run_ista_net_plus",
        "gpu": True,
        "reference": "Zhang & Ghanem, CVPR 2018",
    },
}


class MRIOperator:
    """Lightweight MRI forward/adjoint operator for solvers that need one.

    Single-coil Cartesian MRI with undersampling mask.
    """
    def __init__(self, mask: np.ndarray, image_size: int = 320):
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
    spec = SOLVERS[solver_key]

    # Apply cfg_override from solver spec
    if "cfg_override" in spec:
        for k, v in spec["cfg_override"].items():
            if k not in cfg:
                cfg[k] = v

    # Convert real+imag to complex
    y_complex = _to_complex(y)

    # Auto-create MRI operator when none provided
    if operator is None and y_complex.ndim == 2:
        mask = _infer_mask(y_complex)
        operator = MRIOperator(mask, y_complex.shape[0])

    # Load and call solver function
    fn = _load_fn(solver_key)
    result = fn(y_complex.astype(np.complex64), operator, cfg)

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
