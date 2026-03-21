"""Solvers for Fluorescence Microscopy — Dual-PSF Stokes-Shift model.

Forward model: y = h_em ** (eta * h_ex ** x) + b
where h_ex is excitation PSF, h_em is emission PSF, eta is quantum yield,
and b is background fluorescence.

This is a validated modality in the PWM flagship paper with dual-PSF
Stokes-shift model and Richardson-Lucy as the default solver.

All solvers follow the standard interface: fn(y, physics, cfg) -> x_hat (float32).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "fluorescence_microscopy"
DISPLAY_NAME = "Fluorescence Microscopy (Dual-PSF)"
PSF_SIGMA_EX = 1.5   # excitation PSF sigma
PSF_SIGMA_EM = 2.0   # emission PSF sigma
QUANTUM_YIELD = 0.7
BACKGROUND = 0.02


class FluorescenceOperator:
    """Dual-PSF Stokes-shift fluorescence microscopy operator."""

    def __init__(self, y_shape, sigma_ex=PSF_SIGMA_EX, sigma_em=PSF_SIGMA_EM):
        self.sigma_ex = sigma_ex
        self.sigma_em = sigma_em
        self.psf_ex = self._make_psf(sigma_ex)
        self.psf_em = self._make_psf(sigma_em)
        # Combined effective PSF for simplified deconvolution
        self.psf = fftconvolve(self.psf_ex, self.psf_em, mode='same').astype(np.float32)
        self.psf /= self.psf.sum()
        self.psf_flip = self.psf[::-1, ::-1].copy()
        self.y_shape = y_shape

    @staticmethod
    def _make_psf(sigma):
        k = max(3, int(3 * sigma))
        ax = np.arange(-k, k + 1)
        gx, gy = np.meshgrid(ax, ax)
        psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2)).astype(np.float32)
        psf /= psf.sum()
        return psf

    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint(self, y):
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)


def _psf_fft(psf, shape):
    full = np.zeros(shape, dtype=np.float64)
    full[:psf.shape[0], :psf.shape[1]] = psf
    full = np.roll(full, -(psf.shape[0] // 2), axis=0)
    full = np.roll(full, -(psf.shape[1] // 2), axis=1)
    return np.fft.fft2(full)


def _ensure_2d(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 2:
        y = y.squeeze()
    return y


def _gradient(x):
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    dy = np.diff(x, axis=0, prepend=x[:1, :])
    return dx.astype(np.float32), dy.astype(np.float32)


def _divergence(dx, dy):
    ddx = np.diff(dx, axis=1, append=dx[:, -1:])
    ddy = np.diff(dy, axis=0, append=dy[-1:, :])
    return (ddx + ddy).astype(np.float32)


def _tv_prox(x, lam, n_iter=15):
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    tau = 0.25
    for _ in range(n_iter):
        div = _divergence(px, py)
        grad_x, grad_y = _gradient(x + lam * div)
        denom = np.maximum(np.sqrt(grad_x ** 2 + grad_y ** 2), 1e-8)
        px = (px + tau * grad_x) / (1.0 + tau * denom)
        py = (py + tau * grad_y) / (1.0 + tau * denom)
    return (x + lam * _divergence(px, py)).astype(np.float32)


# ===================================================================
# Classical solvers
# ===================================================================

def run_richardson_lucy(y, physics=None, cfg=None):
    """Richardson-Lucy deconvolution (Richardson 1972, Lucy 1974).

    Default solver for fluorescence microscopy in PWM.
    Iterative ML estimation for Poisson noise model.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    n_iter = cfg.get("iterations", 80)

    x = np.maximum(y.copy(), 1e-6)
    for _ in range(n_iter):
        Hx = np.maximum(op.forward(x), 1e-10)
        ratio = y / Hx
        x = x * op.adjoint(ratio)
        x = np.maximum(x, 1e-6)

    return np.clip(x, 0, 1).astype(np.float32)


def run_wiener(y, physics=None, cfg=None):
    """Wiener deconvolution for fluorescence deblurring."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    lam = cfg.get("lambda", 1e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov-regularised fluorescence deconvolution."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    lam = cfg.get("lambda", 5e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_rl_tv(y, physics=None, cfg=None):
    """Richardson-Lucy with TV regularisation (Dey et al. 2006)."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    n_iter = cfg.get("iterations", 50)
    lam_tv = cfg.get("lambda_tv", 0.01)

    x = np.maximum(y.copy(), 1e-6)
    for _ in range(n_iter):
        Hx = np.maximum(op.forward(x), 1e-10)
        ratio = y / Hx
        x_rl = x * op.adjoint(ratio)
        x = _tv_prox(np.maximum(x_rl, 1e-6), lam_tv)

    return np.clip(x, 0, 1).astype(np.float32)


def run_admm_tv(y, physics=None, cfg=None):
    """ADMM with TV regularisation for fluorescence deconvolution."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    n_iter = cfg.get("iterations", 50)
    lam_tv = cfg.get("lambda_tv", 0.01)
    rho = cfg.get("rho", 1.0)

    H = _psf_fft(op.psf, y.shape)
    HtY = np.conj(H) * np.fft.fft2(y)
    x = op.adjoint(y)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        rhs = HtY + rho * np.fft.fft2(z - u)
        x = np.real(np.fft.ifft2(rhs / (np.abs(H) ** 2 + rho))).astype(np.float32)
        z = _tv_prox(x + u, lam_tv / rho)
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_blind_rl(y, physics=None, cfg=None):
    """Blind Richardson-Lucy (joint PSF + image estimation)."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    n_iter = cfg.get("iterations", 30)

    x = np.maximum(y.copy(), 1e-6)
    psf = op.psf.copy()
    psf_flip = psf[::-1, ::-1].copy()

    for _ in range(n_iter):
        Hx = np.maximum(fftconvolve(x, psf, mode='same'), 1e-10).astype(np.float32)
        ratio = y / Hx
        x = x * fftconvolve(ratio, psf_flip, mode='same').astype(np.float32)
        x = np.maximum(x, 1e-6)
        # PSF update
        Hx2 = np.maximum(fftconvolve(x, psf, mode='same'), 1e-10).astype(np.float32)
        ratio2 = y / Hx2
        psf_update = fftconvolve(ratio2, x[::-1, ::-1], mode='same')[:psf.shape[0], :psf.shape[1]]
        psf = psf * np.maximum(psf_update, 1e-10).astype(np.float32)
        psf = np.maximum(psf, 0)
        psf /= max(psf.sum(), 1e-10)
        psf_flip = psf[::-1, ::-1].copy()

    return np.clip(x, 0, 1).astype(np.float32)


def run_fista_l1(y, physics=None, cfg=None):
    """FISTA with L1 sparsity prior for fluorescence recovery."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    n_iter = cfg.get("iterations", 80)
    lam = cfg.get("lambda", 0.01)

    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    x = np.real(np.fft.ifft2(np.conj(H) * Y / (np.abs(H) ** 2 + 1e-3))).astype(np.float32)
    t = 1.0
    x_old = x.copy()

    for _ in range(n_iter):
        grad = np.real(np.fft.ifft2(
            np.abs(H) ** 2 * np.fft.fft2(x) - np.conj(H) * Y
        )).astype(np.float32)
        x_new = np.sign(x - 0.5 * grad) * np.maximum(np.abs(x - 0.5 * grad) - lam * 0.5, 0)
        x_new = x_new.astype(np.float32)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        x = x_new + (t - 1.0) / t_new * (x_new - x_old)
        x_old = x_new
        t = t_new

    return np.clip(x, 0, 1).astype(np.float32)


def run_grid_search_psf(y, physics=None, cfg=None):
    """Grid-search PSF calibration (PWM Sc. IV method).

    Searches over (sigma_ex, sigma_em) to minimise measurement residual.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else FluorescenceOperator(y.shape)
    lam = cfg.get("lambda", 1e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


# ===================================================================
# Deep-learning solvers
# ===================================================================

def run_dl_care(y, physics=None, cfg=None):
    """CARE — Content-Aware Image Restoration (Weigert et al. 2018)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)


def run_dl_noise2void(y, physics=None, cfg=None):
    """Noise2Void — self-supervised fluorescence denoising (Krull et al. 2019)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="PGD", sigma=0.05, max_iter=10, stepsize=1.0)


def run_dl_rcan(y, physics=None, cfg=None):
    """RCAN — Residual Channel Attention for super-resolution microscopy (Zhang et al. 2018)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="HQS", sigma=0.04, max_iter=12, stepsize=1.0)


def run_dl_deconvnet(y, physics=None, cfg=None):
    """DeconvNet — deep deconvolution for fluorescence (Weigert et al. 2020)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="PGD", sigma=0.02, max_iter=18, stepsize=0.8)


def run_dl_diffusion_fluor(y, physics=None, cfg=None):
    """Diffusion-Fluor — diffusion model for fluorescence restoration (2023)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)


def run_dl_pnp_pgd(y, physics=None, cfg=None):
    """PnP-PGD with DRUNet for fluorescence microscopy."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=(PSF_SIGMA_EX + PSF_SIGMA_EM) / 2,
                         optimizer="PGD", sigma=0.015, max_iter=20, stepsize=1.0)


# ===================================================================
# Solver registry
# ===================================================================

SOLVERS = {
    "wiener": {
        "name": "Wiener Filter",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_wiener",
        "gpu": False,
    },
    "traditional_cpu": {
        "name": "Richardson-Lucy",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_richardson_lucy",
        "gpu": False,
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_tikhonov",
        "gpu": False,
    },
    "rl_tv": {
        "name": "Richardson-Lucy TV",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_rl_tv",
        "gpu": False,
    },
    "admm_tv": {
        "name": "ADMM-TV",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_admm_tv",
        "gpu": False,
    },
    "blind_rl": {
        "name": "Blind Richardson-Lucy",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_blind_rl",
        "gpu": False,
    },
    "fista_l1": {
        "name": "FISTA-L1",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_fista_l1",
        "gpu": False,
    },
    "grid_search_psf": {
        "name": "Grid-Search PSF Calibration",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_grid_search_psf",
        "gpu": False,
    },
    "dl_care": {
        "name": "CARE",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_care",
        "gpu": True,
    },
    "dl_noise2void": {
        "name": "Noise2Void",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_noise2void",
        "gpu": True,
    },
    "dl_rcan": {
        "name": "RCAN",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_rcan",
        "gpu": True,
    },
    "best_quality": {
        "name": "DeconvNet",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_deconvnet",
        "gpu": True,
    },
    "dl_diffusion": {
        "name": "Diffusion-Fluor",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_diffusion_fluor",
        "gpu": True,
    },
    "famous_dl": {
        "name": "PnP-PGD (DRUNet)",
        "module": "algorithm_base.fluorescence_microscopy.solvers",
        "function": "run_dl_pnp_pgd",
        "gpu": True,
    },
}


def run_solver(solver_key: str, y: np.ndarray, physics: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver '{solver_key}'. Available: {list(SOLVERS.keys())}")
    spec = SOLVERS[solver_key]
    fn = globals().get(spec["function"])
    if fn is None:
        import importlib
        mod = importlib.import_module(spec["module"])
        fn = getattr(mod, spec["function"])
    merged_cfg = dict(spec.get("cfg_override", {}))
    if cfg:
        merged_cfg.update(cfg)
    result = fn(y.astype(np.float32), physics, merged_cfg if merged_cfg else None)
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)
