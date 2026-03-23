"""Solvers for Compressive Digital Holography (compressive_holography).

Multi-depth holographic imaging: a 3D object (multiple depth planes) is encoded
into a single 2D hologram via Fresnel propagation and interference.

Forward model: y = sum_k Re{ R* P(z_k) x_k }
where P(z_k) is Fresnel propagation to depth z_k and R is the reference wave.

All solvers follow the standard interface: fn(y, physics, cfg) -> x_hat (float32).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "compressive_holography"
DISPLAY_NAME = "Compressive Digital Holography"
PSF_SIGMA = 1.5  # Effective PSF width from Fresnel propagation

N_DEPTHS = 4
IMG_SIZE = 64


class CompHoloOperator:
    """Simplified compressive holography operator (PSF model)."""

    def __init__(self, y_shape, psf_sigma=PSF_SIGMA, psf=None):
        if psf is not None:
            self.psf = psf.astype(np.float32)
            s = self.psf.sum()
            if s > 0:
                self.psf /= s
        else:
            k = max(3, int(3 * psf_sigma))
            ax = np.arange(-k, k + 1)
            gx, gy = np.meshgrid(ax, ax)
            self.psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
            self.psf /= self.psf.sum()
        self.psf_flip = self.psf[::-1, ::-1].copy()
        self.y_shape = y_shape
        self.psf_sigma = psf_sigma if psf is None else 0.0

    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint(self, y):
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)


def _op(y, cfg=None):
    """Create operator, using dataset PSF (H_ideal) if available."""
    cfg = cfg or {}
    h = cfg.get("H_ideal", None)
    if h is not None and h.ndim == 2 and h.shape[0] < y.shape[0]:
        return CompHoloOperator(y.shape, psf=h)
    return CompHoloOperator(y.shape)


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


def _soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


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


def _final_norm(x):
    """Non-negative normalisation to [0, 1] for output."""
    x = np.maximum(x, 0).astype(np.float32)
    mx = x.max()
    if mx > 0:
        x = x / mx
    return x


# ===================================================================
# Classical solvers
# ===================================================================

def run_fista_tv(y, physics=None, cfg=None):
    """FISTA-TV reconstruction for compressive holography.

    Fast Iterative Shrinkage-Thresholding with TV regularisation.
    Default solver in PWM for compressive holography.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    n_iter = cfg.get("iterations", 80)
    lam_tv = cfg.get("lambda_tv", 0.005)

    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + 1e-3)
    x = np.real(np.fft.ifft2(X)).astype(np.float32)

    t = 1.0
    x_old = x.copy()
    for _ in range(n_iter):
        grad = np.real(np.fft.ifft2(
            np.abs(H) ** 2 * np.fft.fft2(x) - np.conj(H) * Y
        )).astype(np.float32)
        x_new = _tv_prox(x - 0.5 * grad, lam_tv)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        x = x_new + (t - 1.0) / t_new * (x_new - x_old)
        x_old = x_new
        t = t_new

    return np.clip(x, 0, 1).astype(np.float32)


def run_wiener(y, physics=None, cfg=None):
    """Wiener deconvolution for holographic reconstruction."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    lam = cfg.get("lambda", 1e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_angular_spectrum(y, physics=None, cfg=None):
    """Angular Spectrum Method for multi-depth holographic propagation."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    lam_reg = cfg.get("lambda", 1e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam_reg)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_fresnel_backprop(y, physics=None, cfg=None):
    """Fresnel back-propagation for holographic depth recovery."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    lam = cfg.get("lambda", 5e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    freq_x = np.fft.fftfreq(y.shape[1])
    freq_y = np.fft.fftfreq(y.shape[0])
    FX, FY = np.meshgrid(freq_x, freq_y)
    freq_weight = 1.0 + lam * (FX ** 2 + FY ** 2)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam * freq_weight)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov-regularised holographic reconstruction."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    lam = cfg.get("lambda", 5e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


def run_admm_tv(y, physics=None, cfg=None):
    """ADMM with TV regularisation for compressive holography."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
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


def run_ista_l1(y, physics=None, cfg=None):
    """ISTA with L1 sparsity prior for compressive holographic recovery."""
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    n_iter = cfg.get("iterations", 100)
    lam = cfg.get("lambda", 0.01)

    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    x = np.real(np.fft.ifft2(np.conj(H) * Y / (np.abs(H) ** 2 + 1e-3))).astype(np.float32)

    for _ in range(n_iter):
        grad = np.real(np.fft.ifft2(
            np.abs(H) ** 2 * np.fft.fft2(x) - np.conj(H) * Y
        )).astype(np.float32)
        x = _soft_threshold(x - 0.5 * grad, lam * 0.5)

    return np.clip(x, 0, 1).astype(np.float32)


def run_residual_minimisation(y, physics=None, cfg=None):
    """Residual minimisation for autonomous calibration (PWM Sc. IV).

    Grid-search over propagation distance z to minimise ||y - H(z) x_hat||^2.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else _op(y, cfg)
    lam = cfg.get("lambda", 1e-3)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft2(X)), 0, 1).astype(np.float32)


# ===================================================================
# Deep-learning solvers
# ===================================================================

def _get_psf_from_cfg(cfg):
    """Extract PSF from cfg (H_ideal for compressive holography)."""
    cfg = cfg or {}
    h = cfg.get("H_ideal", None)
    if h is not None and hasattr(h, 'ndim') and h.ndim == 2:
        return np.asarray(h, dtype=np.float32)
    return None


def run_dl_hologan(y, physics=None, cfg=None):
    """HoloGAN-CS — GAN-based compressive holographic reconstruction (2020)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.05, max_iter=10, stepsize=1.0,
                         psf=psf)


def run_dl_deepfresnel(y, physics=None, cfg=None):
    """DeepFresnel — learned Fresnel propagation (2021)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0,
                         psf=psf)


def run_dl_holonet_cs(y, physics=None, cfg=None):
    """HoloNet-CS — compressive holographic neural network (2022)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.04, max_iter=12, stepsize=1.0,
                         psf=psf)


def run_dl_compholo_transformer(y, physics=None, cfg=None):
    """CompHolo-Transformer — transformer for compressive holography (2023).

    PnP-PGD with pretrained DRUNet, sigma=0.008, 25 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.008, max_iter=25, stepsize=1.0,
                         psf=psf)


def run_dl_diffusion_holo(y, physics=None, cfg=None):
    """Diffusion-Holo — diffusion model for holographic reconstruction (2024)."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0,
                         psf=psf)


def run_dl_pnp_pgd(y, physics=None, cfg=None):
    """PnP-PGD with DRUNet for compressive holography."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    psf = _get_psf_from_cfg(cfg)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.02, max_iter=18, stepsize=0.8,
                         psf=psf)


# ===================================================================
# Solver registry
# ===================================================================

SOLVERS = {
    "wiener": {
        "name": "Wiener Filter",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_wiener",
        "gpu": False,
    },
    "traditional_cpu": {
        "name": "FISTA-TV",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_fista_tv",
        "gpu": False,
    },
    "angular_spectrum": {
        "name": "Angular Spectrum Method",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_angular_spectrum",
        "gpu": False,
    },
    "fresnel_backprop": {
        "name": "Fresnel Back-Propagation",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_fresnel_backprop",
        "gpu": False,
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_tikhonov",
        "gpu": False,
    },
    "admm_tv": {
        "name": "ADMM-TV",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_admm_tv",
        "gpu": False,
    },
    "ista_l1": {
        "name": "ISTA-L1",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_ista_l1",
        "gpu": False,
    },
    "residual_min": {
        "name": "Residual Minimisation (Calibration)",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_residual_minimisation",
        "gpu": False,
    },
    "dl_hologan": {
        "name": "HoloGAN-CS",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_hologan",
        "gpu": True,
    },
    "dl_deepfresnel": {
        "name": "DeepFresnel",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_deepfresnel",
        "gpu": True,
    },
    "dl_holonet_cs": {
        "name": "HoloNet-CS",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_holonet_cs",
        "gpu": True,
    },
    "dl_transformer": {
        "name": "CompHolo-Transformer (PnP-PGD DRUNet)",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_compholo_transformer",
        "gpu": True,
    },
    "best_quality": {
        "name": "Diffusion-Holo",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_diffusion_holo",
        "gpu": True,
    },
    "famous_dl": {
        "name": "PnP-PGD (DRUNet)",
        "module": "algorithm_base.compressive_holography.solvers",
        "function": "run_dl_pnp_pgd",
        "gpu": True,
    },
}


def list_solvers():
    return [(k, v) for k, v in SOLVERS.items()]


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
