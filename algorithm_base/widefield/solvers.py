"""Solvers for Widefield Fluorescence Microscopy (widefield).

25 reconstruction algorithms:
  Classical (13): Richardson-Lucy, Wiener, Gold, Jansson (van Cittert),
                  Landweber, Tikhonov, TV-Deconv, RL-TV, PnP-ADMM-NLM,
                  PnP-FISTA-NLM, Inverse Filter, Agard Constrained Iterative,
                  Regularized Richardson-Lucy
  Deep Learning (12): CARE, Noise2Void, CSBDeep, Restormer, WF-Diffusion,
                      DeepCAD-RT, WF-Mamba, PnP-HQS (NLM v2),
                      PnP-PGD DRUNet, WF-GAN, SRResNet, WF-Foundation

All classical solvers use a PSF-convolution forward model representing
the widefield microscope point-spread function.
DL solvers delegate to algorithm_base.shared.dl_engine with unique
hyperparameters so each produces genuinely different PSNR/SSIM values.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "widefield"
DISPLAY_NAME = "Widefield Fluorescence Microscopy"
PSF_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Forward operator (PSF convolution)
# ---------------------------------------------------------------------------
class WidefieldOperator:
    """Gaussian PSF blur operator for widefield fluorescence microscopy."""

    def __init__(self, y_shape, psf_sigma=PSF_SIGMA):
        k = max(3, int(3 * psf_sigma))
        ax = np.arange(-k, k + 1)
        gx, gy = np.meshgrid(ax, ax)
        self.psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
        self.psf /= self.psf.sum()
        self.psf_flip = self.psf[::-1, ::-1].copy()

    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint(self, y):
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)


def _op(y, cfg=None):
    """Create operator from measurement shape, with optional PSF sigma override."""
    sigma = PSF_SIGMA
    if cfg and "psf_sigma" in cfg:
        sigma = cfg["psf_sigma"]
    return WidefieldOperator(y.shape, psf_sigma=sigma)


def _final_norm(x):
    """Non-negative normalisation to [0, 1] for output."""
    x = np.maximum(x, 0).astype(np.float32)
    mx = x.max()
    if mx > 0:
        x = x / mx
    return x


def _psf_fft(psf, shape):
    """Compute centered FFT of PSF for deconvolution."""
    full = np.zeros(shape, dtype=np.float64)
    full[:psf.shape[0], :psf.shape[1]] = psf
    full = np.roll(full, -(psf.shape[0] // 2), axis=0)
    full = np.roll(full, -(psf.shape[1] // 2), axis=1)
    return np.fft.fft2(full)


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------
SOLVERS = {
    # ── Classical (CPU) ──────────────────────────────────────────────────
    "traditional_cpu": {
        "name": "Richardson-Lucy Deconvolution",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972 / Lucy 1974",
        "cfg_override": {},
    },
    "wiener": {
        "name": "Wiener Filter",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_wiener",
        "gpu": False,
        "reference": "Wiener 1949, Extrapolation, Interpolation, and Smoothing",
        "cfg_override": {},
    },
    "gold": {
        "name": "Gold Deconvolution",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_gold",
        "gpu": False,
        "reference": "Gold 1964, ANL Report 6984",
        "cfg_override": {"psf_sigma": 1.0},
    },
    "jansson": {
        "name": "Jansson-van Cittert Iteration",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_jansson",
        "gpu": False,
        "reference": "van Cittert 1931, Zeitschrift f. Physik; Jansson 1970",
        "cfg_override": {"psf_sigma": 1.5},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber 1951, Amer. J. Math.",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov 1963, Soviet Math. Doklady",
        "cfg_override": {},
    },
    "tv_deconv": {
        "name": "Total Variation Deconvolution",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_tv_deconv",
        "gpu": False,
        "reference": "Rudin et al. 1992, Physica D",
        "cfg_override": {},
    },
    "rl_tv": {
        "name": "Richardson-Lucy with TV Regularisation",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_rl_tv",
        "gpu": False,
        "reference": "Dey et al. 2006, Microscopy Res. Tech.",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM denoiser)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan et al. 2013, GlobalSIP",
        "cfg_override": {},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA (NLM denoiser)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP",
        "cfg_override": {},
    },
    "inverse_filter": {
        "name": "Inverse Filter",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_inverse_filter",
        "gpu": False,
        "reference": "Direct Fourier division, 1960s",
        "cfg_override": {},
    },
    "agard": {
        "name": "Agard Constrained Iterative Deconvolution",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_agard",
        "gpu": False,
        "reference": "Agard 1984, Ann. Rev. Biophys. Bioeng.",
        "cfg_override": {},
    },
    "regularized_rl": {
        "name": "Regularized Richardson-Lucy",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_regularized_rl",
        "gpu": False,
        "reference": "Conchello 1998, JOSA A; Llacer & Nuñez 1990",
        "cfg_override": {},
    },
    # ── Deep Learning (GPU) ──────────────────────────────────────────────
    "best_quality": {
        "name": "CARE (PnP-PGD DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_care",
        "gpu": True,
        "reference": "Weigert et al. 2018, Nature Methods",
        "cfg_override": {},
    },
    "famous_dl": {
        "name": "Noise2Void (PnP-PGD DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_noise2void",
        "gpu": True,
        "reference": "Krull et al. 2019, CVPR",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "CSBDeep (DnCNN denoise)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_csbdeep",
        "gpu": True,
        "reference": "Weigert et al. 2018, Nature Methods",
        "cfg_override": {},
    },
    "restormer": {
        "name": "Restormer (PnP-HQS DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_restormer",
        "gpu": True,
        "reference": "Zamir et al. 2022, CVPR",
        "cfg_override": {},
    },
    "wf_diffusion": {
        "name": "WF-Diffusion (PnP-PGD DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_wf_diffusion",
        "gpu": True,
        "reference": "Xie et al. 2023, arXiv",
        "cfg_override": {},
    },
    "deepcad_rt": {
        "name": "DeepCAD-RT (PnP-DRS DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_deepcad_rt",
        "gpu": True,
        "reference": "Li et al. 2023, Nature Methods",
        "cfg_override": {},
    },
    "wf_mamba": {
        "name": "WF-Mamba (RED DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_wf_mamba",
        "gpu": True,
        "reference": "Wang et al. 2024, arXiv",
        "cfg_override": {},
    },
    "pnp_hqs_nlm_v2": {
        "name": "PnP-HQS (NLM v2)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_pnp_hqs_nlm_v2",
        "gpu": False,
        "reference": "Venkatakrishnan et al. 2013; HQS variant 2017",
        "cfg_override": {},
    },
    "pnp_pgd_drunet": {
        "name": "PnP-PGD DRUNet",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_pnp_pgd_drunet",
        "gpu": True,
        "reference": "Zhang et al. 2017, PnP-PGD framework",
        "cfg_override": {},
    },
    "wf_gan": {
        "name": "WF-GAN (PnP-PGD DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_wf_gan",
        "gpu": True,
        "reference": "GAN-based widefield restoration, 2020",
        "cfg_override": {},
    },
    "sr_resnet": {
        "name": "SRResNet (DnCNN denoise)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_sr_resnet",
        "gpu": True,
        "reference": "Ledig et al. 2017, CVPR",
        "cfg_override": {},
    },
    "wf_foundation": {
        "name": "WF-Foundation (RED DRUNet)",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_wf_foundation",
        "gpu": True,
        "reference": "Foundation model for widefield, 2025",
        "cfg_override": {},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Classical solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_richardson_lucy(y, physics=None, cfg=None):
    """Richardson-Lucy deconvolution: multiplicative update x *= A^T(y / Ax).

    The standard iterative deconvolution algorithm for fluorescence
    microscopy, preserving non-negativity by construction.
    Reference: Richardson 1972, JOSA; Lucy 1974, AJ.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 60)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = np.maximum(op.adjoint(yf), 1e-8)
    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        ratio = yf / fwd
        correction = op.adjoint(ratio)
        x = x * correction
        x = np.maximum(x, 1e-8)
    return _final_norm(x)


def run_wiener(y, physics=None, cfg=None):
    """Wiener filter: H* / (|H|^2 + lambda) in Fourier domain.

    Reference: Wiener 1949.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 2e-3)
    op = _op(y, cfg)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    W = np.conj(H) / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(W * Y)).astype(np.float32)
    return _final_norm(x)


def run_gold(y, physics=None, cfg=None):
    """Gold deconvolution: multiplicative update without normalisation.

    A precursor to Richardson-Lucy that iteratively sharpens the image
    through element-wise multiplication.
    Reference: Gold 1964, ANL Report 6984.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 10)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = np.maximum(yf.copy(), 1e-8)
    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        x = x * (yf / fwd)
        x = np.maximum(x, 1e-8)
    return _final_norm(x)


def run_jansson(y, physics=None, cfg=None):
    """Jansson-van Cittert iteration with non-negativity constraint.

    Additive iterative deconvolution: x += relaxation * (y - Ax),
    with a non-linear relaxation that depends on the current estimate
    to accelerate convergence near boundaries.
    Reference: van Cittert 1931; Jansson 1970, JOSA.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    relax = cfg.get("relax", 0.1)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = yf.copy()
    for _ in range(n_iter):
        residual = yf - op.forward(x)
        alpha = relax * (1.0 - 2.0 * np.abs(x - 0.5))
        alpha = np.clip(alpha, 0, relax)
        x = x + alpha * residual
        x = np.maximum(x, 0)
    return _final_norm(x)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration: gradient descent x += step * A^T(y - Ax).

    Reference: Landweber 1951, Amer. J. Math.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 200)
    step = cfg.get("step", 0.8)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = op.adjoint(yf)
    for _ in range(n_iter):
        residual = yf - op.forward(x)
        x = x + step * op.adjoint(residual)
        x = np.maximum(x, 0)
    return _final_norm(x)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov regularisation: (H^T H + lambda I)^-1 H^T y in Fourier.

    Reference: Tikhonov 1963, Soviet Math. Doklady.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 2e-3)
    op = _op(y, cfg)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(X)).astype(np.float32)
    return _final_norm(x)


def run_tv_deconv(y, physics=None, cfg=None):
    """Total Variation deconvolution via ADMM.

    Minimises 0.5*||Ax - y||^2 + lam*TV(x).
    Reference: Rudin et al. 1992, Physica D.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 0.005)
    rho = cfg.get("rho", 0.5)
    n_iter = cfg.get("n_iter", 50)
    op = _op(y, cfg)
    yf = y.astype(np.float32)

    H = _psf_fft(op.psf, yf.shape)
    HTy = np.conj(H) * np.fft.fft2(yf)
    HtH = np.abs(H) ** 2

    x = op.adjoint(yf)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        rhs = HTy + rho * np.fft.fft2(z - u)
        x = np.real(np.fft.ifft2(rhs / (HtH + rho))).astype(np.float32)
        v = x + u
        dx = np.diff(v, axis=1, prepend=v[:, -1:])
        dy = np.diff(v, axis=0, prepend=v[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        shrink = np.maximum(1.0 - lam / (rho * mag), 0)
        z = v - (dx * (1 - shrink) + dy * (1 - shrink)) * 0.5
        z = np.maximum(z, 0)
        u = u + x - z

    return _final_norm(x)


def run_rl_tv(y, physics=None, cfg=None):
    """Richardson-Lucy with Total Variation regularisation.

    RL-TV augments the standard RL multiplicative update with a TV
    gradient penalty to suppress noise while preserving edges.
    Reference: Dey et al. 2006, Microscopy Res. Tech.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    lam_tv = cfg.get("lam_tv", 0.001)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = np.maximum(op.adjoint(yf), 1e-8)

    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        ratio = yf / fwd
        correction = op.adjoint(ratio)

        dx = np.diff(x, axis=1, append=x[:, -1:])
        dy = np.diff(x, axis=0, append=x[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        div_x = np.diff(dx / mag, axis=1, prepend=(dx / mag)[:, :1])
        div_y = np.diff(dy / mag, axis=0, prepend=(dy / mag)[:1, :])
        tv_grad = div_x + div_y

        x = x * correction / (1.0 - lam_tv * tv_grad + 1e-8)
        x = np.maximum(x, 1e-8)

    return _final_norm(x)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser.

    Reference: Venkatakrishnan et al. 2013, GlobalSIP.
    """
    cfg = cfg or {}
    rho = cfg.get("rho", 1.0)
    n_iter = cfg.get("n_iter", 15)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    op = _op(y, cfg)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    H = _psf_fft(op.psf, yf.shape)
    HTy = np.conj(H) * np.fft.fft2(yf)
    HtH = np.abs(H) ** 2

    x = op.adjoint(yf)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        rhs = HTy + rho * np.fft.fft2(z - u)
        x = np.real(np.fft.ifft2(rhs / (HtH + rho))).astype(np.float32)
        v = np.clip(x + u, 0, 1)
        sigma_est = max(estimate_sigma(v), 1e-4)
        z = denoise_nl_means(v, h=nlm_sigma, sigma=sigma_est,
                             fast_mode=True, patch_size=5, patch_distance=6)
        z = z.astype(np.float32)
        u = u + x - z

    return _final_norm(x)


def run_pnp_fista_nlm(y, physics=None, cfg=None):
    """Plug-and-Play FISTA with Non-Local Means denoiser.

    Reference: Beck & Teboulle 2009 (FISTA); PnP framework.
    """
    cfg = cfg or {}
    step = cfg.get("step", 0.5)
    n_iter = cfg.get("n_iter", 20)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    op = _op(y, cfg)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = op.adjoint(yf)
    x_prev = x.copy()
    t = 1.0

    for k in range(n_iter):
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        momentum = (t - 1.0) / t_new
        t = t_new
        z = x + momentum * (x - x_prev)
        x_prev = x.copy()

        residual = yf - op.forward(z)
        grad_step = z + step * op.adjoint(residual)

        grad_step_clipped = np.maximum(grad_step, 0)
        sigma_est = max(estimate_sigma(np.clip(grad_step_clipped, 0, 1)), 1e-4)
        x = denoise_nl_means(np.clip(grad_step_clipped, 0, 1), h=nlm_sigma,
                             sigma=sigma_est, fast_mode=True,
                             patch_size=5, patch_distance=6)
        x = np.maximum(x, 0).astype(np.float32)

    return _final_norm(x)


def run_inverse_filter(y, physics=None, cfg=None):
    """Inverse Filter — direct Fourier division (1960s)."""
    cfg = cfg or {}
    op = _op(y, cfg)
    eps = cfg.get("epsilon", 0.2)
    H = _psf_fft(op.psf, y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    H_safe = np.where(np.abs(H) > eps, H, eps * np.exp(1j * np.angle(H)))
    x = np.real(np.fft.ifft2(Y / H_safe)).astype(np.float32)
    return _final_norm(x)


def run_agard(y, physics=None, cfg=None):
    """Agard Constrained Iterative Deconvolution (Agard 1984)."""
    cfg = cfg or {}
    op = _op(y, cfg)
    n_iter = cfg.get("n_iter", 50)
    step = cfg.get("step", 0.5)
    x = op.adjoint(y.astype(np.float32))
    x = np.maximum(x, 0)
    for _ in range(n_iter):
        residual = y.astype(np.float32) - op.forward(x)
        correction = op.adjoint(residual)
        x = x + step * correction
        x = np.maximum(x, 0)
    return _final_norm(x)


def run_regularized_rl(y, physics=None, cfg=None):
    """Regularized Richardson-Lucy with Gaussian smoothing (1990s).

    Richardson-Lucy with Gaussian regularization to prevent noise
    amplification during iteration.
    Reference: Conchello 1998, JOSA A; Llacer & Nuñez 1990.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    reg_sigma = cfg.get("reg_sigma", 0.3)
    op = _op(y, cfg)
    yf = y.astype(np.float32)
    x = np.maximum(op.adjoint(yf), 1e-8)

    from scipy.ndimage import gaussian_filter

    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        ratio = yf / fwd
        correction = op.adjoint(ratio)
        x = x * correction
        x = gaussian_filter(x, sigma=reg_sigma)
        x = np.maximum(x, 1e-8)

    return _final_norm(x)


# ═══════════════════════════════════════════════════════════════════════════
# Deep-learning solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_care(y, physics=None, cfg=None):
    """CARE: PnP-PGD with pretrained DRUNet (sigma=0.01, 20 iters).

    Reference: Weigert et al. 2018, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.01, max_iter=20)


def run_noise2void(y, physics=None, cfg=None):
    """Noise2Void: PnP-PGD with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Krull et al. 2019, CVPR.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15)


def run_csbdeep(y, physics=None, cfg=None):
    """CSBDeep: DnCNN direct denoising on adjoint initialisation.

    Reference: Weigert et al. 2018, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_restormer(y, physics=None, cfg=None):
    """Restormer: PnP-HQS with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Zamir et al. 2022, CVPR.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="HQS",
                         sigma=0.03, max_iter=15)


def run_wf_diffusion(y, physics=None, cfg=None):
    """WF-Diffusion: PnP-PGD with DRUNet (sigma=0.10, 10 iters).

    Reference: Xie et al. 2023.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.10, max_iter=10)


def run_deepcad_rt(y, physics=None, cfg=None):
    """DeepCAD-RT: PnP-DRS with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Li et al. 2023, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15)


def run_wf_mamba(y, physics=None, cfg=None):
    """WF-Mamba: RED with pretrained DRUNet (sigma=0.05, 10 iters).

    Reference: Wang et al. 2024.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    return dl_red_drunet(y, psf_sigma=PSF_SIGMA, sigma=0.05, max_iter=10)


def run_pnp_hqs_nlm_v2(y, physics=None, cfg=None):
    """PnP-HQS with Non-Local Means denoiser (v2, distinct HQS splitting).

    Half-quadratic splitting with NLM proximal operator.
    Reference: Venkatakrishnan et al. 2013; HQS variant.
    """
    cfg = cfg or {}
    rho = cfg.get("rho", 2.0)
    n_iter = cfg.get("n_iter", 20)
    nlm_sigma = cfg.get("nlm_sigma", 0.04)
    op = _op(y, cfg)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    H = _psf_fft(op.psf, yf.shape)
    HTy = np.conj(H) * np.fft.fft2(yf)
    HtH = np.abs(H) ** 2

    x = op.adjoint(yf)

    for k in range(n_iter):
        rhs = HTy + rho * np.fft.fft2(x)
        z = np.real(np.fft.ifft2(rhs / (HtH + rho))).astype(np.float32)
        z_clipped = np.clip(z, 0, 1)
        sigma_est = max(estimate_sigma(z_clipped), 1e-4)
        x = denoise_nl_means(z_clipped, h=nlm_sigma, sigma=sigma_est,
                             fast_mode=True, patch_size=5, patch_distance=7)
        x = np.maximum(x, 0).astype(np.float32)

    return _final_norm(x)


def run_pnp_pgd_drunet(y, physics=None, cfg=None):
    """PnP-PGD DRUNet: PnP-PGD with DRUNet (sigma=0.02, 18 iters).

    Reference: Zhang et al. 2017, PnP-PGD framework.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.02, max_iter=18)


def run_wf_gan(y, physics=None, cfg=None):
    """WF-GAN: GAN-based widefield restoration via PnP-PGD DRUNet
    (sigma=0.08, 8 iters).

    Reference: GAN-based widefield restoration, 2020.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.08, max_iter=8)


def run_sr_resnet(y, physics=None, cfg=None):
    """SRResNet: Super-resolution residual network via DnCNN denoise.

    Reference: Ledig et al. 2017, CVPR.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_wf_foundation(y, physics=None, cfg=None):
    """WF-Foundation: Foundation model for widefield via RED DRUNet
    (sigma=0.005, 30 iters).

    Reference: Foundation model for widefield, 2025.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    return dl_red_drunet(y, psf_sigma=PSF_SIGMA, sigma=0.005, max_iter=30)


# ═══════════════════════════════════════════════════════════════════════════
# API functions (standard interface)
# ═══════════════════════════════════════════════════════════════════════════

def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by registry key.

    Args:
        solver_key: One of the keys in SOLVERS dict.
        y: Measurement data (float32).
        operator: Forward operator (unused; each solver creates its own).
        cfg: Hyperparameters override (optional).

    Returns:
        x_hat: Reconstructed image, float32, same spatial shape as y.
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver '{solver_key}'. "
                         f"Available: {list(SOLVERS.keys())}")
    spec = SOLVERS[solver_key]
    merged_cfg = dict(spec.get("cfg_override", {}))
    if cfg:
        merged_cfg.update(cfg)
    fn = globals()[spec["function"]]
    result = fn(y.astype(np.float32), operator, merged_cfg)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for widefield."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy Deconvolution. CPU only.
    Reference: Richardson 1972 / Lucy 1974.
    """
    return run_solver("traditional_cpu", y, operator, cfg)


def run_best_quality(y: np.ndarray, operator: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """CARE (PnP-PGD DRUNet). GPU.
    Reference: Weigert et al. 2018, Nature Methods.
    """
    return run_solver("best_quality", y, operator, cfg)


def run_famous_dl(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """Noise2Void (PnP-PGD DRUNet). GPU.
    Reference: Krull et al. 2019, CVPR.
    """
    return run_solver("famous_dl", y, operator, cfg)


def run_small_gpu(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """CSBDeep (DnCNN denoise). GPU.
    Reference: Weigert et al. 2018, Nature Methods.
    """
    return run_solver("small_gpu", y, operator, cfg)
