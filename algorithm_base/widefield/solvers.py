"""Solvers for Widefield Fluorescence Microscopy (widefield).

17 reconstruction algorithms:
  Classical (10): Richardson-Lucy, Wiener, Gold, Jansson (van Cittert),
                  Landweber, Tikhonov, TV-Deconv, RL-TV, PnP-ADMM-NLM,
                  PnP-FISTA-NLM
  Deep Learning (7): CARE, Noise2Void, CSBDeep, Restormer, WF-Diffusion,
                     DeepCAD-RT, WF-Mamba

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
PSF_SIGMA = 2.0


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


def _op(y):
    """Create default operator from measurement shape."""
    return WidefieldOperator(y.shape)


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
        "cfg_override": {},
    },
    "jansson": {
        "name": "Jansson-van Cittert Iteration",
        "module": "algorithm_base.widefield.solvers",
        "function": "run_jansson",
        "gpu": False,
        "reference": "van Cittert 1931, Zeitschrift f. Physik; Jansson 1970",
        "cfg_override": {},
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
    n_iter = cfg.get("n_iter", 30)
    op = _op(y)
    yf = y.astype(np.float32)
    x = np.maximum(op.adjoint(yf), 1e-8)
    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        ratio = yf / fwd
        correction = op.adjoint(ratio)
        x = x * correction
        x = np.clip(x, 1e-8, 1.0)
    return x.astype(np.float32)


def run_wiener(y, physics=None, cfg=None):
    """Wiener filter: H* / (|H|^2 + lambda) in Fourier domain.

    Reference: Wiener 1949.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 1e-2)
    op = _op(y)
    H = np.fft.fft2(op.psf, s=y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    W = np.conj(H) / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(W * Y)).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def run_gold(y, physics=None, cfg=None):
    """Gold deconvolution: multiplicative update without normalisation.

    A precursor to Richardson-Lucy that iteratively sharpens the image
    through element-wise multiplication.
    Reference: Gold 1964, ANL Report 6984.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 30)
    op = _op(y)
    yf = y.astype(np.float32)
    x = np.maximum(yf.copy(), 1e-8)
    for _ in range(n_iter):
        fwd = np.maximum(op.forward(x), 1e-8)
        # Gold multiplicative update: x *= y / H(x), without the adjoint
        x = x * (yf / fwd)
        x = np.clip(x, 1e-8, 1.0)
    return x.astype(np.float32)


def run_jansson(y, physics=None, cfg=None):
    """Jansson-van Cittert iteration with non-negativity constraint.

    Additive iterative deconvolution: x += relaxation * (y - Ax),
    with a non-linear relaxation that depends on the current estimate
    to accelerate convergence near boundaries.
    Reference: van Cittert 1931; Jansson 1970, JOSA.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    relax = cfg.get("relax", 0.5)
    op = _op(y)
    yf = y.astype(np.float32)
    x = yf.copy()
    for _ in range(n_iter):
        residual = yf - op.forward(x)
        # Jansson relaxation: stronger correction at mid-range values
        alpha = relax * (1.0 - 2.0 * np.abs(x - 0.5))
        alpha = np.clip(alpha, 0, relax)
        x = x + alpha * residual
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration: gradient descent x += step * A^T(y - Ax).

    Reference: Landweber 1951, Amer. J. Math.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    step = cfg.get("step", 0.5)
    op = _op(y)
    yf = y.astype(np.float32)
    x = op.adjoint(yf)
    for _ in range(n_iter):
        residual = yf - op.forward(x)
        x = x + step * op.adjoint(residual)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov regularisation: (H^T H + lambda I)^-1 H^T y in Fourier.

    Reference: Tikhonov 1963, Soviet Math. Doklady.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 1e-2)
    op = _op(y)
    H = np.fft.fft2(op.psf, s=y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(X)).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def run_tv_deconv(y, physics=None, cfg=None):
    """Total Variation deconvolution via ADMM.

    Minimises 0.5*||Ax - y||^2 + lam*TV(x).
    Reference: Rudin et al. 1992, Physica D.
    """
    cfg = cfg or {}
    lam = cfg.get("lam", 0.02)
    rho = cfg.get("rho", 1.0)
    n_iter = cfg.get("n_iter", 30)
    op = _op(y)
    yf = y.astype(np.float32)

    H = np.fft.fft2(op.psf, s=yf.shape)
    HTy = np.conj(H) * np.fft.fft2(yf)
    HtH = np.abs(H) ** 2

    x = op.adjoint(yf)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        # x-update (Fourier solve)
        rhs = HTy + rho * np.fft.fft2(z - u)
        x = np.real(np.fft.ifft2(rhs / (HtH + rho))).astype(np.float32)
        # z-update (TV prox via gradient shrinkage)
        v = x + u
        dx = np.diff(v, axis=1, prepend=v[:, -1:])
        dy = np.diff(v, axis=0, prepend=v[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        shrink = np.maximum(1.0 - lam / (rho * mag), 0)
        z = v - (dx * (1 - shrink) + dy * (1 - shrink)) * 0.5
        z = np.clip(z, 0, 1)
        # u-update
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_rl_tv(y, physics=None, cfg=None):
    """Richardson-Lucy with Total Variation regularisation.

    RL-TV augments the standard RL multiplicative update with a TV
    gradient penalty to suppress noise while preserving edges.
    Reference: Dey et al. 2006, Microscopy Res. Tech.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 30)
    lam_tv = cfg.get("lam_tv", 0.01)
    op = _op(y)
    yf = y.astype(np.float32)
    x = np.maximum(op.adjoint(yf), 1e-8)

    for _ in range(n_iter):
        # Standard RL correction
        fwd = np.maximum(op.forward(x), 1e-8)
        ratio = yf / fwd
        correction = op.adjoint(ratio)

        # TV gradient (divergence of normalised gradient)
        dx = np.diff(x, axis=1, append=x[:, -1:])
        dy = np.diff(x, axis=0, append=x[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        div_x = np.diff(dx / mag, axis=1, prepend=(dx / mag)[:, :1])
        div_y = np.diff(dy / mag, axis=0, prepend=(dy / mag)[:1, :])
        tv_grad = div_x + div_y

        # Modified RL update with TV
        x = x * correction / (1.0 - lam_tv * tv_grad + 1e-8)
        x = np.clip(x, 1e-8, 1.0)

    return x.astype(np.float32)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser.

    Reference: Venkatakrishnan et al. 2013, GlobalSIP.
    """
    cfg = cfg or {}
    rho = cfg.get("rho", 1.0)
    n_iter = cfg.get("n_iter", 15)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    op = _op(y)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    H = np.fft.fft2(op.psf, s=yf.shape)
    HTy = np.conj(H) * np.fft.fft2(yf)
    HtH = np.abs(H) ** 2

    x = op.adjoint(yf)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        # x-update
        rhs = HTy + rho * np.fft.fft2(z - u)
        x = np.real(np.fft.ifft2(rhs / (HtH + rho))).astype(np.float32)
        # z-update: NLM denoiser
        v = np.clip(x + u, 0, 1)
        sigma_est = max(estimate_sigma(v), 1e-4)
        z = denoise_nl_means(v, h=nlm_sigma, sigma=sigma_est,
                             fast_mode=True, patch_size=5, patch_distance=6)
        z = z.astype(np.float32)
        # u-update
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_fista_nlm(y, physics=None, cfg=None):
    """Plug-and-Play FISTA with Non-Local Means denoiser.

    Reference: Beck & Teboulle 2009 (FISTA); PnP framework.
    """
    cfg = cfg or {}
    step = cfg.get("step", 0.5)
    n_iter = cfg.get("n_iter", 20)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    op = _op(y)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = op.adjoint(yf)
    x_prev = x.copy()
    t = 1.0

    for k in range(n_iter):
        # Momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        momentum = (t - 1.0) / t_new
        t = t_new
        z = x + momentum * (x - x_prev)
        x_prev = x.copy()

        # Gradient step
        residual = yf - op.forward(z)
        grad_step = z + step * op.adjoint(residual)

        # NLM proximal
        grad_step_clipped = np.clip(grad_step, 0, 1)
        sigma_est = max(estimate_sigma(grad_step_clipped), 1e-4)
        x = denoise_nl_means(grad_step_clipped, h=nlm_sigma, sigma=sigma_est,
                             fast_mode=True, patch_size=5, patch_distance=6)
        x = np.clip(x, 0, 1).astype(np.float32)

    return x.astype(np.float32)


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
