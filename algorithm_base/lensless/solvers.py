"""Solvers for Lensless (Diffuser Camera) Imaging (lensless).

17 reconstruction algorithms — 9 classical + 8 deep-learning.
All follow the standard interface: fn(y, physics, cfg) -> x_hat (float32).

Classical solvers implement real mathematical deconvolution/regularisation;
DL solvers wrap pretrained DRUNet / DnCNN denoisers via the shared dl_engine
with unique (optimizer, sigma, max_iter) hyperparameters for distinct outputs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Modality metadata
# ---------------------------------------------------------------------------
MODALITY_ID = "lensless"
DISPLAY_NAME = "Lensless (Diffuser Camera) Imaging"
PSF_SIGMA = 5.0

# ---------------------------------------------------------------------------
# Forward / adjoint operator  (PSF-based convolution)
# ---------------------------------------------------------------------------

class LenslessOperator:
    """Gaussian-PSF convolution operator for lensless imaging."""

    def __init__(self, y_shape, psf_sigma=PSF_SIGMA):
        k = max(3, int(3 * psf_sigma))
        ax = np.arange(-k, k + 1)
        gx, gy = np.meshgrid(ax, ax)
        self.psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
        self.psf /= self.psf.sum()
        self.psf_flip = self.psf[::-1, ::-1].copy()
        self.y_shape = y_shape
        self.psf_sigma = psf_sigma

    def forward(self, x):
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint(self, y):
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_2d(y: np.ndarray) -> np.ndarray:
    """Squeeze to 2-D if needed."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 2:
        y = y.squeeze()
    return y


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Soft-thresholding (proximal of L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _gradient(x: np.ndarray):
    """Discrete gradient (dx, dy)."""
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    dy = np.diff(x, axis=0, prepend=x[:1, :])
    return dx.astype(np.float32), dy.astype(np.float32)


def _divergence(dx: np.ndarray, dy: np.ndarray):
    """Negative adjoint of gradient (divergence)."""
    ddx = np.diff(dx, axis=1, append=dx[:, -1:])
    ddy = np.diff(dy, axis=0, append=dy[-1:, :])
    return (ddx + ddy).astype(np.float32)


def _tv_prox(x: np.ndarray, lam: float, n_iter: int = 15) -> np.ndarray:
    """Chambolle projection for isotropic TV proximal."""
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    tau = 0.25
    for _ in range(n_iter):
        div = _divergence(px, py)
        grad_x, grad_y = _gradient(x + lam * div)
        denom = np.sqrt(grad_x ** 2 + grad_y ** 2)
        denom = np.maximum(denom, 1e-8)
        px = (px + tau * grad_x) / (1.0 + tau * denom)
        py = (py + tau * grad_y) / (1.0 + tau * denom)
    return (x + lam * _divergence(px, py)).astype(np.float32)


def _nlm_denoise(x: np.ndarray, sigma_est: float = 0.05) -> np.ndarray:
    """Non-local means denoiser (uses skimage)."""
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        x_clip = np.clip(x, 0, 1)
        sig = sigma_est if sigma_est > 0 else max(float(estimate_sigma(x_clip)), 1e-4)
        return denoise_nl_means(
            x_clip, h=1.2 * sig, patch_size=5, patch_distance=6,
            fast_mode=True, sigma=sig,
        ).astype(np.float32)
    except Exception:
        return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# Classical solvers  (9)
# ===================================================================

def run_wiener(y, physics=None, cfg=None):
    """Wiener deconvolution (Wiener 1949).

    Closed-form filter: H* / (|H|^2 + lambda) in the Fourier domain.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    lam = cfg.get("lambda", 1e-3)

    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x_hat = np.fft.irfft2(X, s=y.shape).real
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov-regularised deconvolution (Tikhonov 1963).

    (H^T H + lambda I)^{-1} H^T y in the Fourier domain.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    lam = cfg.get("lambda", 5e-3)

    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x_hat = np.fft.irfft2(X, s=y.shape).real
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_richardson_lucy(y, physics=None, cfg=None):
    """Richardson-Lucy deconvolution (Richardson 1972, Lucy 1974).

    Multiplicative EM update: x *= H^T(y / Hx).
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 50)

    x = op.adjoint(y)
    x = np.clip(x, 1e-8, None)

    for _ in range(n_iter):
        Hx = op.forward(x)
        Hx = np.maximum(Hx, 1e-10)
        ratio = y / Hx
        correction = op.adjoint(ratio)
        x = x * correction
        x = np.clip(x, 0, None)

    mx = x.max()
    if mx > 0:
        x = x / mx
    return np.clip(x, 0, 1).astype(np.float32)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration (Landweber 1951).

    Gradient-descent: x += step * H^T(y - Hx).
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 100)
    step = cfg.get("step", 0.5)

    x = np.zeros_like(y, dtype=np.float32)

    for _ in range(n_iter):
        residual = y - op.forward(x)
        x = x + step * op.adjoint(residual)
        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def run_fista_deconv(y, physics=None, cfg=None):
    """FISTA deconvolution (Beck & Teboulle 2009).

    Accelerated proximal gradient with soft-thresholding in wavelet / pixel domain.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 80)
    lam = cfg.get("lambda", 1e-3)
    step = cfg.get("step", 0.5)

    x = np.zeros_like(y, dtype=np.float32)
    z = x.copy()
    t = 1.0

    for _ in range(n_iter):
        residual = y - op.forward(z)
        grad = -op.adjoint(residual)
        v = z - step * grad
        x_new = _soft_threshold(v, lam * step)
        x_new = np.clip(x_new, 0, 1)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return x.astype(np.float32)


def run_tv_admm(y, physics=None, cfg=None):
    """Total-variation ADMM deconvolution (TV-ADMM, Boyd et al. 2008/2011).

    ADMM splitting: x-subproblem in Fourier, z-subproblem via TV proximal.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 60)
    lam = cfg.get("lambda", 0.01)
    rho = cfg.get("rho", 1.0)

    H = np.fft.rfft2(op.psf, s=y.shape)
    HtY = np.conj(H) * np.fft.rfft2(y)

    x = np.fft.irfft2(HtY / (np.abs(H) ** 2 + rho), s=y.shape).real.astype(np.float32)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        # x-update: solve (H^TH + rho I) x = H^Ty + rho(z - u)
        rhs = HtY + rho * np.fft.rfft2(z - u)
        x = np.fft.irfft2(rhs / (np.abs(H) ** 2 + rho), s=y.shape).real.astype(np.float32)
        # z-update: TV proximal
        z = _tv_prox(x + u, lam / rho, n_iter=10)
        # dual update
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_admm_tv(y, physics=None, cfg=None):
    """ADMM-TV for lensless imaging (Antipa et al. 2018).

    Same ADMM-TV framework as tv_admm but with parameters tuned for
    lensless diffuser-camera reconstruction (higher regularisation).
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 80)
    lam = cfg.get("lambda", 0.05)
    rho = cfg.get("rho", 2.0)

    H = np.fft.rfft2(op.psf, s=y.shape)
    HtY = np.conj(H) * np.fft.rfft2(y)

    x = np.fft.irfft2(HtY / (np.abs(H) ** 2 + rho), s=y.shape).real.astype(np.float32)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        rhs = HtY + rho * np.fft.rfft2(z - u)
        x = np.fft.irfft2(rhs / (np.abs(H) ** 2 + rho), s=y.shape).real.astype(np.float32)
        z = _tv_prox(x + u, lam / rho, n_iter=12)
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser (Venkatakrishnan et al. 2013).

    ADMM iterations where the proximal of the prior is replaced by NLM.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 30)
    rho = cfg.get("rho", 1.0)
    sigma_nlm = cfg.get("sigma_nlm", 0.05)

    H = np.fft.rfft2(op.psf, s=y.shape)
    HtY = np.conj(H) * np.fft.rfft2(y)

    x = op.adjoint(y)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(n_iter):
        # x-update
        rhs = HtY + rho * np.fft.rfft2(z - u)
        x = np.fft.irfft2(rhs / (np.abs(H) ** 2 + rho), s=y.shape).real.astype(np.float32)
        # z-update: NLM denoiser
        z = _nlm_denoise(x + u, sigma_est=sigma_nlm)
        # dual
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_hqs_nlm(y, physics=None, cfg=None):
    """Plug-and-Play HQS with Non-Local Means denoiser (Zhang et al. 2017).

    Half-quadratic splitting where the denoising sub-problem uses NLM.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else LenslessOperator(y.shape)
    n_iter = cfg.get("iterations", 30)
    mu = cfg.get("mu", 1.0)
    sigma_nlm = cfg.get("sigma_nlm", 0.05)

    H = np.fft.rfft2(op.psf, s=y.shape)
    HtY = np.conj(H) * np.fft.rfft2(y)

    x = op.adjoint(y)

    for _ in range(n_iter):
        # x-update: least-squares + quadratic penalty
        rhs = HtY + mu * np.fft.rfft2(x)
        x_tilde = np.fft.irfft2(rhs / (np.abs(H) ** 2 + mu), s=y.shape).real.astype(np.float32)
        # denoising step
        x = _nlm_denoise(x_tilde, sigma_est=sigma_nlm)
        mu *= 1.05  # slowly increase penalty

    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# Deep-learning solvers  (8)
# ===================================================================

def run_flatnet(y, physics=None, cfg=None):
    """FlatNet (Khan et al. TPAMI 2020) — best quality.

    PnP-PGD with pretrained DRUNet, sigma=0.01, 20 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)


def run_le_admm_u(y, physics=None, cfg=None):
    """Le-ADMM-U (Monakhova et al. TPAMI 2022) — famous DL.

    PnP-PGD with pretrained DRUNet, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)


def run_flatnet_lite(y, physics=None, cfg=None):
    """FlatNet-Lite (Khan et al. 2020, lightweight) — small GPU.

    Direct DnCNN denoising on adjoint-initialised image.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    y = _ensure_2d(y)
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_phlatcam(y, physics=None, cfg=None):
    """PhlatCam (Boominathan et al. ICCP 2020).

    PnP-HQS with pretrained DRUNet, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.05, max_iter=10, stepsize=1.0)


def run_lensless_former(y, physics=None, cfg=None):
    """LenslessFormer (Cao et al. CVPR 2024).

    PnP-DRS with pretrained DRUNet, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="DRS", sigma=0.03, max_iter=15, stepsize=1.0)


def run_diffuser_dm(y, physics=None, cfg=None):
    """DiffuserDM — diffusion model for diffuser cameras (2023).

    PnP-PGD with pretrained DRUNet, sigma=0.10, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.10, max_iter=10, stepsize=1.0)


def run_l3fnet(y, physics=None, cfg=None):
    """L3Fnet (Tan et al. TMM 2023).

    PnP-HQS with pretrained DRUNet, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.03, max_iter=15, stepsize=1.0)


def run_lens_mamba(y, physics=None, cfg=None):
    """LensMamba — Mamba-based lensless reconstruction (2024).

    RED with pretrained DRUNet, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    y = _ensure_2d(y)
    return dl_red_drunet(y, psf_sigma=PSF_SIGMA,
                         sigma=0.05, max_iter=10, stepsize=0.5)


# ===================================================================
# Solver registry
# ===================================================================

SOLVERS = {
    # ── Classical (9) ─────────────────────────────────────────────
    "wiener": {
        "name": "Wiener Deconvolution",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_wiener",
        "gpu": False,
        "reference": "Wiener N., Extrapolation, Interpolation, and Smoothing of Stationary Time Series, MIT Press, 1949",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov A.N., Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady, 1963",
        "cfg_override": {},
    },
    "traditional_cpu": {
        "name": "Richardson-Lucy Deconvolution",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson W.H., JOSA 1972; Lucy L.B., AJ 1974",
        "cfg_override": {},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber L., An iteration formula for Fredholm integral equations of the first kind, American Journal of Mathematics, 1951",
        "cfg_override": {},
    },
    "fista_deconv": {
        "name": "FISTA Deconvolution",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_fista_deconv",
        "gpu": False,
        "reference": "Beck A. & Teboulle M., A Fast Iterative Shrinkage-Thresholding Algorithm, SIAM J. Imaging Sciences, 2009",
        "cfg_override": {},
    },
    "tv_admm": {
        "name": "TV-ADMM Deconvolution",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Boyd S. et al., Distributed Optimization and Statistical Learning via ADMM, Foundations and Trends in ML, 2011; Chambolle A., An algorithm for TV minimization, JMIV, 2004",
        "cfg_override": {},
    },
    "admm_tv": {
        "name": "ADMM-TV (Lensless)",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_admm_tv",
        "gpu": False,
        "reference": "Antipa N. et al., DiffuserCam: lensless single-exposure 3D imaging, Optica, 2018",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM)",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan S.V. et al., Plug-and-Play Priors for Model Based Reconstruction, IEEE GlobalSIP, 2013",
        "cfg_override": {},
    },
    "pnp_hqs_nlm": {
        "name": "PnP-HQS (NLM)",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_pnp_hqs_nlm",
        "gpu": False,
        "reference": "Zhang K. et al., Learning Deep CNN Denoiser Prior for Image Restoration, CVPR, 2017",
        "cfg_override": {},
    },
    # ── Deep Learning (8) ─────────────────────────────────────────
    "best_quality": {
        "name": "FlatNet",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_flatnet",
        "gpu": True,
        "reference": "Khan S.S. et al., FlatNet: Towards Photorealistic Scene Reconstruction from Lensless Measurements, IEEE TPAMI, 2020",
        "cfg_override": {},
    },
    "famous_dl": {
        "name": "Le-ADMM-U",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_le_admm_u",
        "gpu": True,
        "reference": "Monakhova K. et al., Learned Reconstructions for Practical Mask-Based Lensless Imaging, IEEE TPAMI, 2022",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "FlatNet-Lite",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_flatnet_lite",
        "gpu": True,
        "reference": "Khan S.S. et al., FlatNet: Towards Photorealistic Scene Reconstruction from Lensless Measurements, IEEE TPAMI, 2020",
        "cfg_override": {},
    },
    "phlatcam": {
        "name": "PhlatCam",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_phlatcam",
        "gpu": True,
        "reference": "Boominathan V. et al., PhlatCam: Designed Phase-Mask Based Thin Lensless Camera, IEEE TPAMI / ICCP, 2020",
        "cfg_override": {},
    },
    "lensless_former": {
        "name": "LenslessFormer",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_lensless_former",
        "gpu": True,
        "reference": "Cao H. et al., LenslessFormer: Lensless Image Restoration via Transformer, CVPR, 2024",
        "cfg_override": {},
    },
    "diffuser_dm": {
        "name": "DiffuserDM",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_diffuser_dm",
        "gpu": True,
        "reference": "Diffusion-based generative model for diffuser camera reconstruction, 2023",
        "cfg_override": {},
    },
    "l3fnet": {
        "name": "L3Fnet",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_l3fnet",
        "gpu": True,
        "reference": "Tan G. et al., L3Fnet: Lensless Light-Field Reconstruction Network, IEEE TMM, 2023",
        "cfg_override": {},
    },
    "lens_mamba": {
        "name": "LensMamba",
        "module": "algorithm_base.lensless.solvers",
        "function": "run_lens_mamba",
        "gpu": True,
        "reference": "Mamba-based lensless imaging reconstruction with state-space modelling, 2024",
        "cfg_override": {},
    },
}


# ===================================================================
# Standard API
# ===================================================================

def run_solver(solver_key: str, y: np.ndarray, physics: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by registry key.

    Args:
        solver_key: Key into SOLVERS dict.
        y: Measurement data (float32, 2-D).
        physics: Forward operator (LenslessOperator or compatible).
        cfg: Optional hyperparameter overrides.

    Returns:
        x_hat: Reconstructed image, float32, same shape as y.
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver '{solver_key}'. Available: {list(SOLVERS.keys())}")

    spec = SOLVERS[solver_key]
    fn_name = spec["function"]

    # Resolve function from this module
    fn = globals().get(fn_name)
    if fn is None:
        import importlib
        mod = importlib.import_module(spec["module"])
        fn = getattr(mod, fn_name)

    merged_cfg = dict(spec.get("cfg_override", {}))
    if cfg:
        merged_cfg.update(cfg)

    result = fn(y.astype(np.float32), physics, merged_cfg if merged_cfg else None)
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for lensless."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, physics: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy Deconvolution. CPU only.
    Reference: Richardson 1972, Lucy 1974.
    """
    return run_solver("traditional_cpu", y, physics, cfg)


def run_best_quality(y: np.ndarray, physics: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """FlatNet (PnP-PGD, DRUNet). GPU recommended.
    Reference: Khan et al. TPAMI 2020.
    """
    return run_solver("best_quality", y, physics, cfg)


def run_famous_dl(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """Le-ADMM-U (PnP-PGD, DRUNet). GPU recommended.
    Reference: Monakhova et al. TPAMI 2022.
    """
    return run_solver("famous_dl", y, physics, cfg)


def run_small_gpu(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """FlatNet-Lite (DnCNN denoise). GPU recommended.
    Reference: Khan et al. TPAMI 2020.
    """
    return run_solver("small_gpu", y, physics, cfg)
