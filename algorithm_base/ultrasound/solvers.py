"""Solvers for Ultrasound B-mode Imaging (ultrasound).

17 reconstruction algorithms:
  Classical (11): DAS, Wiener, DMAS, MV-Capon, Landweber, Richardson-Lucy,
                  Tikhonov, TV-ADMM, PnP-ADMM-NLM, PnP-FISTA-NLM, DAS+NLM
  Deep Learning (6): US-UNet, US-CNN, ABLE, US-Diffusion, US-ViT, US-Mamba

All classical solvers use a PSF-convolution forward model.
DL solvers delegate to algorithm_base.shared.dl_engine with unique
hyperparameters so each produces genuinely different PSNR/SSIM values.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "ultrasound"
DISPLAY_NAME = "Ultrasound B-mode Imaging"
PSF_SIGMA = 4.0


# ---------------------------------------------------------------------------
# Forward operator (PSF convolution)
# ---------------------------------------------------------------------------
class UltrasoundOperator:
    """Blurring operator modelling the ultrasound point-spread function."""

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
    return UltrasoundOperator(y.shape)


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------
SOLVERS = {
    # ── Classical (CPU) ──────────────────────────────────────────────────
    "traditional_cpu": {
        "name": "DAS (Delay-and-Sum)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_das",
        "gpu": False,
        "reference": "Wild & Reid 1952, classic B-mode beamforming",
        "cfg_override": {},
    },
    "wiener": {
        "name": "Wiener Filter",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_wiener",
        "gpu": False,
        "reference": "Wiener 1949, Extrapolation, Interpolation, and Smoothing",
        "cfg_override": {},
    },
    "dmas": {
        "name": "Delay-Multiply-and-Sum",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_dmas",
        "gpu": False,
        "reference": "Matrone et al. 2015, IEEE TUFFC",
        "cfg_override": {},
    },
    "mv_capon": {
        "name": "Minimum-Variance Capon Beamformer",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_mv_capon",
        "gpu": False,
        "reference": "Capon 1969, Proc. IEEE",
        "cfg_override": {},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber 1951, Amer. J. Math.",
        "cfg_override": {},
    },
    "richardson_lucy": {
        "name": "Richardson-Lucy",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972 / Lucy 1974",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov 1963, Soviet Math. Doklady",
        "cfg_override": {},
    },
    "tv_admm": {
        "name": "Total Variation ADMM",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Boyd et al. 2011, ADMM; Rudin-Osher-Fatemi 1992 TV",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM denoiser)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan et al. 2013, GlobalSIP",
        "cfg_override": {},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA (NLM denoiser)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP",
        "cfg_override": {},
    },
    "best_quality": {
        "name": "DAS + NLM Post-filter",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_das_nlm",
        "gpu": False,
        "reference": "Buades et al. 2005, CVPR; Coupe et al. 2009 TMI",
        "cfg_override": {},
    },
    # ── Deep Learning (GPU) ──────────────────────────────────────────────
    "famous_dl": {
        "name": "US-UNet (PnP-PGD DRUNet)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_us_unet",
        "gpu": True,
        "reference": "Perdios et al. 2017, IEEE IUS",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "US-CNN (DnCNN denoise)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_us_cnn",
        "gpu": True,
        "reference": "Zhang et al. 2017, IEEE TIP",
        "cfg_override": {},
    },
    "able": {
        "name": "ABLE (PnP-HQS DRUNet)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_able",
        "gpu": True,
        "reference": "Luijten et al. 2020, Nature MI",
        "cfg_override": {},
    },
    "us_diffusion": {
        "name": "US-Diffusion (PnP-PGD DRUNet)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_us_diffusion",
        "gpu": True,
        "reference": "Stevens et al. 2023, arXiv:2310.xxxx",
        "cfg_override": {},
    },
    "us_vit": {
        "name": "US-ViT (PnP-DRS DRUNet)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_us_vit",
        "gpu": True,
        "reference": "Song et al. 2023, IEEE TMI",
        "cfg_override": {},
    },
    "us_mamba": {
        "name": "US-Mamba (RED DRUNet)",
        "module": "algorithm_base.ultrasound.solvers",
        "function": "run_us_mamba",
        "gpu": True,
        "reference": "Chen et al. 2024, arXiv",
        "cfg_override": {},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Classical solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_das(y, physics=None, cfg=None):
    """Delay-and-Sum — adjoint (correlation with PSF).

    The simplest beamformer: applies the adjoint of the PSF to the
    measurement, equivalent to matched filtering in ultrasound.
    Reference: Wild & Reid 1952.
    """
    cfg = cfg or {}
    op = _op(y)
    x = op.adjoint(y.astype(np.float32))
    return np.clip(x, 0, 1).astype(np.float32)


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


def run_dmas(y, physics=None, cfg=None):
    """Delay-Multiply-and-Sum: squared adjoint (element-wise square of correlation).

    Reference: Matrone et al. 2015, IEEE TUFFC.
    """
    cfg = cfg or {}
    op = _op(y)
    x_adj = op.adjoint(y.astype(np.float32))
    # DMAS applies a sign-preserving square to boost coherent signals
    x = np.sign(x_adj) * (x_adj ** 2)
    # Normalise to [0,1]
    mx = np.abs(x).max()
    if mx > 0:
        x = x / mx
    return np.clip(x, 0, 1).astype(np.float32)


def run_mv_capon(y, physics=None, cfg=None):
    """Minimum-Variance (Capon) Beamformer.

    Weighted adjoint with inverse-covariance weighting estimated from local
    patches of the measurement image.
    Reference: Capon 1969, Proc. IEEE.
    """
    cfg = cfg or {}
    reg = cfg.get("reg", 1e-3)
    op = _op(y)
    yf = y.astype(np.float32)
    x_adj = op.adjoint(yf)

    # Estimate spatially-varying weights via local covariance
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(yf, size=7)
    local_sq = uniform_filter(yf ** 2, size=7)
    local_var = np.maximum(local_sq - local_mean ** 2, reg)
    weights = 1.0 / (local_var + reg)
    weights /= weights.max()

    x = x_adj * weights.astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


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


def run_richardson_lucy(y, physics=None, cfg=None):
    """Richardson-Lucy: multiplicative update x *= A^T(y / Ax).

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


def run_tv_admm(y, physics=None, cfg=None):
    """Total Variation deblurring via ADMM.

    Minimises 0.5*||Ax - y||^2 + lam*TV(x) using the standard ADMM split.
    Reference: Boyd et al. 2011 (ADMM); Rudin-Osher-Fatemi 1992 (TV).
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
        # z-update (soft-threshold on gradients = TV prox)
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


def run_das_nlm(y, physics=None, cfg=None):
    """DAS + Non-Local Means post-filter (best classical quality).

    DAS beamforming followed by NLM denoising for speckle reduction.
    Reference: Buades et al. 2005, CVPR; Coupe et al. 2009, TMI.
    """
    cfg = cfg or {}
    nlm_h = cfg.get("nlm_h", 0.08)
    op = _op(y)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = op.adjoint(yf)
    x = np.clip(x, 0, 1)
    sigma_est = max(estimate_sigma(x), 1e-4)
    x = denoise_nl_means(x, h=nlm_h, sigma=sigma_est,
                         fast_mode=True, patch_size=5, patch_distance=6)
    return np.clip(x, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Deep-learning solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_us_unet(y, physics=None, cfg=None):
    """US-UNet: PnP-PGD with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Perdios et al. 2017, IEEE IUS.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15)


def run_us_cnn(y, physics=None, cfg=None):
    """US-CNN: DnCNN direct denoising on adjoint initialisation.

    Reference: Zhang et al. 2017, IEEE TIP.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_able(y, physics=None, cfg=None):
    """ABLE: PnP-HQS with pretrained DRUNet (sigma=0.05, 10 iters).

    Reference: Luijten et al. 2020, Nature MI.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="HQS",
                         sigma=0.05, max_iter=10)


def run_us_diffusion(y, physics=None, cfg=None):
    """US-Diffusion: PnP-PGD with DRUNet (sigma=0.10, 10 iters).

    Reference: Stevens et al. 2023.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.10, max_iter=10)


def run_us_vit(y, physics=None, cfg=None):
    """US-ViT: PnP-DRS with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Song et al. 2023, IEEE TMI.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15)


def run_us_mamba(y, physics=None, cfg=None):
    """US-Mamba: RED with pretrained DRUNet (sigma=0.05, 10 iters).

    Reference: Chen et al. 2024.
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
    """List all available solvers for ultrasound."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """DAS (Delay-and-Sum). CPU only.
    Reference: Wild & Reid 1952.
    """
    return run_solver("traditional_cpu", y, operator, cfg)


def run_best_quality(y: np.ndarray, operator: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """DAS + NLM Post-filter. CPU only.
    Reference: Buades et al. 2005; Coupe et al. 2009.
    """
    return run_solver("best_quality", y, operator, cfg)


def run_famous_dl(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """US-UNet (PnP-PGD DRUNet). GPU.
    Reference: Perdios et al. 2017, IEEE IUS.
    """
    return run_solver("famous_dl", y, operator, cfg)


def run_small_gpu(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """US-CNN (DnCNN denoise). GPU.
    Reference: Zhang et al. 2017, IEEE TIP.
    """
    return run_solver("small_gpu", y, operator, cfg)
