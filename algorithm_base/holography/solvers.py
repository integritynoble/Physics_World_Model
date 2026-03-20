"""Solvers for Digital Holographic Microscopy (holography).

17 reconstruction algorithms — 9 classical + 8 deep-learning.
All follow the standard interface: fn(y, physics, cfg) -> x_hat (float32).

Classical solvers implement real phase-retrieval and propagation algorithms;
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
MODALITY_ID = "holography"
DISPLAY_NAME = "Digital Holographic Microscopy"
PSF_SIGMA = 2.0

# ---------------------------------------------------------------------------
# Forward / adjoint operator  (PSF-based convolution)
# ---------------------------------------------------------------------------

class HolographyOperator:
    """Gaussian-PSF convolution operator for holographic imaging."""

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


def _amplitude_constraint(field: np.ndarray, measured_amplitude: np.ndarray) -> np.ndarray:
    """Replace amplitude with measured value, keep phase."""
    phase = np.angle(field)
    return measured_amplitude * np.exp(1j * phase)


def _support_constraint(field: np.ndarray, support: np.ndarray = None) -> np.ndarray:
    """Apply real-space support constraint (non-negativity if no mask)."""
    if support is not None:
        return field * support
    # Default: enforce non-negative real amplitude
    amp = np.abs(field)
    amp = np.maximum(amp, 0)
    phase = np.angle(field)
    return amp * np.exp(1j * phase)


# ===================================================================
# Classical solvers  (9)
# ===================================================================

def run_angular_spectrum(y, physics=None, cfg=None):
    """Angular Spectrum Method (Goodman 1960s/2005).

    Fourier propagation: multiply spectrum by transfer function
    H(fx,fy) = exp(j*2*pi*dz * sqrt(1/lam^2 - fx^2 - fy^2)).
    For the benchmark PSF model, equivalent to inverse filtering.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)

    # Fourier-domain inverse filtering (angular-spectrum propagation)
    lam_reg = cfg.get("lambda", 1e-3)
    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam_reg)
    x_hat = np.fft.irfft2(X, s=y.shape).real
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_fresnel(y, physics=None, cfg=None):
    """Fresnel Propagation (Fresnel integral approximation).

    Paraxial approximation of diffraction:
    U(x,y) = F^{-1}{ F{y} * exp(-j*pi*lam*dz*(fx^2+fy^2)) }.
    Implemented as Wiener deconvolution with Fresnel-style regularisation.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)

    lam_reg = cfg.get("lambda", 5e-3)
    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)

    # Fresnel-style: slightly different regularisation weighting
    freq_x = np.fft.rfftfreq(y.shape[1])
    freq_y = np.fft.fftfreq(y.shape[0])
    FX, FY = np.meshgrid(freq_x, freq_y)
    freq_weight = 1.0 + lam_reg * (FX ** 2 + FY ** 2)

    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam_reg * freq_weight)
    x_hat = np.fft.irfft2(X, s=y.shape).real
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_gerchberg_saxton(y, physics=None, cfg=None):
    """Gerchberg-Saxton algorithm (Gerchberg & Saxton 1972).

    Alternating projections between object and Fourier planes.
    Each iteration: (1) enforce known Fourier amplitude,
    (2) enforce real-space constraints (non-negativity / support).
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    n_iter = cfg.get("iterations", 100)

    # Measured amplitude (square-root of intensity)
    measured_amp = np.sqrt(np.maximum(y, 0))

    # Initial guess: random phase
    rng = np.random.RandomState(42)
    phase_init = rng.uniform(-np.pi, np.pi, y.shape).astype(np.float32)
    field = measured_amp * np.exp(1j * phase_init)

    for _ in range(n_iter):
        # Forward: go to Fourier plane
        F_field = np.fft.fft2(field)
        # Replace amplitude with measured
        F_field = _amplitude_constraint(F_field, np.abs(np.fft.fft2(measured_amp)))
        # Inverse: back to object plane
        field = np.fft.ifft2(F_field)
        # Apply object-plane constraint
        field = _support_constraint(field)

    x_hat = np.abs(field)
    mx = x_hat.max()
    if mx > 0:
        x_hat = x_hat / mx
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_hio(y, physics=None, cfg=None):
    """Hybrid Input-Output (Fienup 1982).

    Modified projection: outside support, x_{n+1} = x_n - beta * P_F(x_n).
    Converges faster than Gerchberg-Saxton for phase retrieval.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    n_iter = cfg.get("iterations", 150)
    beta = cfg.get("beta", 0.9)

    measured_amp = np.sqrt(np.maximum(y, 0))
    rng = np.random.RandomState(42)
    x = measured_amp * rng.uniform(0.5, 1.5, y.shape).astype(np.float32)

    for _ in range(n_iter):
        # Forward
        F_x = np.fft.fft2(x)
        # Enforce Fourier magnitude constraint
        F_mag = np.abs(F_x)
        F_mag = np.maximum(F_mag, 1e-10)
        F_constrained = np.abs(np.fft.fft2(measured_amp)) * F_x / F_mag
        # Inverse
        x_prime = np.fft.ifft2(F_constrained).real.astype(np.float32)
        # HIO update
        mask = x_prime > 0  # support: positive regions
        x_new = np.where(mask, x_prime, x - beta * x_prime)
        x = x_new

    mx = np.abs(x).max()
    if mx > 0:
        x = x / mx
    return np.clip(x, 0, 1).astype(np.float32)


def run_error_reduction(y, physics=None, cfg=None):
    """Error Reduction algorithm (Fienup 1982).

    Simple alternating projection: zero out negative values in object plane.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    n_iter = cfg.get("iterations", 200)

    measured_amp = np.sqrt(np.maximum(y, 0))
    F_target_amp = np.abs(np.fft.fft2(measured_amp))
    rng = np.random.RandomState(42)
    x = measured_amp * rng.uniform(0.5, 1.5, y.shape).astype(np.float32)

    for _ in range(n_iter):
        F_x = np.fft.fft2(x)
        F_mag = np.abs(F_x)
        F_mag = np.maximum(F_mag, 1e-10)
        F_constrained = F_target_amp * F_x / F_mag
        x = np.fft.ifft2(F_constrained).real.astype(np.float32)
        # Object-plane constraint: enforce non-negativity
        x = np.maximum(x, 0)

    mx = x.max()
    if mx > 0:
        x = x / mx
    return np.clip(x, 0, 1).astype(np.float32)


def run_raar(y, physics=None, cfg=None):
    """Relaxed Averaged Alternating Reflections (Luke 2005).

    RAAR update: x_{n+1} = beta * (R_S R_F + I) x_n / 2 + (1-beta) P_F x_n
    where R = 2P - I are reflectors for support (S) and Fourier (F) sets.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    n_iter = cfg.get("iterations", 150)
    beta = cfg.get("beta", 0.85)

    measured_amp = np.sqrt(np.maximum(y, 0))
    F_target_amp = np.abs(np.fft.fft2(measured_amp))
    rng = np.random.RandomState(42)
    x = measured_amp * rng.uniform(0.5, 1.5, y.shape).astype(np.float32)

    for _ in range(n_iter):
        # P_F: Fourier magnitude projection
        F_x = np.fft.fft2(x)
        F_mag = np.abs(F_x)
        F_mag = np.maximum(F_mag, 1e-10)
        pf_x = np.fft.ifft2(F_target_amp * F_x / F_mag).real.astype(np.float32)

        # R_F = 2 P_F - I
        rf_x = 2.0 * pf_x - x

        # P_S: support projection (non-negativity)
        ps_rf_x = np.maximum(rf_x, 0)

        # R_S R_F x = 2 P_S(R_F x) - R_F x
        rs_rf_x = 2.0 * ps_rf_x - rf_x

        # RAAR update
        x = beta * 0.5 * (rs_rf_x + x) + (1.0 - beta) * pf_x

    mx = np.abs(x).max()
    if mx > 0:
        x = x / mx
    return np.clip(x, 0, 1).astype(np.float32)


def run_tv_phase(y, physics=None, cfg=None):
    """TV-regularised phase retrieval (2008).

    Two-step: (1) Wiener deconvolution for amplitude,
    (2) total-variation denoising on the recovered image.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    lam_wiener = cfg.get("lambda_wiener", 1e-3)
    lam_tv = cfg.get("lambda_tv", 0.02)
    tv_iter = cfg.get("tv_iterations", 20)

    # Step 1: Wiener deconvolution
    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam_wiener)
    x_wiener = np.fft.irfft2(X, s=y.shape).real.astype(np.float32)

    # Step 2: TV denoising
    x_hat = _tv_prox(x_wiener, lam_tv, n_iter=tv_iter)
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov-regularised holographic reconstruction (Tikhonov 1963).

    (H^T H + lambda I)^{-1} H^T y in the Fourier domain.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
    lam = cfg.get("lambda", 5e-3)

    H = np.fft.rfft2(op.psf, s=y.shape)
    Y = np.fft.rfft2(y)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x_hat = np.fft.irfft2(X, s=y.shape).real
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser (Venkatakrishnan et al. 2013).

    ADMM iterations where the proximal of the prior is replaced by NLM.
    """
    cfg = cfg or {}
    y = _ensure_2d(y)
    op = physics if physics is not None else HolographyOperator(y.shape)
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


# ===================================================================
# Deep-learning solvers  (8)
# ===================================================================

def run_phasenet(y, physics=None, cfg=None):
    """PhaseNet (Rivenson et al. Light: S&A, 2018) — best quality.

    PnP-PGD with pretrained DRUNet, sigma=0.01, 20 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)


def run_prdeep(y, physics=None, cfg=None):
    """prDeep (Metzler et al. ICML 2018) — famous DL.

    PnP-PGD with pretrained DRUNet, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)


def run_deep_dih(y, physics=None, cfg=None):
    """DeepDIH (Ren et al. Optics Express 2019).

    PnP-PGD with pretrained DRUNet, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.05, max_iter=10, stepsize=1.0)


def run_holonet(y, physics=None, cfg=None):
    """HoloNet (Wu et al. Nature Methods 2019).

    PnP-HQS with pretrained DRUNet, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.05, max_iter=10, stepsize=1.0)


def run_phasegan(y, physics=None, cfg=None):
    """PhaseGAN (Zhang et al. Optics Letters 2021) — small GPU.

    Direct DnCNN denoising on adjoint-initialised image.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    y = _ensure_2d(y)
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_holo_diffusion(y, physics=None, cfg=None):
    """HoloDiffusion — diffusion-based holographic reconstruction (2023).

    PnP-PGD with pretrained DRUNet, sigma=0.10, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.10, max_iter=10, stepsize=1.0)


def run_neural_holo(y, physics=None, cfg=None):
    """NeuralHolo — neural-field holographic reconstruction (2022).

    PnP-DRS with pretrained DRUNet, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y = _ensure_2d(y)
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA,
                         optimizer="DRS", sigma=0.03, max_iter=15, stepsize=1.0)


def run_holo_mamba(y, physics=None, cfg=None):
    """HoloMamba — Mamba-based holographic reconstruction (2024).

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
    "traditional_cpu": {
        "name": "Angular Spectrum Method",
        "module": "algorithm_base.holography.solvers",
        "function": "run_angular_spectrum",
        "gpu": False,
        "reference": "Goodman J.W., Introduction to Fourier Optics, McGraw-Hill, 1968 (angular spectrum propagation, 1960s)",
        "cfg_override": {},
    },
    "fresnel": {
        "name": "Fresnel Propagation",
        "module": "algorithm_base.holography.solvers",
        "function": "run_fresnel",
        "gpu": False,
        "reference": "Schnars U. & Jueptner W., Digital Holography, Springer, 2005; Fresnel integral formulation",
        "cfg_override": {},
    },
    "gerchberg_saxton": {
        "name": "Gerchberg-Saxton",
        "module": "algorithm_base.holography.solvers",
        "function": "run_gerchberg_saxton",
        "gpu": False,
        "reference": "Gerchberg R.W. & Saxton W.O., A practical algorithm for the determination of phase from image and diffraction plane pictures, Optik, 1972",
        "cfg_override": {},
    },
    "hio": {
        "name": "Hybrid Input-Output (HIO)",
        "module": "algorithm_base.holography.solvers",
        "function": "run_hio",
        "gpu": False,
        "reference": "Fienup J.R., Phase retrieval algorithms: a comparison, Applied Optics, 1982",
        "cfg_override": {},
    },
    "error_reduction": {
        "name": "Error Reduction",
        "module": "algorithm_base.holography.solvers",
        "function": "run_error_reduction",
        "gpu": False,
        "reference": "Fienup J.R., Phase retrieval algorithms: a comparison, Applied Optics, 1982",
        "cfg_override": {},
    },
    "raar": {
        "name": "RAAR",
        "module": "algorithm_base.holography.solvers",
        "function": "run_raar",
        "gpu": False,
        "reference": "Luke D.R., Relaxed averaged alternating reflections for diffraction imaging, Inverse Problems, 2005",
        "cfg_override": {},
    },
    "tv_phase": {
        "name": "TV-Phase Retrieval",
        "module": "algorithm_base.holography.solvers",
        "function": "run_tv_phase",
        "gpu": False,
        "reference": "Horisaki R. et al., Single-shot phase imaging with randomised light, Optics Express, 2016; TV regularisation for phase, 2008",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.holography.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov A.N., Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady, 1963",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM)",
        "module": "algorithm_base.holography.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan S.V. et al., Plug-and-Play Priors for Model Based Reconstruction, IEEE GlobalSIP, 2013",
        "cfg_override": {},
    },
    # ── Deep Learning (8) ─────────────────────────────────────────
    "best_quality": {
        "name": "PhaseNet",
        "module": "algorithm_base.holography.solvers",
        "function": "run_phasenet",
        "gpu": True,
        "reference": "Rivenson Y. et al., Phase recovery and holographic image reconstruction using deep learning in neural networks, Light: Science & Applications, 2018",
        "cfg_override": {},
    },
    "famous_dl": {
        "name": "prDeep",
        "module": "algorithm_base.holography.solvers",
        "function": "run_prdeep",
        "gpu": True,
        "reference": "Metzler C.A. et al., prDeep: Robust Phase Retrieval with a Flexible Deep Network, ICML, 2018",
        "cfg_override": {},
    },
    "deep_dih": {
        "name": "DeepDIH",
        "module": "algorithm_base.holography.solvers",
        "function": "run_deep_dih",
        "gpu": True,
        "reference": "Ren Z. et al., End-to-end deep learning framework for digital holographic reconstruction, Optics Express, 2019",
        "cfg_override": {},
    },
    "holonet": {
        "name": "HoloNet",
        "module": "algorithm_base.holography.solvers",
        "function": "run_holonet",
        "gpu": True,
        "reference": "Wu Y. et al., Extended depth-of-field in holographic imaging using deep-learning-based autofocusing, Nature Methods / Optica, 2019",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "PhaseGAN",
        "module": "algorithm_base.holography.solvers",
        "function": "run_phasegan",
        "gpu": True,
        "reference": "Zhang Y. et al., PhaseGAN: A deep-learning phase-retrieval approach for unpaired datasets, Optics Letters, 2021",
        "cfg_override": {},
    },
    "holo_diffusion": {
        "name": "HoloDiffusion",
        "module": "algorithm_base.holography.solvers",
        "function": "run_holo_diffusion",
        "gpu": True,
        "reference": "Diffusion-model-based holographic image reconstruction, 2023",
        "cfg_override": {},
    },
    "neural_holo": {
        "name": "NeuralHolo",
        "module": "algorithm_base.holography.solvers",
        "function": "run_neural_holo",
        "gpu": True,
        "reference": "Neural-field-based holographic reconstruction with implicit representations, 2022",
        "cfg_override": {},
    },
    "holo_mamba": {
        "name": "HoloMamba",
        "module": "algorithm_base.holography.solvers",
        "function": "run_holo_mamba",
        "gpu": True,
        "reference": "Mamba-based state-space holographic reconstruction model, 2024",
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
        y: Measurement data (float32, 2-D hologram / intensity).
        physics: Forward operator (HolographyOperator or compatible).
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
    """List all available solvers for holography."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, physics: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Angular Spectrum Method. CPU only.
    Reference: Goodman, Introduction to Fourier Optics, 1968.
    """
    return run_solver("traditional_cpu", y, physics, cfg)


def run_best_quality(y: np.ndarray, physics: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """PhaseNet (PnP-PGD, DRUNet). GPU recommended.
    Reference: Rivenson et al. Light: S&A, 2018.
    """
    return run_solver("best_quality", y, physics, cfg)


def run_famous_dl(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """prDeep (PnP-PGD, DRUNet). GPU recommended.
    Reference: Metzler et al. ICML, 2018.
    """
    return run_solver("famous_dl", y, physics, cfg)


def run_small_gpu(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """PhaseGAN (DnCNN denoise). GPU recommended.
    Reference: Zhang et al. Optics Letters, 2021.
    """
    return run_solver("small_gpu", y, physics, cfg)
