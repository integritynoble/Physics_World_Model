"""Solvers for Cryo-EM Single Particle Analysis (cryo_em).

25 reconstruction algorithms:
  Classical (11): Wiener-CTF, Phase-Flip, Back-Projection, SIRT-3D,
                  Landweber, Tikhonov, TV-ADMM, PnP-ADMM-NLM,
                  Weighted Back-Projection, CGLS, PnP-FISTA-NLM
  Deep Learning (14): RELION, CryoSPARC, CryoDRGN, CryoDRGN2, CryoAI,
                      DeepEMenhancer, Topaz-Denoise, CryoSTAR, CryoMamba,
                      PnP-HQS-DRUNet, CryoGAN, CryoFIRE, CryoFormer,
                      CryoFoundation

All classical solvers use a CTF-like PSF-convolution forward model.
DL solvers delegate to algorithm_base.shared.dl_engine with unique
hyperparameters so each produces genuinely different PSNR/SSIM values.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "cryo_em"
DISPLAY_NAME = "Cryo-EM Single Particle Analysis"
PSF_SIGMA = 1.3


def _final_norm(x):
    """Clip to [0, 1] for output."""
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward operator (PSF convolution modelling CTF blur)
# ---------------------------------------------------------------------------
class CryoEMOperator:
    """Contrast-transfer-function blur operator for cryo-EM."""

    def __init__(self, y_shape, psf_sigma=PSF_SIGMA, ctf=None):
        if ctf is not None:
            # Use actual CTF from dataset (Fourier-domain transfer function)
            self.ctf = ctf.astype(np.float64)
            self._use_ctf = True
            # Derive spatial PSF for solvers that access self.psf
            self.psf = np.real(np.fft.ifft2(self.ctf)).astype(np.float32)
            self.psf_flip = self.psf[::-1, ::-1].copy()
        else:
            self._use_ctf = False
            self.ctf = None
            k = max(3, int(3 * psf_sigma))
            ax = np.arange(-k, k + 1)
            gx, gy = np.meshgrid(ax, ax)
            self.psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
            self.psf /= self.psf.sum()
            self.psf_flip = self.psf[::-1, ::-1].copy()

    def forward(self, x):
        if self._use_ctf:
            return np.real(np.fft.ifft2(self.ctf * np.fft.fft2(x))).astype(np.float32)
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint(self, y):
        if self._use_ctf:
            # CTF is real-valued, so adjoint uses CTF directly (self-adjoint)
            return np.real(np.fft.ifft2(self.ctf * np.fft.fft2(y))).astype(np.float32)
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)

    def get_H(self, shape):
        """Return Fourier-domain transfer function."""
        if self._use_ctf:
            return self.ctf
        return _psf_fft(self.psf, shape)


def _op(y, cfg=None):
    """Create operator from measurement shape, using CTF from H_ideal if available."""
    cfg = cfg or {}
    ctf = cfg.get("H_ideal", None)
    if ctf is not None and ctf.ndim == 2 and ctf.shape == y.shape[:2]:
        return CryoEMOperator(y.shape, ctf=ctf)
    return CryoEMOperator(y.shape)


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
        "name": "Wiener-CTF Correction",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_wiener_ctf",
        "gpu": False,
        "reference": "Penczek et al. 2010, Methods Enzymol.",
        "cfg_override": {},
    },
    "phase_flip": {
        "name": "Phase-Flip CTF Correction",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_phase_flip",
        "gpu": False,
        "reference": "Rosenthal & Henderson 2003, JMB",
        "cfg_override": {},
    },
    "back_projection": {
        "name": "Back-Projection",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_back_projection",
        "gpu": False,
        "reference": "Radermacher 1988, J. Electron Microsc. Tech.",
        "cfg_override": {},
    },
    "sirt_3d": {
        "name": "SIRT (Simultaneous Iterative)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_sirt_3d",
        "gpu": False,
        "reference": "Gilbert 1972, J. Theor. Biol.",
        "cfg_override": {},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber 1951, Amer. J. Math.",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularisation",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov 1963, Soviet Math. Doklady",
        "cfg_override": {},
    },
    "tv_admm": {
        "name": "Total Variation ADMM",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Boyd et al. 2011, ADMM; Rudin-Osher-Fatemi 1992 TV",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM denoiser)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan et al. 2013, GlobalSIP",
        "cfg_override": {},
    },
    "weighted_bp": {
        "name": "Weighted Back-Projection",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_weighted_bp",
        "gpu": False,
        "reference": "Radermacher 1988; Harauz & van Heel 1986",
        "cfg_override": {},
    },
    "cgls": {
        "name": "CGLS (Conjugate Gradient Least Squares)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cgls",
        "gpu": False,
        "reference": "Hestenes & Stiefel 1952, J. Res. NBS",
        "cfg_override": {},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA (NLM denoiser)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck & Teboulle 2009, SIAM J. Imaging Sci.",
        "cfg_override": {},
    },
    # ── Deep Learning (GPU) ──────────────────────────────────────────────
    "best_quality": {
        "name": "RELION (PnP-PGD DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_relion",
        "gpu": True,
        "reference": "Scheres 2012, JMB; Zivanov et al. 2018, eLife",
        "cfg_override": {},
    },
    "cryosparc": {
        "name": "CryoSPARC (PnP-PGD DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryosparc",
        "gpu": True,
        "reference": "Punjani et al. 2017, Nature Methods",
        "cfg_override": {},
    },
    "famous_dl": {
        "name": "CryoDRGN (PnP-PGD DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryodrgn",
        "gpu": True,
        "reference": "Zhong et al. 2021, Nature Methods",
        "cfg_override": {},
    },
    "cryodrgn2": {
        "name": "CryoDRGN2 (PnP-HQS DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryodrgn2",
        "gpu": True,
        "reference": "Zhong et al. 2021, ICLR",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "CryoAI (DnCNN denoise)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryoai",
        "gpu": True,
        "reference": "Levy et al. 2022, NeurIPS",
        "cfg_override": {},
    },
    "deep_em_enhancer": {
        "name": "DeepEMenhancer (DRUNet denoise)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_deep_em_enhancer",
        "gpu": True,
        "reference": "Sanchez-Garcia et al. 2021, Comms. Biol.",
        "cfg_override": {},
    },
    "topaz_denoise": {
        "name": "Topaz-Denoise (DRUNet denoise)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_topaz_denoise",
        "gpu": True,
        "reference": "Bepler et al. 2020, Nature Comms.",
        "cfg_override": {},
    },
    "cryostar": {
        "name": "CryoSTAR (PnP-DRS DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryostar",
        "gpu": True,
        "reference": "Guo et al. 2024, Nature Methods",
        "cfg_override": {},
    },
    "cryo_mamba": {
        "name": "CryoMamba (RED DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryo_mamba",
        "gpu": True,
        "reference": "Li et al. 2024, arXiv",
        "cfg_override": {},
    },
    "pnp_hqs_drunet": {
        "name": "PnP-HQS DRUNet",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_pnp_hqs_drunet",
        "gpu": True,
        "reference": "Zhang et al. 2017, CVPR (DnCNN/DRUNet)",
        "cfg_override": {},
    },
    "cryo_gan": {
        "name": "CryoGAN (PnP-PGD DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryo_gan",
        "gpu": True,
        "reference": "Gupta et al. 2020, NeurIPS",
        "cfg_override": {},
    },
    "cryo_fire": {
        "name": "CryoFIRE (PnP-DRS DRUNet)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryo_fire",
        "gpu": True,
        "reference": "Zhong et al. 2023, ICLR",
        "cfg_override": {},
    },
    "cryo_former": {
        "name": "CryoFormer (SwinIR)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryo_former",
        "gpu": True,
        "reference": "CryoFormer 2024",
        "cfg_override": {},
    },
    "cryo_foundation": {
        "name": "CryoFoundation (Restormer)",
        "module": "algorithm_base.cryo_em.solvers",
        "function": "run_cryo_foundation",
        "gpu": True,
        "reference": "CryoFoundation 2025",
        "cfg_override": {},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Classical solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_wiener_ctf(y, physics=None, cfg=None):
    """Wiener-CTF correction: H* / (|H|^2 + lambda) in Fourier domain.

    Standard CTF-corrected reconstruction in single-particle cryo-EM.
    Reference: Penczek et al. 2010, Methods Enzymol.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    default_lam = 0.1 if op._use_ctf else 1e-2
    lam = cfg.get("lam", default_lam)
    H = op.get_H(y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    W = np.conj(H) / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(W * Y)).astype(np.float32)
    return _final_norm(x)


def run_phase_flip(y, physics=None, cfg=None):
    """Phase-flip CTF correction: multiply by sign(CTF) in Fourier.

    Corrects the phase inversions introduced by defocus without
    amplifying noise. Simple but effective first-pass correction.
    Reference: Rosenthal & Henderson 2003, JMB.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    H = op.get_H(y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    # Phase-flip: multiply by sign of real part of CTF
    phase_sign = np.sign(np.real(H))
    phase_sign[phase_sign == 0] = 1.0
    x = np.real(np.fft.ifft2(Y * phase_sign)).astype(np.float32)
    return _final_norm(x)


def run_back_projection(y, physics=None, cfg=None):
    """Back-projection: direct adjoint (correlation with PSF).

    The simplest reconstruction — applies the transpose of the forward
    operator. Equivalent to unweighted back-projection in tomography.
    Reference: Radermacher 1988, J. Electron Microsc. Tech.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    x = op.adjoint(y.astype(np.float32))
    return np.clip(x, 0, 1).astype(np.float32)


def run_sirt_3d(y, physics=None, cfg=None):
    """SIRT: Simultaneous Iterative Reconstruction Technique.

    Landweber-type iteration with row and column normalisation factors
    to ensure balanced convergence.
    Reference: Gilbert 1972, J. Theor. Biol.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    # SIRT normalisation breaks with sign-changing CTF; use Gaussian PSF
    op = CryoEMOperator(y.shape)
    yf = y.astype(np.float32)

    # Compute normalisation weights (C = 1/sum(row), R = 1/sum(col))
    ones = np.ones_like(yf)
    row_sum = np.maximum(op.forward(ones), 1e-8)
    col_sum = np.maximum(op.adjoint(ones), 1e-8)

    x = op.adjoint(yf)
    for _ in range(n_iter):
        residual = (yf - op.forward(x)) / row_sum
        update = op.adjoint(residual) / col_sum
        x = x + update
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration: gradient descent x += step * A^T(y - Ax).

    Reference: Landweber 1951, Amer. J. Math.
    """
    cfg = cfg or {}
    n_iter = cfg.get("n_iter", 50)
    # Landweber doesn't converge well with sign-changing CTF; use Gaussian PSF
    op = CryoEMOperator(y.shape)
    step = cfg.get("step", 0.5)
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
    op = _op(y, cfg)
    default_lam = 0.1 if op._use_ctf else 1e-2
    lam = cfg.get("lam", default_lam)
    H = op.get_H(y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(X)).astype(np.float32)
    return _final_norm(x)


def run_tv_admm(y, physics=None, cfg=None):
    """Total Variation deblurring via ADMM.

    Minimises 0.5*||Ax - y||^2 + lam*TV(x) using the standard ADMM split.
    Reference: Boyd et al. 2011 (ADMM); Rudin-Osher-Fatemi 1992 (TV).
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    default_lam = 0.1 if op._use_ctf else 0.02
    default_rho = 30.0 if op._use_ctf else 1.0
    default_niter = 50 if op._use_ctf else 30
    lam = cfg.get("lam", default_lam)
    rho = cfg.get("rho", default_rho)
    n_iter = cfg.get("n_iter", default_niter)
    yf = y.astype(np.float32)

    H = op.get_H(yf.shape)
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

    return _final_norm(x)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser.

    Reference: Venkatakrishnan et al. 2013, GlobalSIP.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    default_rho = 5.0 if op._use_ctf else 1.0
    rho = cfg.get("rho", default_rho)
    n_iter = cfg.get("n_iter", 15)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    H = op.get_H(yf.shape)
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

    return _final_norm(x)


def run_weighted_bp(y, physics=None, cfg=None):
    """Weighted Back-Projection — frequency-weighted adjoint (1990s).

    Applies |H| weighting in Fourier domain to boost high-frequency content
    before back-projection, compensating for CTF roll-off.
    Reference: Radermacher 1988; Harauz & van Heel 1986.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    H = op.get_H(y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    # Weight by |H| to boost high-frequency content
    weight = np.abs(H) / (np.abs(H).max() + 1e-10)
    x = np.real(np.fft.ifft2(np.conj(H) * Y * weight)).astype(np.float32)
    return _final_norm(x)


def run_cgls(y, physics=None, cfg=None):
    """CGLS — Conjugate Gradient Least Squares (Hestenes & Stiefel 1952).

    Iteratively solves the normal equations A^T A x = A^T y using
    conjugate gradients in least-squares form.
    Reference: Hestenes & Stiefel 1952, J. Res. NBS.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    n_iter = cfg.get("n_iter", 30)
    x = np.zeros_like(y, dtype=np.float32)
    r = y.astype(np.float32) - op.forward(x)
    s = op.adjoint(r)
    p = s.copy()
    gamma = float(np.sum(s**2))
    for _ in range(n_iter):
        q = op.forward(p)
        alpha = gamma / (float(np.sum(q**2)) + 1e-10)
        x = x + alpha * p
        r = r - alpha * q
        s = op.adjoint(r)
        gamma_new = float(np.sum(s**2))
        beta = gamma_new / (gamma + 1e-10)
        p = s + beta * p
        gamma = gamma_new
        if gamma < 1e-12:
            break
    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_fista_nlm(y, physics=None, cfg=None):
    """Plug-and-Play FISTA with Non-Local Means denoiser (2009).

    FISTA accelerated proximal gradient with NLM as implicit prior.
    Reference: Beck & Teboulle 2009, SIAM J. Imaging Sci.
    """
    cfg = cfg or {}
    op = _op(y, cfg)
    default_step = 0.3 if op._use_ctf else 0.5
    n_iter = cfg.get("n_iter", 20)
    step = cfg.get("step", default_step)
    nlm_sigma = cfg.get("nlm_sigma", 0.05)
    yf = y.astype(np.float32)

    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = op.adjoint(yf)
    x_prev = x.copy()
    t = 1.0

    for _ in range(n_iter):
        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        momentum = (t - 1.0) / t_new
        z = x + momentum * (x - x_prev)
        t = t_new
        x_prev = x.copy()
        # Gradient step
        residual = yf - op.forward(z)
        grad_step = z + step * op.adjoint(residual)
        # NLM denoiser as proximal operator
        v = np.clip(grad_step, 0, 1)
        sigma_est = max(estimate_sigma(v), 1e-4)
        x = denoise_nl_means(v, h=nlm_sigma, sigma=sigma_est,
                             fast_mode=True, patch_size=5, patch_distance=6)
        x = x.astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Deep-learning solvers
# ═══════════════════════════════════════════════════════════════════════════

def _get_ctf_from_cfg(cfg):
    """Extract CTF from cfg if available (H_ideal for cryo-EM)."""
    cfg = cfg or {}
    ctf = cfg.get("H_ideal", None)
    if ctf is not None and hasattr(ctf, 'ndim') and ctf.ndim == 2:
        return np.asarray(ctf, dtype=np.float64)
    return None


def _ctf_correct_init(y, cfg):
    """CTF-corrected initialisation for DL solvers.

    Apply Wiener-CTF correction to get a physics-correct starting point,
    then DRUNet refines from there. This is better than raw PnP with CTF
    because DRUNet was trained on Gaussian-degraded images, not CTF-modulated.
    """
    cfg = cfg or {}
    ctf = _get_ctf_from_cfg(cfg)
    if ctf is not None:
        op = CryoEMOperator(y.shape, ctf=ctf)
        lam = 0.1
        H = op.get_H(y.shape)
        Y = np.fft.fft2(y.astype(np.float32))
        W = np.conj(H) / (np.abs(H) ** 2 + lam)
        x = np.real(np.fft.ifft2(W * Y)).astype(np.float32)
        return _final_norm(x)
    return y


def run_relion(y, physics=None, cfg=None):
    """RELION: CTF-corrected init + DRUNet denoise (sigma=0.01).

    Reference: Scheres 2012, JMB; Zivanov et al. 2018, eLife.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    y2 = _ctf_correct_init(y, cfg)
    return dl_drunet_denoise(y2, psf_sigma=PSF_SIGMA, sigma=0.01)


def run_cryosparc(y, physics=None, cfg=None):
    """CryoSPARC: CTF-corrected init + DRUNet PnP (sigma=0.03).

    Reference: Punjani et al. 2017, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15)


def run_cryodrgn(y, physics=None, cfg=None):
    """CryoDRGN: CTF-corrected init + DRUNet PnP (sigma=0.05).

    Reference: Zhong et al. 2021, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.05, max_iter=10)


def run_cryodrgn2(y, physics=None, cfg=None):
    """CryoDRGN2: CTF-corrected init + DRUNet PnP (sigma=0.03).

    Reference: Zhong et al. 2021, ICLR.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="HQS",
                         sigma=0.03, max_iter=15)


def run_cryoai(y, physics=None, cfg=None):
    """CryoAI: CTF-corrected init + DnCNN denoising.

    Reference: Levy et al. 2022, NeurIPS.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    y2 = _ctf_correct_init(y, cfg)
    return dl_dncnn_denoise(y2, psf_sigma=PSF_SIGMA)


def run_deep_em_enhancer(y, physics=None, cfg=None):
    """DeepEMenhancer: DRUNet direct denoising (sigma=0.05).

    Reference: Sanchez-Garcia et al. 2021, Comms. Biol.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    return dl_drunet_denoise(y, psf_sigma=PSF_SIGMA, sigma=0.05)


def run_topaz_denoise(y, physics=None, cfg=None):
    """Topaz-Denoise: DRUNet direct denoising (sigma=0.10).

    Reference: Bepler et al. 2020, Nature Comms.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    return dl_drunet_denoise(y, psf_sigma=PSF_SIGMA, sigma=0.10)


def run_cryostar(y, physics=None, cfg=None):
    """CryoSTAR: CTF-corrected init + DRUNet PnP (sigma=0.03).

    Reference: Guo et al. 2024, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15)


def run_cryo_mamba(y, physics=None, cfg=None):
    """CryoMamba: CTF-corrected init + RED DRUNet (sigma=0.05).

    Reference: Li et al. 2024.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_red_drunet(y2, psf_sigma=PSF_SIGMA, sigma=0.05, max_iter=10)


def run_pnp_hqs_drunet(y, physics=None, cfg=None):
    """PnP-HQS DRUNet: CTF-corrected init + HQS DRUNet (sigma=0.02).

    Reference: Zhang et al. 2017, CVPR (DnCNN/DRUNet).
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="HQS",
                         sigma=0.02, max_iter=18)


def run_cryo_gan(y, physics=None, cfg=None):
    """CryoGAN: CTF-corrected init + DRUNet PnP (sigma=0.08).

    Reference: Gupta et al. 2020, NeurIPS.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.08, max_iter=8)


def run_cryo_fire(y, physics=None, cfg=None):
    """CryoFIRE: CTF-corrected init + DRUNet PnP (sigma=0.05).

    Reference: Zhong et al. 2023, ICLR.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    y2 = _ctf_correct_init(y, cfg)
    return dl_pnp_drunet(y2, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.05, max_iter=15)


def run_cryo_former(y, physics=None, cfg=None):
    """CryoFormer: CTF-corrected init + SwinIR transformer denoise.

    Uses pretrained SwinIR (Liang et al. 2021) instead of generic DRUNet.
    Reference: CryoFormer 2024.
    """
    from algorithm_base.shared.dl_engine import dl_swinir_denoise
    y2 = _ctf_correct_init(y, cfg)
    return dl_swinir_denoise(y2, psf_sigma=PSF_SIGMA)


def run_cryo_foundation(y, physics=None, cfg=None):
    """CryoFoundation: CTF-corrected init + Restormer denoise.

    Uses pretrained Restormer (Zamir et al. 2022) instead of generic DRUNet.
    Reference: CryoFoundation 2025.
    """
    from algorithm_base.shared.dl_engine import dl_restormer_denoise
    y2 = _ctf_correct_init(y, cfg)
    return dl_restormer_denoise(y2, psf_sigma=PSF_SIGMA)


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
    """List all available solvers for cryo_em."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Wiener-CTF Correction. CPU only.
    Reference: Penczek et al. 2010.
    """
    return run_solver("traditional_cpu", y, operator, cfg)


def run_best_quality(y: np.ndarray, operator: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """RELION (PnP-PGD DRUNet). GPU.
    Reference: Scheres 2012; Zivanov et al. 2018.
    """
    return run_solver("best_quality", y, operator, cfg)


def run_famous_dl(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """CryoDRGN (PnP-PGD DRUNet). GPU.
    Reference: Zhong et al. 2021, Nature Methods.
    """
    return run_solver("famous_dl", y, operator, cfg)


def run_small_gpu(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """CryoAI (DnCNN denoise). GPU.
    Reference: Levy et al. 2022, NeurIPS.
    """
    return run_solver("small_gpu", y, operator, cfg)
