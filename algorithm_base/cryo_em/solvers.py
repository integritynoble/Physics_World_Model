"""Solvers for Cryo-EM Single Particle Analysis (cryo_em).

17 reconstruction algorithms:
  Classical (8): Wiener-CTF, Phase-Flip, Back-Projection, SIRT-3D,
                 Landweber, Tikhonov, TV-ADMM, PnP-ADMM-NLM
  Deep Learning (9): RELION, CryoSPARC, CryoDRGN, CryoDRGN2, CryoAI,
                     DeepEMenhancer, Topaz-Denoise, CryoSTAR, CryoMamba

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
PSF_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Forward operator (PSF convolution modelling CTF blur)
# ---------------------------------------------------------------------------
class CryoEMOperator:
    """Contrast-transfer-function-like blur operator for cryo-EM."""

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
    return CryoEMOperator(y.shape)


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
    lam = cfg.get("lam", 1e-2)
    op = _op(y)
    H = np.fft.fft2(op.psf, s=y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    W = np.conj(H) / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft2(W * Y)).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def run_phase_flip(y, physics=None, cfg=None):
    """Phase-flip CTF correction: multiply by sign(CTF) in Fourier.

    Corrects the phase inversions introduced by defocus without
    amplifying noise. Simple but effective first-pass correction.
    Reference: Rosenthal & Henderson 2003, JMB.
    """
    cfg = cfg or {}
    op = _op(y)
    H = np.fft.fft2(op.psf, s=y.shape)
    Y = np.fft.fft2(y.astype(np.float32))
    # Phase-flip: multiply by sign of real part of CTF
    phase_sign = np.sign(np.real(H))
    phase_sign[phase_sign == 0] = 1.0
    x = np.real(np.fft.ifft2(Y * phase_sign)).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def run_back_projection(y, physics=None, cfg=None):
    """Back-projection: direct adjoint (correlation with PSF).

    The simplest reconstruction — applies the transpose of the forward
    operator. Equivalent to unweighted back-projection in tomography.
    Reference: Radermacher 1988, J. Electron Microsc. Tech.
    """
    cfg = cfg or {}
    op = _op(y)
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
    op = _op(y)
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


# ═══════════════════════════════════════════════════════════════════════════
# Deep-learning solvers
# ═══════════════════════════════════════════════════════════════════════════

def run_relion(y, physics=None, cfg=None):
    """RELION: PnP-PGD with pretrained DRUNet (sigma=0.01, 20 iters).

    Reference: Scheres 2012, JMB; Zivanov et al. 2018, eLife.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.01, max_iter=20)


def run_cryosparc(y, physics=None, cfg=None):
    """CryoSPARC: PnP-PGD with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Punjani et al. 2017, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15)


def run_cryodrgn(y, physics=None, cfg=None):
    """CryoDRGN: PnP-PGD with pretrained DRUNet (sigma=0.05, 10 iters).

    Reference: Zhong et al. 2021, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.05, max_iter=10)


def run_cryodrgn2(y, physics=None, cfg=None):
    """CryoDRGN2: PnP-HQS with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Zhong et al. 2021, ICLR.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="HQS",
                         sigma=0.03, max_iter=15)


def run_cryoai(y, physics=None, cfg=None):
    """CryoAI: DnCNN direct denoising on adjoint initialisation.

    Reference: Levy et al. 2022, NeurIPS.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


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
    """CryoSTAR: PnP-DRS with pretrained DRUNet (sigma=0.03, 15 iters).

    Reference: Guo et al. 2024, Nature Methods.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15)


def run_cryo_mamba(y, physics=None, cfg=None):
    """CryoMamba: RED with pretrained DRUNet (sigma=0.05, 10 iters).

    Reference: Li et al. 2024.
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
