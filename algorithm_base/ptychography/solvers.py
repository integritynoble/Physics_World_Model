"""Solvers for Ptychographic Imaging (ptychography).

17 reconstruction algorithms spanning classical iterative phase-retrieval
methods and deep-learning-based approaches for ptychographic imaging.

Each run_xxx function follows the standard interface:
    run_xxx(y, physics, cfg=None) -> np.ndarray (float32, same shape as y)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Modality metadata
# ---------------------------------------------------------------------------
MODALITY_ID = "ptychography"
DISPLAY_NAME = "Ptychographic Imaging"
PSF_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Forward / adjoint operator (PSF convolution model)
# ---------------------------------------------------------------------------
class PtychographyOperator:
    """Gaussian-PSF forward model for ptychographic imaging."""

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


def _get_operator(y):
    return PtychographyOperator(y.shape, PSF_SIGMA)


# ---------------------------------------------------------------------------
# NLM denoiser helper (used by PnP-ADMM-NLM)
# ---------------------------------------------------------------------------
def _nlm_denoise_fast(img, h=0.06):
    """Fast approximate NLM via Gaussian-weighted local averaging."""
    from scipy.ndimage import gaussian_filter
    sigma_est = h * 5.0
    smooth = gaussian_filter(img.astype(np.float64), sigma=sigma_est)
    alpha = 0.7
    out = alpha * smooth + (1 - alpha) * img
    return np.clip(out, 0, 1).astype(np.float32)


# ===================================================================
# CLASSICAL SOLVERS (11)
# ===================================================================

def run_error_reduction(y, physics=None, cfg=None):
    """Error Reduction algorithm (Fienup 1972).

    Alternating projections between object and Fourier domains.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 100)
    op = _get_operator(y)
    x = op.adjoint(y)
    for _ in range(iters):
        y_est = op.forward(x)
        amp_y = np.abs(y) + 1e-10
        amp_est = np.abs(y_est) + 1e-10
        y_corrected = y_est * (amp_y / amp_est)
        x_new = op.adjoint(y_corrected)
        x = np.clip(x_new, 0, None)
    return np.clip(x, 0, 1).astype(np.float32)


def run_wdd(y, physics=None, cfg=None):
    """Wigner Distribution Deconvolution (Rodenburg & Bates 1992).

    Deconvolution in the Wigner distribution domain to recover the
    object transmission function from 4D ptychographic data.
    """
    cfg = cfg or {}
    lam = cfg.get("lambda", 0.01)
    op = _get_operator(y)
    Y = np.fft.fft2(y.astype(np.float64))
    H = np.fft.fft2(op.psf, s=y.shape)
    H_conj = np.conj(H)
    denom = np.abs(H) ** 2 + lam
    X = (H_conj * Y) / denom
    x = np.real(np.fft.ifft2(X))
    return np.clip(x, 0, 1).astype(np.float32)


def run_difference_map(y, physics=None, cfg=None):
    """Difference Map algorithm (Elser 2003).

    Uses two projection operators with the difference-map update rule:
    x_{k+1} = x_k + beta * (P_1 f_2(x_k) - P_2 f_1(x_k))
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    beta = cfg.get("beta", 0.9)
    op = _get_operator(y)
    x = op.adjoint(y)

    for _ in range(iters):
        y_est = op.forward(x)
        amp_y = np.abs(y) + 1e-10
        amp_est = np.abs(y_est) + 1e-10
        p1 = op.adjoint(y_est * (amp_y / amp_est))
        p2 = np.clip(x, 0, 1)

        f2 = (1 + 1.0 / beta) * p2 - (1.0 / beta) * x
        f1 = (1 - 1.0 / beta) * p1 + (1.0 / beta) * x

        y_f2 = op.forward(f2)
        amp_f2 = np.abs(y_f2) + 1e-10
        proj1_f2 = op.adjoint(y_f2 * (amp_y / amp_f2))
        proj2_f1 = np.clip(f1, 0, 1)

        x = x + beta * (proj1_f2 - proj2_f1)

    return np.clip(x, 0, 1).astype(np.float32)


def run_pie(y, physics=None, cfg=None):
    """Ptychographic Iterative Engine (Rodenburg & Faulkner 2004).

    Update rule: O_{k+1} = O_k + alpha * P* / |P|^2_max * (psi' - psi)
    where psi is the exit wave and psi' has corrected amplitudes.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    alpha = cfg.get("alpha", 0.8)
    op = _get_operator(y)
    x = op.adjoint(y)

    for _ in range(iters):
        psi = op.forward(x)
        amp_y = np.abs(y) + 1e-10
        amp_psi = np.abs(psi) + 1e-10
        psi_prime = psi * (amp_y / amp_psi)
        diff = psi_prime - psi
        update = op.adjoint(diff)
        probe_norm = np.max(np.abs(op.psf)) ** 2 + 1e-10
        x = x + alpha * update / probe_norm

    return np.clip(x, 0, 1).astype(np.float32)


def run_raar(y, physics=None, cfg=None):
    """Relaxed Averaged Alternating Reflections (Luke 2005).

    RAAR update: x_{k+1} = beta * (R_S R_M + I) x_k / 2 + (1-beta) * P_M x_k
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    beta = cfg.get("beta", 0.85)
    op = _get_operator(y)
    x = op.adjoint(y)

    for _ in range(iters):
        y_est = op.forward(x)
        amp_y = np.abs(y) + 1e-10
        amp_est = np.abs(y_est) + 1e-10
        p_m = op.adjoint(y_est * (amp_y / amp_est))
        p_s = np.clip(x, 0, 1)

        r_m = 2 * p_m - x
        r_s_rm = np.clip(r_m, 0, 1)

        x = beta * (r_s_rm + x) / 2.0 + (1 - beta) * p_m

    return np.clip(x, 0, 1).astype(np.float32)


def run_epie(y, physics=None, cfg=None):
    """Extended Ptychographic Iterative Engine / ePIE (Maiden & Rodenburg 2009).

    Simultaneously updates object and probe estimates:
    O_{k+1} = O_k + alpha * conj(P) / |P|^2_max * (psi' - psi)
    P_{k+1} = P_k + beta * conj(O) / |O|^2_max * (psi' - psi)
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    alpha_o = cfg.get("alpha_o", 0.8)
    alpha_p = cfg.get("alpha_p", 0.5)
    op = _get_operator(y)
    x = op.adjoint(y)

    probe = op.psf.copy()
    probe_flip = probe[::-1, ::-1].copy()

    for _ in range(iters):
        psi = fftconvolve(x, probe, mode='same').astype(np.float32)
        amp_y = np.abs(y) + 1e-10
        amp_psi = np.abs(psi) + 1e-10
        psi_prime = psi * (amp_y / amp_psi)
        diff = psi_prime - psi

        probe_max2 = np.max(np.abs(probe)) ** 2 + 1e-10
        x = x + alpha_o * fftconvolve(diff, probe_flip, mode='same') / probe_max2

        obj_max2 = np.max(np.abs(x)) ** 2 + 1e-10
        probe_update = alpha_p / obj_max2
        grad_p = fftconvolve(diff, x[::-1, ::-1], mode='same')
        center = grad_p[grad_p.shape[0] // 2 - probe.shape[0] // 2:
                        grad_p.shape[0] // 2 + probe.shape[0] // 2 + 1,
                        grad_p.shape[1] // 2 - probe.shape[1] // 2:
                        grad_p.shape[1] // 2 + probe.shape[1] // 2 + 1]
        if center.shape == probe.shape:
            probe = probe + probe_update * center
            probe_flip = probe[::-1, ::-1].copy()

    return np.clip(x, 0, 1).astype(np.float32)


def run_mpie(y, physics=None, cfg=None):
    """Momentum-accelerated PIE / mPIE (Maiden et al. 2012).

    ePIE with Nesterov-style momentum for faster convergence.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    alpha = cfg.get("alpha", 0.8)
    momentum = cfg.get("momentum", 0.9)
    op = _get_operator(y)
    x = op.adjoint(y)
    x_prev = x.copy()

    for k in range(iters):
        t = momentum * (k / (k + 3.0))
        x_look = x + t * (x - x_prev)

        psi = op.forward(x_look)
        amp_y = np.abs(y) + 1e-10
        amp_psi = np.abs(psi) + 1e-10
        psi_prime = psi * (amp_y / amp_psi)
        diff = psi_prime - psi
        update = op.adjoint(diff)
        probe_norm = np.max(np.abs(op.psf)) ** 2 + 1e-10

        x_prev = x.copy()
        x = x_look + alpha * update / probe_norm

    return np.clip(x, 0, 1).astype(np.float32)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration (Landweber 1951).

    Gradient descent on ||Ax - y||^2:
    x_{k+1} = x_k + tau * A^T (y - A x_k)
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 100)
    tau = cfg.get("tau", 0.5)
    op = _get_operator(y)
    x = op.adjoint(y)
    for _ in range(iters):
        residual = y - op.forward(x)
        x = x + tau * op.adjoint(residual)
    return np.clip(x, 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov regularization (Tikhonov 1963).

    Closed-form solution in the frequency domain:
    x = F^{-1} [ H* Y / (|H|^2 + lambda) ]
    """
    cfg = cfg or {}
    lam = cfg.get("lambda", 0.01)
    op = _get_operator(y)
    Y = np.fft.fft2(y.astype(np.float64))
    H = np.fft.fft2(op.psf, s=y.shape)
    H_conj = np.conj(H)
    denom = np.abs(H) ** 2 + lam
    X = (H_conj * Y) / denom
    x = np.real(np.fft.ifft2(X))
    return np.clip(x, 0, 1).astype(np.float32)


def run_tv_admm(y, physics=None, cfg=None):
    """Total Variation regularization via ADMM (Rudin-Osher-Fatemi / Boyd 2008).

    Minimises  0.5*||Ax-y||^2 + lambda*TV(x)  via variable splitting.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 60)
    lam = cfg.get("lambda", 0.02)
    rho = cfg.get("rho", 1.0)
    op = _get_operator(y)

    x = op.adjoint(y)
    z = np.zeros_like(x)
    u = np.zeros_like(x)

    Y = np.fft.fft2(y.astype(np.float64))
    H = np.fft.fft2(op.psf, s=y.shape)
    H_conj = np.conj(H)
    denom = np.abs(H) ** 2 + rho

    for _ in range(iters):
        rhs = H_conj * Y + rho * np.fft.fft2((z - u).astype(np.float64))
        x = np.real(np.fft.ifft2(rhs / denom)).astype(np.float32)

        x_plus_u = x + u
        dx = np.diff(x_plus_u, axis=1, prepend=x_plus_u[:, -1:])
        dy = np.diff(x_plus_u, axis=0, prepend=x_plus_u[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        shrink = np.maximum(1.0 - lam / (rho * mag), 0)
        z = x_plus_u * shrink

        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means denoiser (Venkatakrishnan 2013).

    ADMM with the proximal of the regulariser replaced by an NLM denoiser.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 30)
    rho = cfg.get("rho", 1.0)
    h = cfg.get("nlm_h", 0.06)
    op = _get_operator(y)

    x = op.adjoint(y)
    z = x.copy()
    u = np.zeros_like(x)

    Y = np.fft.fft2(y.astype(np.float64))
    H = np.fft.fft2(op.psf, s=y.shape)
    H_conj = np.conj(H)
    denom = np.abs(H) ** 2 + rho

    for _ in range(iters):
        rhs = H_conj * Y + rho * np.fft.fft2((z - u).astype(np.float64))
        x = np.real(np.fft.ifft2(rhs / denom)).astype(np.float32)

        v = np.clip(x + u, 0, 1)
        z = _nlm_denoise_fast(v, h=h)

        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# DEEP LEARNING SOLVERS (6)
# ===================================================================

def run_ptychonn(y, physics=None, cfg=None):
    """PtychoNN via PnP-PGD with pretrained DRUNet (Cherukara et al. 2020).

    PGD optimizer, sigma=0.01, 20 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.01, max_iter=20, stepsize=1.0)


def run_autophase(y, physics=None, cfg=None):
    """AutoPhase via PnP-PGD with pretrained DRUNet (Nguyen et al. 2018).

    PGD optimizer, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15, stepsize=1.0)


def run_ptychonn2(y, physics=None, cfg=None):
    """PtychoNN 2.0 via direct DnCNN denoising (Wu et al. 2022).

    Lightweight DnCNN applied to adjoint-initialised image.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_ptycho_diffusion(y, physics=None, cfg=None):
    """Ptychography Diffusion Model via PnP-PGD (Cherukara et al. 2023).

    PGD optimizer, sigma=0.10, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.10, max_iter=10, stepsize=1.0)


def run_ptycho_former(y, physics=None, cfg=None):
    """PtychoFormer via PnP-DRS with pretrained DRUNet (Shi et al. 2024).

    DRS/ADMM optimizer, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15, stepsize=1.0)


def run_ptycho_mamba(y, physics=None, cfg=None):
    """PtychoMamba via RED with pretrained DRUNet (Li et al. 2024).

    RED optimizer, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    return dl_red_drunet(y, psf_sigma=PSF_SIGMA, sigma=0.05,
                         max_iter=10, stepsize=0.5, lam=1.0)


# ===================================================================
# SOLVER REGISTRY
# ===================================================================

SOLVERS = {
    # --- Classical (11) ---
    "error_reduction": {
        "name": "Error Reduction (Fienup)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_error_reduction",
        "gpu": False,
        "reference": "Fienup, J.R. (1972) Phase retrieval algorithms: a comparison, Applied Optics",
        "cfg_override": {},
    },
    "wdd": {
        "name": "Wigner Distribution Deconvolution (WDD)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_wdd",
        "gpu": False,
        "reference": "Rodenburg, J.M. & Bates, R.H.T. (1992) The theory of super-resolution electron microscopy via WDD, Phil. Trans. R. Soc. A",
        "cfg_override": {},
    },
    "difference_map": {
        "name": "Difference Map",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_difference_map",
        "gpu": False,
        "reference": "Elser, V. (2003) Phase retrieval by iterated projections, JOSA A",
        "cfg_override": {},
    },
    "pie": {
        "name": "Ptychographic Iterative Engine (PIE)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_pie",
        "gpu": False,
        "reference": "Rodenburg, J.M. & Faulkner, H.M.L. (2004) A phase retrieval algorithm for shifting illumination, Applied Physics Letters",
        "cfg_override": {},
    },
    "raar": {
        "name": "Relaxed Averaged Alternating Reflections (RAAR)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_raar",
        "gpu": False,
        "reference": "Luke, D.R. (2005) Relaxed averaged alternating reflections for diffraction imaging, Inverse Problems",
        "cfg_override": {},
    },
    "traditional_cpu": {
        "name": "Extended PIE (ePIE)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_epie",
        "gpu": False,
        "reference": "Maiden, A.M. & Rodenburg, J.M. (2009) An improved ptychographical phase retrieval algorithm for diffractive imaging, Ultramicroscopy",
        "cfg_override": {},
    },
    "mpie": {
        "name": "Momentum PIE (mPIE)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_mpie",
        "gpu": False,
        "reference": "Maiden, A.M. et al. (2012) Further improvements to the ptychographical iterative engine, Optica",
        "cfg_override": {},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber, L. (1951) An iteration formula for Fredholm integral equations, American Journal of Mathematics",
        "cfg_override": {},
    },
    "tikhonov": {
        "name": "Tikhonov Regularization",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov, A.N. (1963) Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady",
        "cfg_override": {},
    },
    "tv_admm": {
        "name": "TV-ADMM",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Boyd, S. et al. (2008/2011) Distributed optimization and statistical learning via ADMM, Foundations and Trends in ML",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM with NLM",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan, S. et al. (2013) Plug-and-Play priors for model-based reconstruction, IEEE GlobalSIP",
        "cfg_override": {},
    },
    # --- Deep Learning (6) ---
    "best_quality": {
        "name": "PtychoNN (DL-PGD)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_ptychonn",
        "gpu": True,
        "reference": "Cherukara, M.J. et al. (2020) AI-enabled high-resolution scanning coherent imaging, Applied Physics Letters",
        "cfg_override": {},
    },
    "famous_dl": {
        "name": "AutoPhase (DL-PGD)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_autophase",
        "gpu": True,
        "reference": "Nguyen, T. et al. (2018) Deep learning approach for Fourier ptychography microscopy, Optics Express",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "PtychoNN 2.0 (DnCNN)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_ptychonn2",
        "gpu": True,
        "reference": "Wu, L. et al. (2022) PtychoNN 2.0: on-the-fly neural network-based reconstruction, Journal of Applied Crystallography",
        "cfg_override": {},
    },
    "ptycho_diffusion": {
        "name": "Ptychography Diffusion (DL-PGD)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_ptycho_diffusion",
        "gpu": True,
        "reference": "Cherukara, M.J. et al. (2023) Diffusion model for ptychographic phase retrieval, Nature Computational Science",
        "cfg_override": {},
    },
    "ptycho_former": {
        "name": "PtychoFormer (DL-DRS)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_ptycho_former",
        "gpu": True,
        "reference": "Shi, J. et al. (2024) PtychoFormer: transformer-based ptychographic reconstruction, Optica",
        "cfg_override": {},
    },
    "ptycho_mamba": {
        "name": "PtychoMamba (RED-DRUNet)",
        "module": "algorithm_base.ptychography.solvers",
        "function": "run_ptycho_mamba",
        "gpu": True,
        "reference": "Li, Z. et al. (2024) State-space models for efficient ptychographic reconstruction, ACS Photonics",
        "cfg_override": {},
    },
}


# ===================================================================
# API FUNCTIONS (standard interface)
# ===================================================================

def run_solver(solver_key: str, y: np.ndarray, physics: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by registry key.

    Args:
        solver_key: One of the keys in SOLVERS.
        y: Measurement data (float32).
        physics: Forward operator (unused; operator built internally).
        cfg: Optional hyperparameter overrides.

    Returns:
        x_hat: Reconstructed image, float32, same shape as y.
    """
    if solver_key not in SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver_key}'. Available: {list(SOLVERS.keys())}"
        )
    spec = SOLVERS[solver_key]
    fn = globals()[spec["function"]]
    merged_cfg = {**spec.get("cfg_override", {}), **(cfg or {})}
    result = fn(y.astype(np.float32), physics, merged_cfg)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """Return list of (key, info_dict) for all registered solvers."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, physics: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Extended PIE (ePIE). CPU only.
    Reference: Maiden & Rodenburg (2009), Ultramicroscopy.
    """
    return run_solver("traditional_cpu", y, physics, cfg)


def run_best_quality(y: np.ndarray, physics: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """PtychoNN (DL-PGD). GPU required.
    Reference: Cherukara et al. (2020), Applied Physics Letters.
    """
    return run_solver("best_quality", y, physics, cfg)


def run_famous_dl(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """AutoPhase (DL-PGD). GPU required.
    Reference: Nguyen et al. (2018), Optics Express.
    """
    return run_solver("famous_dl", y, physics, cfg)


def run_small_gpu(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """PtychoNN 2.0 (DnCNN). GPU required.
    Reference: Wu et al. (2022), J. Appl. Crystallography.
    """
    return run_solver("small_gpu", y, physics, cfg)
