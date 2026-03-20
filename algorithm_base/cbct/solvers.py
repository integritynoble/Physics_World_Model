"""Solvers for Cone-Beam Computed Tomography (CBCT) (cbct).

22 reconstruction algorithms spanning classical filtered back-projection
variants, iterative algebraic/statistical methods, regularised inversions,
and deep-learning-based approaches for CBCT reconstruction.

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
MODALITY_ID = "cbct"
DISPLAY_NAME = "Cone-Beam Computed Tomography (CBCT)"
PSF_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Forward / adjoint operator (PSF convolution model)
# ---------------------------------------------------------------------------
class CBCTOperator:
    """Gaussian-PSF forward model for cone-beam CT imaging."""

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
    return CBCTOperator(y.shape, PSF_SIGMA)


# ---------------------------------------------------------------------------
# Filter kernels for FDK variants
# ---------------------------------------------------------------------------
def _ramp_filter(n):
    """Ram-Lak (ramp) filter in frequency domain."""
    freqs = np.fft.rfftfreq(n)
    return np.abs(freqs).astype(np.float32)


def _shepp_logan_filter(n):
    """Shepp-Logan filter: ramp * sinc."""
    freqs = np.fft.rfftfreq(n)
    ramp = np.abs(freqs)
    sinc = np.sinc(freqs / (2.0 * np.max(freqs + 1e-10)))
    return (ramp * sinc).astype(np.float32)


def _hamming_filter(n):
    """Hamming-windowed ramp filter."""
    freqs = np.fft.rfftfreq(n)
    ramp = np.abs(freqs)
    fmax = np.max(freqs) + 1e-10
    hamming = 0.54 + 0.46 * np.cos(np.pi * freqs / fmax)
    return (ramp * hamming).astype(np.float32)


def _hann_filter(n):
    """Hann-windowed ramp filter."""
    freqs = np.fft.rfftfreq(n)
    ramp = np.abs(freqs)
    fmax = np.max(freqs) + 1e-10
    hann = 0.5 * (1.0 + np.cos(np.pi * freqs / fmax))
    return (ramp * hann).astype(np.float32)


def _fdk_reconstruct(y, filt_fn):
    """FDK-style reconstruction: adjoint + frequency-domain filter.

    1. Apply chosen ramp/window filter along each row of y.
    2. Back-project via the adjoint operator.
    """
    op = _get_operator(y)
    n = y.shape[-1]
    H = filt_fn(n)
    # Filter each row in frequency domain
    Y_f = np.fft.rfft(y.astype(np.float64), axis=-1)
    Y_filtered = Y_f * H[np.newaxis, :]
    y_filt = np.fft.irfft(Y_filtered, n=n, axis=-1).astype(np.float32)
    # Back-project
    x = op.adjoint(y_filt)
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# NLM denoiser helper (used by PnP and FDK+NLM)
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
# CLASSICAL SOLVERS (17)
# ===================================================================

def run_fdk_ramlak(y, physics=None, cfg=None):
    """FDK with Ram-Lak (ramp) filter (Feldkamp, Davis & Kress 1984).

    Standard FDK: ramp-filtered back-projection for cone-beam geometry.
    """
    return _fdk_reconstruct(y, _ramp_filter)


def run_fdk_shepp_logan(y, physics=None, cfg=None):
    """FDK with Shepp-Logan filter (Shepp & Logan 1974).

    Shepp-Logan windowed ramp filter suppresses high-frequency noise.
    """
    return _fdk_reconstruct(y, _shepp_logan_filter)


def run_fdk_hamming(y, physics=None, cfg=None):
    """FDK with Hamming window.

    Hamming-windowed ramp filter provides smooth roll-off.
    """
    return _fdk_reconstruct(y, _hamming_filter)


def run_fdk_hann(y, physics=None, cfg=None):
    """FDK with Hann window.

    Hann-windowed ramp filter — strongest noise suppression among FDK variants.
    """
    return _fdk_reconstruct(y, _hann_filter)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration (Landweber 1951).

    Gradient descent on ||Ax - y||^2:
    x_{k+1} = x_k + tau * A^T(y - Ax_k)
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


def run_art(y, physics=None, cfg=None):
    """Algebraic Reconstruction Technique / ART (Gordon, Bender & Herman 1970).

    Row-by-row (Kaczmarz) updates on the system Ax = y.
    Approximated here via sequential pixel-row projection.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)
    lam = cfg.get("lambda", 0.25)
    op = _get_operator(y)
    x = op.adjoint(y)

    rows = y.shape[0]
    for _ in range(iters):
        for r in range(0, rows, max(1, rows // 16)):
            row_slice = slice(r, min(r + max(1, rows // 16), rows))
            y_row = y[row_slice, :]
            # Forward project current estimate for this row subset
            ax_row = op.forward(x)[row_slice, :]
            residual_row = y_row - ax_row
            # Create full-size residual padded with zeros
            res_full = np.zeros_like(y)
            res_full[row_slice, :] = residual_row
            # Back-project and update
            x = x + lam * op.adjoint(res_full)
        x = np.clip(x, 0, None)

    return np.clip(x, 0, 1).astype(np.float32)


def run_sirt(y, physics=None, cfg=None):
    """Simultaneous Iterative Reconstruction Technique (Gilbert 1972).

    SIRT averages corrections over all equations simultaneously:
    x_{k+1} = x_k + C * A^T R (y - Ax_k)
    where C, R are diagonal normalisation matrices.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    op = _get_operator(y)
    x = op.adjoint(y)

    # Compute row/column sums for normalisation
    ones_y = np.ones_like(y)
    ones_x = np.ones_like(x)
    col_sum = op.adjoint(ones_y) + 1e-10  # C^{-1}
    row_sum = op.forward(ones_x) + 1e-10  # R^{-1}

    for _ in range(iters):
        residual = (y - op.forward(x)) / row_sum
        correction = op.adjoint(residual) / col_sum
        x = x + correction
        x = np.clip(x, 0, None)

    return np.clip(x, 0, 1).astype(np.float32)


def run_cgls(y, physics=None, cfg=None):
    """Conjugate Gradient Least Squares (Hestenes & Stiefel 1952).

    Solves min ||Ax - y||^2 using the CG method on the normal equations
    A^T A x = A^T y.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)
    op = _get_operator(y)

    x = np.zeros_like(y)
    r = y - op.forward(x)
    s = op.adjoint(r)
    p = s.copy()
    gamma = np.sum(s ** 2)

    for _ in range(iters):
        Ap = op.forward(p)
        alpha = gamma / (np.sum(Ap ** 2) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        s = op.adjoint(r)
        gamma_new = np.sum(s ** 2)
        if gamma_new < 1e-14:
            break
        beta = gamma_new / (gamma + 1e-12)
        p = s + beta * p
        gamma = gamma_new

    return np.clip(x, 0, 1).astype(np.float32)


def run_sart(y, physics=None, cfg=None):
    """Simultaneous ART / SART (Andersen & Kak 1984).

    SART uses subset-based updates with ray-length weighting,
    converging faster than standard ART.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 60)
    lam = cfg.get("lambda", 0.8)
    op = _get_operator(y)
    x = op.adjoint(y)

    # Precompute normalisation
    ones_y = np.ones_like(y)
    ones_x = np.ones_like(x)
    col_norm = op.adjoint(ones_y) + 1e-10
    row_norm = op.forward(ones_x) + 1e-10

    n_subsets = cfg.get("n_subsets", 4)
    rows = y.shape[0]
    subset_size = max(1, rows // n_subsets)

    for _ in range(iters):
        for s in range(n_subsets):
            r0 = s * subset_size
            r1 = min(r0 + subset_size, rows)
            row_slice = slice(r0, r1)

            y_sub = y[row_slice, :]
            ax_sub = op.forward(x)[row_slice, :]
            residual = np.zeros_like(y)
            residual[row_slice, :] = (y_sub - ax_sub) / row_norm[row_slice, :]

            correction = op.adjoint(residual) / col_norm
            x = x + lam * correction
            x = np.clip(x, 0, None)

    return np.clip(x, 0, 1).astype(np.float32)


def run_mlem(y, physics=None, cfg=None):
    """Maximum Likelihood Expectation Maximization / ML-EM (Shepp & Vardi 1982).

    Multiplicative update: x_{k+1} = x_k / s * A^T (y / Ax_k)
    where s = A^T 1.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)
    op = _get_operator(y)

    x = op.adjoint(y)
    x = np.clip(x, 1e-6, None)

    sens = op.adjoint(np.ones_like(y))
    sens = np.clip(sens, 1e-6, None)

    for _ in range(iters):
        y_est = op.forward(x)
        y_est = np.clip(y_est, 1e-10, None)
        ratio = y / y_est
        correction = op.adjoint(ratio)
        x = x * correction / sens
        x = np.clip(x, 1e-6, None)

    return np.clip(x, 0, 1).astype(np.float32)


def run_osem(y, physics=None, cfg=None):
    """Ordered Subsets EM / OS-EM (Hudson & Larkin 1994).

    MLEM with ordered subsets for accelerated convergence.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 30)
    n_subsets = cfg.get("n_subsets", 4)
    op = _get_operator(y)

    x = op.adjoint(y)
    x = np.clip(x, 1e-6, None)

    rows = y.shape[0]
    subset_size = max(1, rows // n_subsets)

    for _ in range(iters):
        for s in range(n_subsets):
            r0 = s * subset_size
            r1 = min(r0 + subset_size, rows)
            row_slice = slice(r0, r1)

            y_sub = y[row_slice, :]
            y_est_full = op.forward(x)
            y_est_sub = np.clip(y_est_full[row_slice, :], 1e-10, None)
            ratio = np.ones_like(y)
            ratio[row_slice, :] = y_sub / y_est_sub

            sens_sub = np.zeros_like(y)
            sens_sub[row_slice, :] = 1.0
            sens = op.adjoint(sens_sub)
            sens = np.clip(sens, 1e-10, None)

            correction = op.adjoint(ratio)
            x = x * correction / sens
            x = np.clip(x, 1e-6, None)

    return np.clip(x, 0, 1).astype(np.float32)


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov regularization (Tikhonov 1963).

    Closed-form in frequency domain:
    x = F^{-1}[ H* Y / (|H|^2 + lambda) ]
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
    """Total Variation via ADMM (Sidky, Kao & Pan 2008).

    Minimises  0.5*||Ax-y||^2 + lambda*TV(x)  using ADMM splitting.
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
        # x-update
        rhs = H_conj * Y + rho * np.fft.fft2((z - u).astype(np.float64))
        x = np.real(np.fft.ifft2(rhs / denom)).astype(np.float32)

        # z-update: anisotropic TV proximal (soft-threshold on gradients)
        x_plus_u = x + u
        dx = np.diff(x_plus_u, axis=1, prepend=x_plus_u[:, -1:])
        dy = np.diff(x_plus_u, axis=0, prepend=x_plus_u[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        shrink = np.maximum(1.0 - lam / (rho * mag), 0)
        z = x_plus_u * shrink

        # u-update (dual variable)
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_chambolle_pock(y, physics=None, cfg=None):
    """Chambolle-Pock primal-dual algorithm (Chambolle & Pock 2011).

    Solves min_x  0.5||Ax-y||^2 + lambda ||nabla x||_1
    via primal-dual splitting with optimal step sizes.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    lam = cfg.get("lambda", 0.02)
    tau = cfg.get("tau", 0.1)
    sigma = cfg.get("sigma", 0.1)
    theta = cfg.get("theta", 1.0)
    op = _get_operator(y)

    x = op.adjoint(y)
    x_bar = x.copy()
    # Dual variables for gradient operator (2 components)
    p_x = np.zeros_like(x)
    p_y = np.zeros_like(x)

    for _ in range(iters):
        # Dual update: gradient of x_bar
        gx = np.diff(x_bar, axis=1, prepend=x_bar[:, -1:])
        gy = np.diff(x_bar, axis=0, prepend=x_bar[-1:, :])
        p_x = p_x + sigma * gx
        p_y = p_y + sigma * gy
        # Project onto ||p|| <= lambda
        mag = np.sqrt(p_x ** 2 + p_y ** 2) + 1e-10
        scale = np.minimum(1.0, lam / mag)
        p_x = p_x * scale
        p_y = p_y * scale

        # Divergence of p (adjoint of gradient)
        div_p = (np.roll(p_x, -1, axis=1) - p_x) + (np.roll(p_y, -1, axis=0) - p_y)

        # Primal update
        x_old = x.copy()
        # Data fidelity gradient
        residual = op.forward(x) - y
        grad_data = op.adjoint(residual)
        x = x - tau * (grad_data - div_p)
        x = np.clip(x, 0, None)

        # Over-relaxation
        x_bar = x + theta * (x - x_old)

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means (Venkatakrishnan et al. 2013).

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
        # x-update
        rhs = H_conj * Y + rho * np.fft.fft2((z - u).astype(np.float64))
        x = np.real(np.fft.ifft2(rhs / denom)).astype(np.float32)

        # z-update: NLM denoiser
        v = np.clip(x + u, 0, 1)
        z = _nlm_denoise_fast(v, h=h)

        # u-update
        u = u + x - z

    return np.clip(x, 0, 1).astype(np.float32)


def run_pnp_fista_nlm(y, physics=None, cfg=None):
    """Plug-and-Play FISTA with NLM denoiser (Beck & Teboulle 2009 + PnP).

    FISTA-accelerated proximal gradient with NLM as the proximal map.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)
    tau = cfg.get("tau", 0.3)
    h = cfg.get("nlm_h", 0.05)
    op = _get_operator(y)

    x = op.adjoint(y)
    z = x.copy()
    t = 1.0

    for _ in range(iters):
        # Gradient step on data fidelity
        residual = op.forward(z) - y
        grad = op.adjoint(residual)
        x_new = z - tau * grad

        # Proximal step: NLM denoising
        x_new = _nlm_denoise_fast(np.clip(x_new, 0, 1), h=h)

        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.0
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return np.clip(x, 0, 1).astype(np.float32)


def run_fdk_nlm(y, physics=None, cfg=None):
    """FDK + Non-Local Means post-processing (Buades, Coll & Morel 2005).

    Ram-Lak FDK reconstruction followed by NLM denoising.
    Best-quality classical method combining analytical and statistical denoising.
    """
    cfg = cfg or {}
    h = cfg.get("nlm_h", 0.05)
    x_fdk = _fdk_reconstruct(y, _ramp_filter)
    x_out = _nlm_denoise_fast(x_fdk, h=h)
    return np.clip(x_out, 0, 1).astype(np.float32)


# ===================================================================
# DEEP LEARNING SOLVERS (5)
# ===================================================================

def run_fdk_dl(y, physics=None, cfg=None):
    """FDK-DL via PnP-PGD with pretrained DRUNet (Chen et al. 2017).

    PGD optimizer, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.03, max_iter=15, stepsize=1.0)


def run_cbct_unet(y, physics=None, cfg=None):
    """CBCT-UNet via direct DnCNN denoising (Jin et al. 2017).

    Lightweight DnCNN applied to adjoint-initialised image.
    """
    from algorithm_base.shared.dl_engine import dl_dncnn_denoise
    return dl_dncnn_denoise(y, psf_sigma=PSF_SIGMA)


def run_cbct_diffusion(y, physics=None, cfg=None):
    """CBCT Diffusion Model via PnP-PGD (Chung et al. 2023).

    PGD optimizer, sigma=0.10, 10 iterations — higher noise level
    for diffusion-style iterative refinement.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="PGD",
                         sigma=0.10, max_iter=10, stepsize=1.0)


def run_cbct_naf(y, physics=None, cfg=None):
    """CBCT Neural Attenuation Fields via PnP-DRS (Zha et al. 2024).

    DRS/ADMM optimizer, sigma=0.03, 15 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    return dl_pnp_drunet(y, psf_sigma=PSF_SIGMA, optimizer="DRS",
                         sigma=0.03, max_iter=15, stepsize=1.0)


def run_cbct_mamba(y, physics=None, cfg=None):
    """CBCT-Mamba via RED with pretrained DRUNet (Wang et al. 2024).

    RED optimizer, sigma=0.05, 10 iterations.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    return dl_red_drunet(y, psf_sigma=PSF_SIGMA, sigma=0.05,
                         max_iter=10, stepsize=0.5, lam=1.0)


# ===================================================================
# SOLVER REGISTRY
# ===================================================================

SOLVERS = {
    # --- Classical: FDK variants (4) ---
    "traditional_cpu": {
        "name": "FDK Ram-Lak",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_ramlak",
        "gpu": False,
        "reference": "Feldkamp, L.A., Davis, L.C. & Kress, J.W. (1984) Practical cone-beam algorithm, JOSA A",
        "cfg_override": {},
    },
    "fdk_shepp_logan": {
        "name": "FDK Shepp-Logan",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_shepp_logan",
        "gpu": False,
        "reference": "Shepp, L.A. & Logan, B.F. (1974) The Fourier reconstruction of a head section, IEEE Trans. Nuclear Science",
        "cfg_override": {},
    },
    "fdk_hamming": {
        "name": "FDK Hamming",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_hamming",
        "gpu": False,
        "reference": "Feldkamp, L.A., Davis, L.C. & Kress, J.W. (1984) Practical cone-beam algorithm, JOSA A",
        "cfg_override": {},
    },
    "fdk_hann": {
        "name": "FDK Hann",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_hann",
        "gpu": False,
        "reference": "Feldkamp, L.A., Davis, L.C. & Kress, J.W. (1984) Practical cone-beam algorithm, JOSA A",
        "cfg_override": {},
    },
    # --- Classical: Iterative algebraic (5) ---
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber, L. (1951) An iteration formula for Fredholm integral equations, American Journal of Mathematics",
        "cfg_override": {},
    },
    "art": {
        "name": "Algebraic Reconstruction Technique (ART)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_art",
        "gpu": False,
        "reference": "Gordon, R., Bender, R. & Herman, G.T. (1970) Algebraic reconstruction techniques (ART), Journal of Theoretical Biology",
        "cfg_override": {},
    },
    "sirt": {
        "name": "Simultaneous Iterative Reconstruction (SIRT)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_sirt",
        "gpu": False,
        "reference": "Gilbert, P. (1972) Iterative methods for the three-dimensional reconstruction of an object, Journal of Theoretical Biology",
        "cfg_override": {},
    },
    "cgls": {
        "name": "Conjugate Gradient Least Squares (CGLS)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cgls",
        "gpu": False,
        "reference": "Hestenes, M.R. & Stiefel, E. (1952) Methods of conjugate gradients for solving linear systems, J. Res. NBS",
        "cfg_override": {},
    },
    "sart": {
        "name": "Simultaneous ART (SART)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_sart",
        "gpu": False,
        "reference": "Andersen, A.H. & Kak, A.C. (1984) Simultaneous algebraic reconstruction technique (SART), Ultrasonic Imaging",
        "cfg_override": {},
    },
    # --- Classical: Statistical (2) ---
    "mlem": {
        "name": "ML-EM",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_mlem",
        "gpu": False,
        "reference": "Shepp, L.A. & Vardi, Y. (1982) Maximum likelihood reconstruction for emission tomography, IEEE TMI",
        "cfg_override": {},
    },
    "osem": {
        "name": "Ordered Subsets EM (OS-EM)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_osem",
        "gpu": False,
        "reference": "Hudson, H.M. & Larkin, R.S. (1994) Accelerated image reconstruction using ordered subsets of projection data, IEEE TMI",
        "cfg_override": {},
    },
    # --- Classical: Regularised inversions (6) ---
    "tikhonov": {
        "name": "Tikhonov Regularization",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov, A.N. (1963) Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady",
        "cfg_override": {},
    },
    "tv_admm": {
        "name": "TV-ADMM",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Sidky, E.Y., Kao, C.-M. & Pan, X. (2008) Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT, JXST",
        "cfg_override": {},
    },
    "chambolle_pock": {
        "name": "Chambolle-Pock Primal-Dual",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_chambolle_pock",
        "gpu": False,
        "reference": "Chambolle, A. & Pock, T. (2011) A first-order primal-dual algorithm for convex problems, J. Math. Imaging Vis.",
        "cfg_override": {},
    },
    "pnp_admm_nlm": {
        "name": "PnP-ADMM with NLM",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan, S. et al. (2013) Plug-and-Play priors for model-based reconstruction, IEEE GlobalSIP",
        "cfg_override": {},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA with NLM",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck, A. & Teboulle, M. (2009) A fast iterative shrinkage-thresholding algorithm, SIAM J. Imaging Sci. + PnP framework",
        "cfg_override": {},
    },
    "best_quality": {
        "name": "FDK + NLM Post-Processing",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_nlm",
        "gpu": False,
        "reference": "Buades, A., Coll, B. & Morel, J.-M. (2005) A non-local algorithm for image denoising, CVPR",
        "cfg_override": {},
    },
    # --- Deep Learning (5) ---
    "famous_dl": {
        "name": "FDK-DL (DL-PGD)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fdk_dl",
        "gpu": True,
        "reference": "Chen, H. et al. (2017) Low-dose CT with a residual encoder-decoder CNN, IEEE TMI",
        "cfg_override": {},
    },
    "small_gpu": {
        "name": "CBCT-UNet (DnCNN)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_unet",
        "gpu": True,
        "reference": "Jin, K.H. et al. (2017) Deep convolutional neural network for inverse problems in imaging, IEEE TIP",
        "cfg_override": {},
    },
    "cbct_diffusion": {
        "name": "CBCT Diffusion (DL-PGD)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_diffusion",
        "gpu": True,
        "reference": "Chung, H. et al. (2023) Solving 3D inverse problems using pre-trained 2D diffusion models, CVPR",
        "cfg_override": {},
    },
    "cbct_naf": {
        "name": "CBCT Neural Attenuation Fields (DL-DRS)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_naf",
        "gpu": True,
        "reference": "Zha, R. et al. (2024) NAF: Neural Attenuation Fields for sparse-view CBCT reconstruction, IEEE TMI",
        "cfg_override": {},
    },
    "cbct_mamba": {
        "name": "CBCT-Mamba (RED-DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_mamba",
        "gpu": True,
        "reference": "Wang, Z. et al. (2024) State-space models for efficient CT reconstruction, Medical Image Analysis",
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
    """FDK Ram-Lak. CPU only.
    Reference: Feldkamp, Davis & Kress (1984), JOSA A.
    """
    return run_solver("traditional_cpu", y, physics, cfg)


def run_best_quality(y: np.ndarray, physics: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK + NLM Post-Processing. CPU only.
    Reference: Buades, Coll & Morel (2005), CVPR.
    """
    return run_solver("best_quality", y, physics, cfg)


def run_famous_dl(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK-DL (DL-PGD). GPU required.
    Reference: Chen et al. (2017), IEEE TMI.
    """
    return run_solver("famous_dl", y, physics, cfg)


def run_small_gpu(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """CBCT-UNet (DnCNN). GPU required.
    Reference: Jin et al. (2017), IEEE TIP.
    """
    return run_solver("small_gpu", y, physics, cfg)
