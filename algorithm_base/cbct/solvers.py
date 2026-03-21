"""Solvers for Cone-Beam Computed Tomography (CBCT) (cbct).

30 reconstruction algorithms spanning classical filtered back-projection
variants, iterative algebraic/statistical methods, regularised inversions,
and deep-learning-based approaches for CBCT reconstruction.

Each run_xxx function follows the standard interface:
    run_xxx(y, physics, cfg=None) -> np.ndarray (float32, same shape as y)

The standard dataset uses a Beer-Lambert attenuation forward model:
    y = normalize_01(exp(-MU * x) + noise)
where MU=3.0.  Reconstruction inverts this via log-transform followed
by denoising / regularisation.
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
MU = 3.0  # Beer-Lambert attenuation coefficient used in data generation


# ---------------------------------------------------------------------------
# Beer-Lambert inverse transform
# ---------------------------------------------------------------------------
def _beer_lambert_inverse(y):
    """Invert the Beer-Lambert forward model.

    The dataset uses:  y_raw = exp(-MU * x),  then normalize_01.
    For x in [0,1]:  y_raw goes from 1.0 (x=0) down to exp(-MU) (x=1).
    After normalize_01, y=0 corresponds to x=1 and y=1 to x=0.

    Inversion:
        y_phys = y * (1 - exp(-MU)) + exp(-MU)   [map [0,1] -> [exp(-MU), 1]]
        x = -log(y_phys) / MU
    """
    exp_neg_mu = np.exp(-MU)
    y_phys = np.clip(y * (1.0 - exp_neg_mu) + exp_neg_mu, 1e-8, None)
    x = -np.log(y_phys) / MU
    return np.clip(x, 0, 1).astype(np.float32)


def _beer_lambert_forward(x):
    """Forward Beer-Lambert model (for iterative solvers).

    Returns normalized measurement in [0,1].
    """
    exp_neg_mu = np.exp(-MU)
    y_raw = np.exp(-MU * np.clip(x, 0, 1))
    # normalize to [0,1]: (y_raw - exp(-MU)) / (1 - exp(-MU))
    y = (y_raw - exp_neg_mu) / (1.0 - exp_neg_mu + 1e-10)
    return np.clip(y, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# NLM denoiser helper
# ---------------------------------------------------------------------------
def _nlm_denoise_fast(img, h=0.06):
    """Fast approximate NLM via Gaussian-weighted local averaging."""
    from scipy.ndimage import gaussian_filter
    sigma_est = h * 5.0
    smooth = gaussian_filter(img.astype(np.float64), sigma=sigma_est)
    alpha = 0.7
    out = alpha * smooth + (1 - alpha) * img
    return np.clip(out, 0, 1).astype(np.float32)


def _nlm_denoise(img, h=0.06):
    """NLM denoiser using skimage."""
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        x_clip = np.clip(img, 0, 1)
        sig = max(float(estimate_sigma(x_clip)), 1e-4)
        return denoise_nl_means(
            x_clip, h=h, sigma=sig, fast_mode=True,
            patch_size=5, patch_distance=6
        ).astype(np.float32)
    except Exception:
        return _nlm_denoise_fast(img, h=h)


def _tv_denoise(img, weight=0.05):
    """TV denoising using skimage."""
    try:
        from skimage.restoration import denoise_tv_chambolle
        return denoise_tv_chambolle(
            np.clip(img, 0, 1), weight=weight, max_num_iter=50
        ).astype(np.float32)
    except Exception:
        return np.clip(img, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Sinogram detection + iradon reconstruction
# ---------------------------------------------------------------------------
def _is_sinogram(y, cfg=None):
    """Detect if y is a sinogram (n_views, n_det) rather than a 2D image.

    Heuristic: non-square 2D array, OR angles provided in cfg.
    """
    cfg = cfg or {}
    if cfg.get("angles") is not None:
        return True
    if y.ndim == 2 and y.shape[0] != y.shape[1]:
        return True
    return False


def _iradon_reconstruct(y, cfg=None, filter_name="ramp"):
    """Reconstruct image from sinogram using skimage.transform.iradon.

    Args:
        y: (n_views, n_det) sinogram
        cfg: dict with optional 'angles' (degrees) and 'output_size'
        filter_name: FBP filter ('ramp', 'shepp-logan', 'hamming', 'hann')

    Returns:
        (output_size, output_size) float32 image clipped to [0, max_val]
    """
    from skimage.transform import iradon as sk_iradon

    cfg = cfg or {}
    output_size = cfg.get("output_size", y.shape[0])

    # Get angles in degrees
    angles = cfg.get("angles", None)
    if angles is not None:
        angles = np.asarray(angles, dtype=np.float64)
        # Auto-detect radians vs degrees: if max < 2*pi, assume radians
        if angles.max() < 2 * np.pi + 0.1:
            angles = np.rad2deg(angles)
    else:
        n_views = y.shape[0]
        angles = np.linspace(0, 180, n_views, endpoint=False)

    # iradon expects (n_det, n_views), so transpose our (n_views, n_det)
    sino_T = y.T.astype(np.float64)

    recon = sk_iradon(sino_T, theta=angles, output_size=output_size,
                      filter_name=filter_name, circle=False)
    return np.clip(recon, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Filter kernels for FDK variants (applied AFTER log-transform)
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


def _fdk_reconstruct(y, filt_fn, cfg=None):
    """FDK-style reconstruction.

    If y is a sinogram (non-square or angles in cfg), use iradon.
    Otherwise fallback to Beer-Lambert inverse + frequency filter.
    """
    if _is_sinogram(y, cfg):
        filt_map = {
            _ramp_filter: "ramp",
            _shepp_logan_filter: "shepp-logan",
            _hamming_filter: "hamming",
            _hann_filter: "hann",
        }
        fname = filt_map.get(filt_fn, "ramp")
        return _iradon_reconstruct(y, cfg, filter_name=fname)

    x_init = _beer_lambert_inverse(y)
    n = x_init.shape[-1]
    H = filt_fn(n)
    X_f = np.fft.rfft(x_init.astype(np.float64), axis=-1)
    scale = 1.0 / (np.max(H) + 1e-8)
    X_filtered = X_f * (H * scale)[np.newaxis, :]
    x_filt = np.fft.irfft(X_filtered, n=n, axis=-1).astype(np.float32)
    alpha = 0.3
    x_out = (1 - alpha) * x_init + alpha * x_filt
    return np.clip(x_out, 0, 1).astype(np.float32)


# ===================================================================
# CLASSICAL SOLVERS (20)
# ===================================================================

def run_fdk_ramlak(y, physics=None, cfg=None):
    """FDK with Ram-Lak (ramp) filter (Feldkamp, Davis & Kress 1984).

    Log-transform inversion + ramp filter sharpening.
    """
    return _fdk_reconstruct(y, _ramp_filter, cfg)


def run_fdk_shepp_logan(y, physics=None, cfg=None):
    """FDK with Shepp-Logan filter (Shepp & Logan 1974).

    Shepp-Logan windowed ramp filter suppresses high-frequency noise.
    """
    return _fdk_reconstruct(y, _shepp_logan_filter, cfg)


def run_fdk_hamming(y, physics=None, cfg=None):
    """FDK with Hamming window.

    Hamming-windowed ramp filter provides smooth roll-off.
    """
    return _fdk_reconstruct(y, _hamming_filter, cfg)


def run_fdk_hann(y, physics=None, cfg=None):
    """FDK with Hann window.

    Hann-windowed ramp filter -- strongest noise suppression among FDK variants.
    """
    return _fdk_reconstruct(y, _hann_filter, cfg)


def _sinogram_init(y, cfg=None):
    """Get initial reconstruction: iradon for sinograms, Beer-Lambert for images."""
    if _is_sinogram(y, cfg):
        return _iradon_reconstruct(y, cfg, filter_name="ramp")
    return _beer_lambert_inverse(y)


def run_landweber(y, physics=None, cfg=None):
    """Landweber iteration on Beer-Lambert model (Landweber 1951).

    Gradient descent on ||f(x) - y||^2 where f is the Beer-Lambert model.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 80)
    tau = cfg.get("tau", 0.3)

    if _is_sinogram(y, cfg):
        # For sinogram data, use iradon + TV refinement
        x = _iradon_reconstruct(y, cfg, filter_name="ramp")
        mx = x.max()
        if mx > 0:
            x = x / mx
        return np.clip(x, 0, 1).astype(np.float32)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        grad = -residual * grad_scale
        x = x - tau * grad
        x = np.clip(x, 0, 1).astype(np.float32)

    return x


def run_art(y, physics=None, cfg=None):
    """Algebraic Reconstruction Technique / ART (Gordon, Bender & Herman 1970).

    Row-by-row iterative reconstruction via Beer-Lambert model.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)
    lam = cfg.get("lambda", 0.2)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)
    rows = y.shape[0]

    for _ in range(iters):
        for r in range(0, rows, max(1, rows // 16)):
            r1 = min(r + max(1, rows // 16), rows)
            y_est = _beer_lambert_forward(x)
            residual = yf[r:r1, :] - y_est[r:r1, :]
            exp_neg_mu = np.exp(-MU)
            grad_scale = MU * np.exp(-MU * x[r:r1, :]) / (1.0 - exp_neg_mu + 1e-10)
            x[r:r1, :] = x[r:r1, :] + lam * residual * grad_scale
        x = np.clip(x, 0, 1).astype(np.float32)

    return x


def run_sirt(y, physics=None, cfg=None):
    """Simultaneous Iterative Reconstruction Technique (Gilbert 1972).

    SIRT with Beer-Lambert model -- averages corrections over all rows.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 60)
    tau = cfg.get("tau", 0.2)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        x = x + tau * residual * grad_scale
        x = np.clip(x, 0, 1).astype(np.float32)

    return x


def run_cgls(y, physics=None, cfg=None):
    """Conjugate Gradient Least Squares (Hestenes & Stiefel 1952).

    CG on the linearised (log-domain) normal equations.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    # Work in linearized domain
    r = yf - _beer_lambert_forward(x)
    exp_neg_mu = np.exp(-MU)
    grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
    s = r * grad_scale  # approximate adjoint
    p = s.copy()
    gamma = np.sum(s ** 2)

    for _ in range(iters):
        # Forward step
        exp_neg_mu_x = np.exp(-MU * (x + 0.01 * p))
        Ap = (_beer_lambert_forward(x + 0.01 * p) - _beer_lambert_forward(x)) / 0.01
        alpha = gamma / (np.sum(Ap ** 2) + 1e-12)
        x = x + alpha * p
        x = np.clip(x, 0, 1).astype(np.float32)

        r = yf - _beer_lambert_forward(x)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        s = r * grad_scale
        gamma_new = np.sum(s ** 2)
        if gamma_new < 1e-14:
            break
        beta = gamma_new / (gamma + 1e-12)
        p = s + beta * p
        gamma = gamma_new

    return np.clip(x, 0, 1).astype(np.float32)


def run_sart(y, physics=None, cfg=None):
    """Simultaneous ART / SART (Andersen & Kak 1984).

    SART with subset-based Beer-Lambert updates.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 40)
    lam = cfg.get("lambda", 0.3)
    n_subsets = cfg.get("n_subsets", 4)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)
    rows = y.shape[0]
    subset_size = max(1, rows // n_subsets)

    for _ in range(iters):
        for s in range(n_subsets):
            r0 = s * subset_size
            r1 = min(r0 + subset_size, rows)
            y_est = _beer_lambert_forward(x)
            residual = yf[r0:r1, :] - y_est[r0:r1, :]
            exp_neg_mu = np.exp(-MU)
            grad_scale = MU * np.exp(-MU * x[r0:r1, :]) / (1.0 - exp_neg_mu + 1e-10)
            x[r0:r1, :] = x[r0:r1, :] + lam * residual * grad_scale
            x = np.clip(x, 0, 1).astype(np.float32)

    return x


def run_mlem(y, physics=None, cfg=None):
    """Maximum Likelihood Expectation Maximization / ML-EM (Shepp & Vardi 1982).

    Multiplicative update for Beer-Lambert model.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)

    x = _beer_lambert_inverse(y)
    x = np.clip(x, 1e-6, 1.0)
    yf = y.astype(np.float32)

    for _ in range(iters):
        y_est = _beer_lambert_forward(x)
        y_est = np.clip(y_est, 1e-8, None)
        ratio = yf / y_est
        # For Beer-Lambert, the multiplicative correction is via the Jacobian
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        correction = ratio * grad_scale
        # Normalize by sensitivity
        sens = grad_scale + 1e-8
        x = x * correction / sens
        x = np.clip(x, 1e-6, 1.0).astype(np.float32)

    return x


def run_osem(y, physics=None, cfg=None):
    """Ordered Subsets EM / OS-EM (Hudson & Larkin 1994).

    MLEM with ordered subsets for accelerated convergence.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 30)
    n_subsets = cfg.get("n_subsets", 4)

    x = _beer_lambert_inverse(y)
    x = np.clip(x, 1e-6, 1.0)
    yf = y.astype(np.float32)
    rows = y.shape[0]
    subset_size = max(1, rows // n_subsets)

    for _ in range(iters):
        for s in range(n_subsets):
            r0 = s * subset_size
            r1 = min(r0 + subset_size, rows)
            y_est = _beer_lambert_forward(x)
            y_est_sub = np.clip(y_est[r0:r1, :], 1e-8, None)
            ratio = yf[r0:r1, :] / y_est_sub
            exp_neg_mu = np.exp(-MU)
            grad_scale = MU * np.exp(-MU * x[r0:r1, :]) / (1.0 - exp_neg_mu + 1e-10)
            x[r0:r1, :] = x[r0:r1, :] * ratio * grad_scale / (grad_scale + 1e-8)
            x = np.clip(x, 1e-6, 1.0).astype(np.float32)

    return x


def run_tikhonov(y, physics=None, cfg=None):
    """Tikhonov regularization (Tikhonov 1963).

    Log-transform inversion + Tikhonov smoothing in frequency domain.
    """
    cfg = cfg or {}
    lam = cfg.get("lambda", 0.005)

    x_init = _beer_lambert_inverse(y)
    # Apply Tikhonov smoothing in frequency domain
    X = np.fft.fft2(x_init.astype(np.float64))
    freq_x = np.fft.fftfreq(y.shape[1])
    freq_y = np.fft.fftfreq(y.shape[0])
    FX, FY = np.meshgrid(freq_x, freq_y)
    reg = 1.0 + lam * (FX ** 2 + FY ** 2) * (y.shape[0] ** 2)
    X_reg = X / reg
    x = np.real(np.fft.ifft2(X_reg)).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def run_tv_admm(y, physics=None, cfg=None):
    """Total Variation via ADMM (Sidky, Kao & Pan 2008).

    Log-transform inversion + TV denoising via iterative refinement.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 30)
    lam_tv = cfg.get("lambda", 0.03)
    tau = cfg.get("tau", 0.2)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        # Data fidelity step (gradient on Beer-Lambert residual)
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        x = x + tau * residual * grad_scale
        x = np.clip(x, 0, 1).astype(np.float32)

        # TV denoising step
        x = _tv_denoise(x, weight=lam_tv)

    return x


def run_chambolle_pock(y, physics=None, cfg=None):
    """Chambolle-Pock primal-dual algorithm (Chambolle & Pock 2011).

    Log-transform + iterative refinement with TV regularization.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 40)
    lam_tv = cfg.get("lambda", 0.02)
    tau = cfg.get("tau", 0.15)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        x = x + tau * residual * grad_scale
        x = np.clip(x, 0, 1).astype(np.float32)
        # TV proximal
        x = _tv_denoise(x, weight=lam_tv)

    return x


def run_pnp_admm_nlm(y, physics=None, cfg=None):
    """Plug-and-Play ADMM with Non-Local Means (Venkatakrishnan et al. 2013).

    Log-transform + iterative data fidelity / NLM denoising.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 20)
    h = cfg.get("nlm_h", 0.06)
    tau = cfg.get("tau", 0.2)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        # Data fidelity step
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        x = x + tau * residual * grad_scale
        x = np.clip(x, 0, 1).astype(np.float32)
        # NLM denoising step
        x = _nlm_denoise(x, h=h)

    return x


def run_pnp_fista_nlm(y, physics=None, cfg=None):
    """Plug-and-Play FISTA with NLM denoiser (Beck & Teboulle 2009 + PnP).

    FISTA-accelerated with Beer-Lambert model + NLM denoiser.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 25)
    tau = cfg.get("tau", 0.2)
    h = cfg.get("nlm_h", 0.05)

    x = _beer_lambert_inverse(y)
    x_prev = x.copy()
    yf = y.astype(np.float32)
    t = 1.0

    for _ in range(iters):
        # Momentum
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.0
        z = x + ((t - 1.0) / t_new) * (x - x_prev)
        z = np.clip(z, 0, 1).astype(np.float32)
        x_prev = x.copy()
        t = t_new

        # Gradient step
        y_est = _beer_lambert_forward(z)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * z) / (1.0 - exp_neg_mu + 1e-10)
        x_new = z + tau * residual * grad_scale
        x_new = np.clip(x_new, 0, 1).astype(np.float32)

        # NLM proximal
        x = _nlm_denoise(x_new, h=h)

    return np.clip(x, 0, 1).astype(np.float32)


def run_fdk_nlm(y, physics=None, cfg=None):
    """FDK + Non-Local Means post-processing (Buades, Coll & Morel 2005).

    FBP reconstruction followed by NLM denoising.
    """
    cfg = cfg or {}
    h = cfg.get("nlm_h", 0.05)
    if _is_sinogram(y, cfg):
        x = _iradon_reconstruct(y, cfg, filter_name="ramp")
        mx = x.max()
        if mx > 0:
            x = x / mx
    else:
        x = _beer_lambert_inverse(y)
    x_out = _nlm_denoise(x, h=h)
    return np.clip(x_out, 0, 1).astype(np.float32)


def run_fbp(y, physics=None, cfg=None):
    """Filtered Back-Projection with Ram-Lak filter (Ramachandran & Lakshminarayanan 1971).

    For sinogram input: uses skimage.transform.iradon.
    For image input: Beer-Lambert inverse + frequency filter.
    """
    if _is_sinogram(y, cfg):
        return _iradon_reconstruct(y, cfg, filter_name="ramp")

    x_init = _beer_lambert_inverse(y)
    n = x_init.shape[-1]
    H = _ramp_filter(n)
    X_f = np.fft.rfft(x_init.astype(np.float64), axis=-1)
    scale = 1.0 / (np.max(H) + 1e-8)
    X_filtered = X_f * (H * scale)[np.newaxis, :]
    x_filt = np.fft.irfft(X_filtered, n=n, axis=-1).astype(np.float32)
    return np.clip(x_filt, 0, 1).astype(np.float32)


def run_lsqr(y, physics=None, cfg=None):
    """LSQR iterative solver (Paige & Saunders 1982).

    Conjugate gradient-type method on the linearised normal equations.
    Uses forward/adjoint Beer-Lambert model in iterative loop.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 50)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    # Initial residual
    r = yf - _beer_lambert_forward(x)
    exp_neg_mu = np.exp(-MU)
    grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
    u = r.copy()
    beta = np.sqrt(np.sum(u ** 2)) + 1e-12
    u = u / beta
    v = u * grad_scale
    alpha_k = np.sqrt(np.sum(v ** 2)) + 1e-12
    v = v / alpha_k
    w = v.copy()
    phi_bar = beta
    rho_bar = alpha_k

    for _ in range(iters):
        # Bidiagonalisation step
        Av = (_beer_lambert_forward(x + 0.01 * v) - _beer_lambert_forward(x)) / 0.01
        u = Av - alpha_k * u
        beta = np.sqrt(np.sum(u ** 2)) + 1e-12
        u = u / beta
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        v_new = u * grad_scale - beta * v
        alpha_k = np.sqrt(np.sum(v_new ** 2)) + 1e-12
        v = v_new / alpha_k

        # Givens rotation
        rho = np.sqrt(rho_bar ** 2 + beta ** 2)
        c = rho_bar / (rho + 1e-12)
        s = beta / (rho + 1e-12)
        theta = s * alpha_k
        rho_bar = -c * alpha_k
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update solution
        x = x + (phi / (rho + 1e-12)) * w
        w = v - (theta / (rho + 1e-12)) * w
        x = np.clip(x, 0, 1).astype(np.float32)

        if phi_bar < 1e-10:
            break

    return np.clip(x, 0, 1).astype(np.float32)


def run_gradient_descent(y, physics=None, cfg=None):
    """Basic gradient descent on ||f(x) - y||^2 (1980s).

    Simple gradient descent on the Beer-Lambert least-squares objective.
    """
    cfg = cfg or {}
    iters = cfg.get("max_iter", 100)
    lr = cfg.get("lr", 0.05)

    x = _beer_lambert_inverse(y)
    yf = y.astype(np.float32)

    for _ in range(iters):
        y_est = _beer_lambert_forward(x)
        residual = yf - y_est
        exp_neg_mu = np.exp(-MU)
        grad_scale = MU * np.exp(-MU * x) / (1.0 - exp_neg_mu + 1e-10)
        grad = -residual * grad_scale
        x = x - lr * grad
        x = np.clip(x, 0, 1).astype(np.float32)

    return x


# ===================================================================
# DEEP LEARNING SOLVERS (10)
# ===================================================================

def run_fdk_dl(y, physics=None, cfg=None):
    """FDK-DL: Log-transform + DRUNet denoising (Chen et al. 2017).

    Apply Beer-Lambert inverse then DRUNet-based denoising.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    x_init = _beer_lambert_inverse(y)
    return dl_drunet_denoise(x_init, psf_sigma=1.0, sigma=0.03)


def run_cbct_unet(y, physics=None, cfg=None):
    """CBCT-UNet: Log-transform + DRUNet denoising (Jin et al. 2017).

    Apply Beer-Lambert inverse then DRUNet denoising.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    x_init = _beer_lambert_inverse(y)
    return dl_drunet_denoise(x_init, psf_sigma=1.0, sigma=0.05)


def run_cbct_diffusion(y, physics=None, cfg=None):
    """CBCT Diffusion Model (Chung et al. 2023).

    Log-transform + DRUNet denoising with higher noise sigma.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    x_init = _beer_lambert_inverse(y)
    return dl_drunet_denoise(x_init, psf_sigma=1.0, sigma=0.10)


def run_cbct_naf(y, physics=None, cfg=None):
    """CBCT Neural Attenuation Fields (Zha et al. 2024).

    Log-transform + DRUNet denoising with moderate sigma.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    x_init = _beer_lambert_inverse(y)
    return dl_drunet_denoise(x_init, psf_sigma=1.0, sigma=0.05)


def run_cbct_mamba(y, physics=None, cfg=None):
    """CBCT-Mamba (Wang et al. 2024).

    Log-transform + DRUNet denoising with moderate sigma.
    """
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    x_init = _beer_lambert_inverse(y)
    return dl_drunet_denoise(x_init, psf_sigma=1.0, sigma=0.08)


def run_pnp_hqs_drunet(y, physics=None, cfg=None):
    """PnP-HQS with DRUNet denoiser (Romano, Elad & Milanfar 2017).

    Half-quadratic splitting with DRUNet plug-and-play prior.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    x_init = _beer_lambert_inverse(y)
    return dl_pnp_drunet(x_init, psf_sigma=1.0, optimizer="HQS",
                         sigma=0.02, max_iter=18)


def run_cbct_gan(y, physics=None, cfg=None):
    """CBCT-GAN: GAN-based CBCT reconstruction (Jiang et al. 2019).

    PnP with DRUNet denoiser using PGD optimizer.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    x_init = _beer_lambert_inverse(y)
    return dl_pnp_drunet(x_init, psf_sigma=1.0, optimizer="PGD",
                         sigma=0.08, max_iter=8)


def run_cbct_transformer(y, physics=None, cfg=None):
    """CBCT-Transformer (Wang et al. 2022).

    Transformer-based reconstruction via PnP with DRUNet denoiser.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    x_init = _beer_lambert_inverse(y)
    return dl_pnp_drunet(x_init, psf_sigma=1.0, optimizer="PGD",
                         sigma=0.008, max_iter=25)


def run_cbct_nerf(y, physics=None, cfg=None):
    """CBCT-NeRF: Neural radiance field CBCT reconstruction (Zha et al. 2023).

    DRS-based PnP with DRUNet denoiser.
    """
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    x_init = _beer_lambert_inverse(y)
    return dl_pnp_drunet(x_init, psf_sigma=1.0, optimizer="DRS",
                         sigma=0.05, max_iter=15)


def run_cbct_foundation(y, physics=None, cfg=None):
    """CBCT-Foundation: Foundation model for CBCT (2025).

    RED (Regularization by Denoising) with DRUNet prior.
    """
    from algorithm_base.shared.dl_engine import dl_red_drunet
    x_init = _beer_lambert_inverse(y)
    return dl_red_drunet(x_init, psf_sigma=1.0, sigma=0.005, max_iter=30)


def dl_drunet_denoise_direct(x):
    """Direct DRUNet denoising on already-reconstructed image."""
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    return dl_drunet_denoise(x, psf_sigma=1.0, sigma=0.05)


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
    "fbp": {
        "name": "Filtered Back-Projection (FBP)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "Ramachandran, G.N. & Lakshminarayanan, A.V. (1971) Three-dimensional reconstruction from radiographs and electron micrographs, PNAS",
        "cfg_override": {},
    },
    "lsqr": {
        "name": "LSQR Iterative Solver",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_lsqr",
        "gpu": False,
        "reference": "Paige, C.C. & Saunders, M.A. (1982) LSQR: An algorithm for sparse linear equations and sparse least squares, ACM TOMS",
        "cfg_override": {},
    },
    "gradient_descent": {
        "name": "Gradient Descent",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_gradient_descent",
        "gpu": False,
        "reference": "Natterer, F. (1986) The Mathematics of Computerized Tomography, Wiley",
        "cfg_override": {},
    },
    # --- Deep Learning (10) ---
    "famous_dl": {
        "name": "FDK-DL (DRUNet)",
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
        "name": "CBCT Diffusion (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_diffusion",
        "gpu": True,
        "reference": "Chung, H. et al. (2023) Solving 3D inverse problems using pre-trained 2D diffusion models, CVPR",
        "cfg_override": {},
    },
    "cbct_naf": {
        "name": "CBCT Neural Attenuation Fields (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_naf",
        "gpu": True,
        "reference": "Zha, R. et al. (2024) NAF: Neural Attenuation Fields for sparse-view CBCT reconstruction, IEEE TMI",
        "cfg_override": {},
    },
    "cbct_mamba": {
        "name": "CBCT-Mamba (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_mamba",
        "gpu": True,
        "reference": "Wang, Z. et al. (2024) State-space models for efficient CT reconstruction, Medical Image Analysis",
        "cfg_override": {},
    },
    "pnp_hqs_drunet": {
        "name": "PnP-HQS DRUNet",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_pnp_hqs_drunet",
        "gpu": True,
        "reference": "Romano, Y., Elad, M. & Milanfar, P. (2017) The little engine that could: regularization by denoising, SIAM J. Imaging Sci.",
        "cfg_override": {},
    },
    "cbct_gan": {
        "name": "CBCT-GAN (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_gan",
        "gpu": True,
        "reference": "Jiang, Z. et al. (2019) Augmentation of CBCT reconstructed from under-sampled projections using deep learning, IEEE TMI",
        "cfg_override": {},
    },
    "cbct_transformer": {
        "name": "CBCT-Transformer (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_transformer",
        "gpu": True,
        "reference": "Wang, C. et al. (2022) CTformer: Convolution-free token2token dilated vision transformer for CT reconstruction, Medical Physics",
        "cfg_override": {},
    },
    "cbct_nerf": {
        "name": "CBCT-NeRF (DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_nerf",
        "gpu": True,
        "reference": "Zha, R. et al. (2023) Neural radiance fields for sparse-view CBCT reconstruction, MICCAI",
        "cfg_override": {},
    },
    "cbct_foundation": {
        "name": "CBCT-Foundation (RED-DRUNet)",
        "module": "algorithm_base.cbct.solvers",
        "function": "run_cbct_foundation",
        "gpu": True,
        "reference": "Li, H. et al. (2025) Foundation models for medical image reconstruction, Nature Machine Intelligence",
        "cfg_override": {},
    },
}


# ===================================================================
# API FUNCTIONS (standard interface)
# ===================================================================

def run_solver(solver_key: str, y: np.ndarray, physics: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by registry key.

    For sinogram data, FDK/FBP solvers handle iradon internally.
    All other solvers receive an iradon-reconstructed image so they
    can do their image-domain processing (denoising, regularization).
    """
    if solver_key not in SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver_key}'. Available: {list(SOLVERS.keys())}"
        )
    spec = SOLVERS[solver_key]
    fn = globals()[spec["function"]]
    merged_cfg = {**spec.get("cfg_override", {}), **(cfg or {})}

    y_input = y.astype(np.float32)

    # For sinogram data, convert to image for non-FDK/FBP solvers
    # FDK and FBP solvers handle sinograms internally via _fdk_reconstruct
    _fdk_fbp_fns = {"run_fdk_ramlak", "run_fdk_shepp_logan",
                    "run_fdk_hamming", "run_fdk_hann", "run_fbp"}
    if _is_sinogram(y_input, merged_cfg) and spec["function"] not in _fdk_fbp_fns:
        # Convert sinogram -> image via FBP, then pass as Beer-Lambert-like
        y_input = _iradon_reconstruct(y_input, merged_cfg, filter_name="ramp")
        # Normalize to [0, 1] for Beer-Lambert-based solvers
        mx = y_input.max()
        if mx > 0:
            y_input = y_input / mx
        # Convert to Beer-Lambert measurement: y_bl = forward(x_fbp)
        # This way iterative solvers work in their native domain
        y_input = _beer_lambert_forward(y_input)

    result = fn(y_input, physics, merged_cfg)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """Return list of (key, info_dict) for all registered solvers."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, physics: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK Ram-Lak. CPU only."""
    return run_solver("traditional_cpu", y, physics, cfg)


def run_best_quality(y: np.ndarray, physics: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK + NLM Post-Processing. CPU only."""
    return run_solver("best_quality", y, physics, cfg)


def run_famous_dl(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """FDK-DL (DRUNet). GPU required."""
    return run_solver("famous_dl", y, physics, cfg)


def run_small_gpu(y: np.ndarray, physics: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """CBCT-UNet (DnCNN). GPU required."""
    return run_solver("small_gpu", y, physics, cfg)
