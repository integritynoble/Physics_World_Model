"""Lensless Imaging Reconstruction: ADMM-TV.

ADMM with Total Variation prior for lensless camera reconstruction.

References:
- Antipa, N. et al. (2018). "DiffuserCam: lensless single-exposure 3D imaging",
  Optica.

Benchmark:
- PSNR: ~25-28 dB on DiffuserCam data
- CPU-only, no GPU needed
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


# ---------------------------------------------------------------------------
# Forward / adjoint helpers
# ---------------------------------------------------------------------------

def lensless_forward(
    image: np.ndarray,
    psf: np.ndarray,
) -> np.ndarray:
    """Forward model: convolution with PSF.

    y = conv(x, psf) = ifft(fft(x) * fft(psf))

    Args:
        image: Input image (H, W)
        psf: Point spread function (H, W), same size as image

    Returns:
        Measurement (H, W) -- real-valued
    """
    X = fft2(image)
    H = fft2(psf)
    y = np.real(ifft2(X * H))
    return y.astype(np.float32)


def lensless_adjoint(
    measurement: np.ndarray,
    psf: np.ndarray,
) -> np.ndarray:
    """Adjoint model: correlation with PSF.

    x_adj = ifft(fft(y) * conj(fft(psf)))

    Args:
        measurement: Measurement (H, W)
        psf: Point spread function (H, W)

    Returns:
        Adjoint image (H, W) -- real-valued
    """
    Y = fft2(measurement)
    H_conj = np.conj(fft2(psf))
    x = np.real(ifft2(Y * H_conj))
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# TV proximal operator (2D, Chambolle dual)
# ---------------------------------------------------------------------------

def _tv_prox_2d(
    x: np.ndarray,
    lam: float,
    iterations: int = 20,
) -> np.ndarray:
    """Proximal operator for isotropic TV (Chambolle's algorithm).

    Reuses the same logic as ``cs_solvers.tv_prox_2d`` but is kept local so
    this module has no intra-package dependencies.

    Args:
        x: Input image (H, W)
        lam: Regularization strength
        iterations: Number of dual iterations

    Returns:
        TV-denoised image (H, W)
    """
    h, w = x.shape[:2]
    p = np.zeros((h, w, 2), dtype=np.float64)
    tau = 0.25

    for _ in range(iterations):
        # Divergence of p
        div_p = np.zeros((h, w), dtype=np.float64)
        div_p[:, :-1] += p[:, :-1, 0]
        div_p[:, 1:]  -= p[:, :-1, 0]
        div_p[:-1, :] += p[:-1, :, 1]
        div_p[1:, :]  -= p[:-1, :, 1]

        # Gradient of (x - lam * div(p))
        u = x - lam * div_p
        grad = np.zeros((h, w, 2), dtype=np.float64)
        grad[:, :-1, 0] = u[:, 1:] - u[:, :-1]
        grad[:-1, :, 1] = u[1:, :] - u[:-1, :]

        # Dual update + projection onto unit ball
        p_new = p + tau * grad
        norm = np.sqrt(p_new[:, :, 0] ** 2 + p_new[:, :, 1] ** 2 + 1e-10)
        norm = np.maximum(norm, 1.0)
        p = p_new / norm[:, :, np.newaxis]

    # Final primal from dual
    div_p = np.zeros((h, w), dtype=np.float64)
    div_p[:, :-1] += p[:, :-1, 0]
    div_p[:, 1:]  -= p[:, :-1, 0]
    div_p[:-1, :] += p[:-1, :, 1]
    div_p[1:, :]  -= p[:-1, :, 1]

    return (x - lam * div_p).astype(np.float32)


# ---------------------------------------------------------------------------
# Tikhonov / Wiener deconvolution
# ---------------------------------------------------------------------------

def tikhonov_lensless(
    measurement: np.ndarray,
    psf: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    """Tikhonov (Wiener) deconvolution for lensless imaging.

    Closed-form solution in Fourier domain:
        x = ifft( conj(H) * Y / (|H|^2 + reg) )

    Args:
        measurement: Measurement image (H, W)
        psf: Point spread function (H, W)
        reg: Tikhonov regularization parameter

    Returns:
        Reconstructed image (H, W)
    """
    Y = fft2(measurement)
    H = fft2(psf)
    H_conj = np.conj(H)

    X = H_conj * Y / (np.abs(H) ** 2 + reg)
    x = np.real(ifft2(X))
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# ADMM-TV solver
# ---------------------------------------------------------------------------

def admm_tv_lensless(
    measurement: np.ndarray,
    psf: np.ndarray,
    rho: float = 1.0,
    lam_tv: float = 0.01,
    iters: int = 100,
    tv_inner_iters: int = 20,
    non_negative: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """ADMM with Total Variation prior for lensless reconstruction.

    Solves:  min_x  0.5 ||Hx - y||^2  +  lam_tv * TV(x)

    ADMM splitting introduces auxiliary variable z = x and dual variable u:
        x-update : Wiener filter  (closed-form in Fourier domain)
        z-update : TV proximal operator (Chambolle)
        u-update : u <- u + x - z

    Args:
        measurement: Lensless measurement (H, W)
        psf: Point spread function (H, W), same spatial size
        rho: ADMM penalty parameter (augmented Lagrangian)
        lam_tv: TV regularization weight
        iters: Number of ADMM outer iterations
        tv_inner_iters: Inner iterations for TV proximal operator
        non_negative: Enforce non-negativity constraint
        verbose: Print convergence info every 10 iterations

    Returns:
        Reconstructed image (H, W)
    """
    measurement = measurement.astype(np.float64)
    psf = psf.astype(np.float64)

    # Pre-compute Fourier quantities (constant across iterations)
    Y = fft2(measurement)
    H = fft2(psf)
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2

    # Denominator for x-update: |H|^2 + rho
    denom = H_abs2 + rho

    # Initialize variables
    x = np.real(ifft2(H_conj * Y / (H_abs2 + 1e-3))).astype(np.float64)  # Wiener init
    z = x.copy()
    u = np.zeros_like(x)

    for k in range(iters):
        # ---- x-update (Wiener filter in Fourier domain) -----------------
        # x = argmin_x 0.5||Hx - y||^2 + (rho/2)||x - (z - u)||^2
        # => (H^H H + rho I) x = H^H y + rho (z - u)
        rhs = H_conj * Y + rho * fft2(z - u)
        X = rhs / denom
        x = np.real(ifft2(X))

        # ---- z-update (TV proximal) --------------------------------------
        # z = prox_{lam_tv/rho * TV}(x + u)
        v = x + u
        z = _tv_prox_2d(v, lam_tv / rho, iterations=tv_inner_iters)

        if non_negative:
            z = np.maximum(z, 0.0)

        # ---- u-update (dual ascent) --------------------------------------
        u = u + x - z

        if verbose and (k + 1) % 10 == 0:
            # Data fidelity
            residual = np.real(ifft2(H * fft2(x))) - measurement
            data_fit = 0.5 * np.sum(residual ** 2)
            primal_res = np.linalg.norm(x - z)
            print(
                f"ADMM iter {k+1:4d} | data_fit={data_fit:.4e} "
                f"| primal_res={primal_res:.4e}"
            )

    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio-style wrapper
# ---------------------------------------------------------------------------

def run_lensless(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run lensless imaging reconstruction.

    Dispatches to ADMM-TV (default) or Tikhonov depending on config.

    Args:
        y: Lensless measurement (H, W)
        physics: Lensless physics operator (must expose ``.psf`` attribute
                 or ``forward``/``adjoint`` methods)
        cfg: Configuration dict with:
            - solver: 'admm_tv' (default) or 'tikhonov'
            - rho: ADMM penalty (default: 1.0)
            - lam_tv: TV weight (default: 0.01)
            - iters: Outer ADMM iterations (default: 100)
            - tv_inner_iters: Inner TV iterations (default: 20)
            - reg: Tikhonov regularization (default: 1e-3)
            - non_negative: Enforce x >= 0 (default: True)

    Returns:
        Tuple of (reconstructed image, info_dict)
    """
    solver = cfg.get("solver", "admm_tv")
    rho = cfg.get("rho", 1.0)
    lam_tv = cfg.get("lam_tv", 0.01)
    iters = cfg.get("iters", 100)
    tv_inner_iters = cfg.get("tv_inner_iters", 20)
    reg = cfg.get("reg", 1e-3)
    non_negative = cfg.get("non_negative", True)

    info: Dict[str, Any] = {
        "solver": solver,
    }

    try:
        # ---- Obtain PSF -------------------------------------------------
        psf: Optional[np.ndarray] = None

        if hasattr(physics, "psf"):
            psf = np.asarray(physics.psf, dtype=np.float64)
        elif hasattr(physics, "info"):
            op_info = physics.info()
            if "psf" in op_info:
                psf = np.asarray(op_info["psf"], dtype=np.float64)

        if psf is None:
            # Cannot reconstruct without a PSF -- fall back to adjoint
            info["error"] = "psf_not_found"
            if hasattr(physics, "adjoint"):
                return physics.adjoint(y).astype(np.float32), info
            return y.astype(np.float32), info

        # Ensure measurement and PSF have matching spatial dimensions
        if y.ndim == 3:
            # Multi-channel: reconstruct each channel independently
            results = []
            for c in range(y.shape[2]):
                if solver == "tikhonov":
                    results.append(tikhonov_lensless(y[:, :, c], psf, reg))
                else:
                    results.append(
                        admm_tv_lensless(
                            y[:, :, c], psf,
                            rho=rho, lam_tv=lam_tv, iters=iters,
                            tv_inner_iters=tv_inner_iters,
                            non_negative=non_negative,
                        )
                    )
            result = np.stack(results, axis=2).astype(np.float32)
        elif y.ndim == 2:
            if solver == "tikhonov":
                result = tikhonov_lensless(y, psf, reg)
                info["reg"] = reg
            else:
                result = admm_tv_lensless(
                    y, psf,
                    rho=rho, lam_tv=lam_tv, iters=iters,
                    tv_inner_iters=tv_inner_iters,
                    non_negative=non_negative,
                )
                info["rho"] = rho
                info["lam_tv"] = lam_tv
                info["iters"] = iters
        else:
            info["error"] = "unexpected_input_ndim"
            return y.astype(np.float32), info

        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, "adjoint"):
            return physics.adjoint(y).astype(np.float32), info
        return y.astype(np.float32), info
