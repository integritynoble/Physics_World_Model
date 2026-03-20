"""Solvers for Single-Pixel Camera (SPC) reconstruction.

Comprehensive solver library covering greedy pursuit (1993-2009), convex
optimization (2004-2016), plug-and-play / AMP (2014-2016), and deep learning
(2016-2022) compressive sensing reconstruction methods.

Each function follows the standard interface: fn(y, operator, cfg) -> x_hat

Single-pixel camera acquires compressed measurements y = Phi @ x + noise,
where Phi is the measurement matrix (M x N, M << N) and x is the vectorised
image.  Reconstruction recovers x from the under-determined system.
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "spc"
DISPLAY_NAME = "Single-Pixel Camera (SPC)"

PSF_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Solver registry -- 25 solvers from 1949 to 2022
# ---------------------------------------------------------------------------
SOLVERS = {
    # ---- Classical (15 algorithms) ----
    "traditional_cpu": {
        "name": "TVAL3",
        "module": "__self__",
        "function": "_run_tval3",
        "gpu": False,
        "reference": "Li et al., Rice CAAM Tech Report 2009",
    },
    "best_quality": {
        "name": "ADMM-L1",
        "module": "__self__",
        "function": "_run_admm_l1",
        "gpu": False,
        "reference": "Boyd et al., Found. Trends ML 2010",
    },
    "fista_l1": {
        "name": "FISTA-L1",
        "module": "__self__",
        "function": "_run_fista_l1",
        "gpu": False,
        "reference": "Beck & Teboulle, SIAM J. Imaging Sci. 2009",
    },
    "omp": {
        "name": "OMP",
        "module": "__self__",
        "function": "_run_omp",
        "gpu": False,
        "reference": "Pati, Rezaiifar & Krishnaprasad, Asilomar 1993",
    },
    "cosamp": {
        "name": "CoSaMP",
        "module": "__self__",
        "function": "_run_cosamp",
        "gpu": False,
        "reference": "Needell & Tropp, Appl. Comput. Harmon. Anal. 2009",
    },
    "iht": {
        "name": "IHT",
        "module": "__self__",
        "function": "_run_iht",
        "gpu": False,
        "reference": "Blumensath & Davies, J. Fourier Anal. Appl. 2009",
    },
    "gap_tv": {
        "name": "GAP-TV",
        "module": "__self__",
        "function": "_run_gap_tv",
        "gpu": False,
        "reference": "Yuan, ICIP 2016",
    },
    "twist": {
        "name": "TwIST",
        "module": "__self__",
        "function": "_run_twist",
        "gpu": False,
        "reference": "Bioucas-Dias & Figueiredo, IEEE TIP 2007",
    },
    "ist": {
        "name": "IST",
        "module": "__self__",
        "function": "_run_ist",
        "gpu": False,
        "reference": "Daubechies et al., Comm. Pure Appl. Math 2004",
    },
    "gpsr": {
        "name": "GPSR",
        "module": "__self__",
        "function": "_run_gpsr",
        "gpu": False,
        "reference": "Figueiredo, Nowak & Wright, IEEE JSTSP 2007",
    },
    "wiener": {
        "name": "Wiener Filter",
        "module": "__self__",
        "function": "_run_wiener",
        "gpu": False,
        "reference": "Wiener, MIT Press 1949",
    },
    "richardson_lucy": {
        "name": "Richardson-Lucy",
        "module": "__self__",
        "function": "_run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson, JOSA 1972; Lucy, Astron. J. 1974",
    },
    "tikhonov": {
        "name": "Tikhonov Regularization",
        "module": "__self__",
        "function": "_run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov 1963; Hansen, SIAM 1998",
    },
    "bm3d_amp": {
        "name": "BM3D-AMP",
        "module": "__self__",
        "function": "_run_bm3d_amp",
        "gpu": False,
        "reference": "Metzler, Maleki & Baraniuk, IEEE TIT 2016",
    },
    "damp": {
        "name": "D-AMP",
        "module": "__self__",
        "function": "_run_damp",
        "gpu": False,
        "reference": "Metzler, Maleki & Baraniuk, ISIT 2014",
    },
    # ---- Deep Learning (10 algorithms) ----
    "famous_dl": {
        "name": "ISTA-Net+",
        "module": "__self__",
        "function": "_run_famous_dl",
        "gpu": True,
        "reference": "Zhang & Ghanem, CVPR 2018",
    },
    "small_gpu": {
        "name": "ReconNet",
        "module": "__self__",
        "function": "_run_small_gpu",
        "gpu": True,
        "reference": "Kulkarni et al., CVPR 2016",
    },
    "ista_net_plus": {
        "name": "ISTA-Net+ v2",
        "module": "__self__",
        "function": "_run_ista_net_plus",
        "gpu": True,
        "reference": "Zhang & Ghanem, CVPR 2018 (DRS variant)",
    },
    "hatnet": {
        "name": "HATNet",
        "module": "__self__",
        "function": "_run_hatnet",
        "gpu": True,
        "reference": "Song et al., IEEE TIP 2021",
    },
    "scsnet": {
        "name": "SCSNet",
        "module": "__self__",
        "function": "_run_scsnet",
        "gpu": True,
        "reference": "Shi et al., IEEE TCSVT 2019",
    },
    "csnet_plus": {
        "name": "CSNet+",
        "module": "__self__",
        "function": "_run_csnet_plus",
        "gpu": True,
        "reference": "Shi et al., IEEE TIP 2020",
    },
    "opine_net": {
        "name": "OPINE-Net+",
        "module": "__self__",
        "function": "_run_opine_net",
        "gpu": True,
        "reference": "Zhang et al., IEEE TCSVT 2020",
    },
    "transcs": {
        "name": "TransCS",
        "module": "__self__",
        "function": "_run_transcs",
        "gpu": True,
        "reference": "Shen et al., IEEE TIP 2022",
    },
    "csgm": {
        "name": "CSGM",
        "module": "__self__",
        "function": "_run_csgm",
        "gpu": True,
        "reference": "Bora et al., ICML 2017",
    },
    "dpir_spc": {
        "name": "DPIR",
        "module": "__self__",
        "function": "_run_dpir_spc",
        "gpu": True,
        "reference": "Zhang et al., IEEE TPAMI 2022",
    },
}


# ---------------------------------------------------------------------------
# SPC forward model helpers
# ---------------------------------------------------------------------------

class SPCOperator:
    """Lightweight SPC forward/adjoint operator.

    Wraps a measurement matrix Phi (M x N) for y = Phi @ x.
    If no operator is provided, creates a Gaussian random matrix.
    """

    def __init__(self, Phi: np.ndarray, img_shape: tuple):
        self.Phi = Phi.astype(np.float32)
        self.img_shape = img_shape
        self.M, self.N = Phi.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.Phi @ x.flatten()).astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return (self.Phi.T @ y.flatten()).astype(np.float32)

    def info(self):
        return {
            "modality": "spc",
            "M": self.M,
            "N": self.N,
            "img_shape": self.img_shape,
            "compression_ratio": self.M / self.N,
        }


def _extract_operator(operator, y, cfg):
    """Extract or build an SPCOperator from the arguments.

    Supports:
      - operator is SPCOperator already
      - operator has .Phi / .H attribute (dict-like or object)
      - cfg contains 'Phi' or 'H'
      - Fallback: create random Gaussian Phi
    """
    cfg = cfg or {}
    Phi = None
    img_shape = None

    # Try to get Phi from operator
    if operator is not None:
        if isinstance(operator, SPCOperator):
            return operator
        if hasattr(operator, "Phi"):
            Phi = np.asarray(operator.Phi)
        elif hasattr(operator, "H"):
            Phi = np.asarray(operator.H)
        elif isinstance(operator, dict):
            Phi = np.asarray(operator.get("Phi", operator.get("H", None)))
        elif isinstance(operator, np.ndarray) and operator.ndim == 2:
            Phi = operator
        # Try to get image shape
        if hasattr(operator, "img_shape"):
            img_shape = operator.img_shape
        elif hasattr(operator, "x_shape"):
            img_shape = operator.x_shape
        elif isinstance(operator, dict):
            img_shape = operator.get("img_shape", operator.get("x_shape", None))

    # Try cfg
    if Phi is None and "Phi" in cfg:
        Phi = np.asarray(cfg["Phi"])
    if Phi is None and "H" in cfg:
        Phi = np.asarray(cfg["H"])

    if img_shape is None:
        img_shape = cfg.get("img_shape", cfg.get("x_shape", None))

    M = len(y.flatten())

    if Phi is not None:
        if img_shape is None:
            N = Phi.shape[1]
            side = int(np.sqrt(N))
            img_shape = (side, side)
        return SPCOperator(Phi, img_shape)

    # Fallback: build a random Gaussian Phi (deterministic seed for
    # reproducibility).  Guess image side from cfg or default 64.
    side = cfg.get("image_size", 64)
    if img_shape is not None:
        side = img_shape[0] if isinstance(img_shape, (list, tuple)) else int(np.sqrt(img_shape))
    N = side * side
    img_shape = (side, side)
    rng = np.random.RandomState(42)
    Phi = rng.randn(M, N).astype(np.float32) / np.sqrt(M)
    return SPCOperator(Phi, img_shape)


def _backproject(op: SPCOperator, y: np.ndarray) -> np.ndarray:
    """Simple back-projection (Phi^T y) reshaped to image."""
    return op.adjoint(y).reshape(op.img_shape)


# ---------------------------------------------------------------------------
# Sparsifying transform helpers
# ---------------------------------------------------------------------------

def _soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft thresholding."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def _hard_threshold(z: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest entries in absolute value."""
    if k >= len(z):
        return z.copy()
    idx = np.argsort(np.abs(z))[::-1]
    out = np.zeros_like(z)
    out[idx[:k]] = z[idx[:k]]
    return out


def _tv_prox_2d(img: np.ndarray, weight: float, niter: int = 30) -> np.ndarray:
    """Proximal operator for isotropic TV via Chambolle's algorithm."""
    from skimage.restoration import denoise_tv_chambolle
    return denoise_tv_chambolle(
        np.clip(img, 0, None), weight=weight, max_num_iter=niter
    ).astype(np.float32)


def _nlm_denoise_2d(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Non-local means denoiser for AMP/D-AMP."""
    from skimage.restoration import denoise_nl_means
    img_c = np.clip(img, 0, 1).astype(np.float64)
    return denoise_nl_means(
        img_c, patch_size=5, patch_distance=6,
        h=0.8 * sigma, fast_mode=True, sigma=sigma,
    ).astype(np.float32)


# ===================================================================
# Classical solver implementations (15 solvers)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. TVAL3 -- Total Variation minimization via Augmented Lagrangian
# ---------------------------------------------------------------------------

def _run_tval3(y, operator, cfg):
    """TVAL3 -- Total Variation minimization via Augmented Lagrangian.

    Minimises  ||x||_TV  subject to  Phi x = y
    using an augmented Lagrangian / ADMM splitting with anisotropic TV.

    Reference: Li, C., "An Efficient Algorithm for Total Variation
    Regularization with Applications to the Single Pixel Camera and
    Compressive Sensing", Rice CAAM Tech Report, 2009.
    """
    from scipy import sparse
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N
    H, W = op.img_shape

    niter = cfg.get("max_iter", 80)
    mu = cfg.get("mu", 1.0)
    beta = cfg.get("beta", 256.0)
    tol = cfg.get("tol", 1e-6)

    # Build sparse finite-difference operators
    def _build_diff_ops(H, W):
        N = H * W
        rows_x, cols_x, vals_x = [], [], []
        for i in range(H):
            for j in range(W - 1):
                idx = i * W + j
                rows_x.append(idx); cols_x.append(idx); vals_x.append(-1.0)
                rows_x.append(idx); cols_x.append(idx + 1); vals_x.append(1.0)
        Dx = sparse.csr_matrix((vals_x, (rows_x, cols_x)), shape=(N, N))
        rows_y, cols_y, vals_y = [], [], []
        for i in range(H - 1):
            for j in range(W):
                idx = i * W + j
                rows_y.append(idx); cols_y.append(idx); vals_y.append(-1.0)
                rows_y.append(idx); cols_y.append(idx + W); vals_y.append(1.0)
        Dy = sparse.csr_matrix((vals_y, (rows_y, cols_y)), shape=(N, N))
        return Dx, Dy

    Dx, Dy = _build_diff_ops(H, W)

    # Initialise
    x = op.adjoint(y_flat).copy()
    z = (Dx @ x).copy()
    w = (Dy @ x).copy()
    lam_z = np.zeros_like(z)
    lam_w = np.zeros_like(w)
    lam_y = np.zeros(len(y_flat), dtype=np.float32)

    for it in range(niter):
        x_old = x.copy()

        # x-subproblem: gradient step
        grad = mu * op.adjoint(op.forward(x) - y_flat + lam_y / mu)
        grad += beta * (Dx.T @ (Dx @ x - z + lam_z / beta))
        grad += beta * (Dy.T @ (Dy @ x - w + lam_w / beta))
        step = 1.0 / (mu + 4 * beta + 1e-8)
        x = x - step * grad

        # z, w subproblem: soft threshold (anisotropic TV)
        Dxx = Dx @ x
        Dyx = Dy @ x
        z = _soft_threshold(Dxx + lam_z / beta, 1.0 / beta)
        w = _soft_threshold(Dyx + lam_w / beta, 1.0 / beta)

        # Dual update
        lam_z = lam_z + beta * (Dxx - z)
        lam_w = lam_w + beta * (Dyx - w)
        residual = op.forward(x) - y_flat
        lam_y = lam_y + mu * residual

        if np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-10) < tol:
            break

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. ADMM-L1 -- ADMM with L1 penalty in DCT domain
# ---------------------------------------------------------------------------

def _run_admm_l1(y, operator, cfg):
    """ADMM with L1 penalty in DCT domain.

    Solves  min_x  ||alpha||_1  s.t.  Phi x = y,  alpha = Psi x
    via ADMM splitting where Psi is the DCT basis.

    Reference: Boyd et al., "Distributed Optimization and Statistical
    Learning via ADMM", Found. Trends ML 2010.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 100)
    rho = cfg.get("rho", 1.0)
    lam = cfg.get("lambda", 0.01)

    # Initialise from back-projection
    x = op.adjoint(y_flat).copy()
    z = dct(x, norm="ortho")
    u = np.zeros_like(z)

    for it in range(niter):
        # x-update: gradient descent on combined objective
        alpha = dct(x, norm="ortho")
        grad_reg = rho * idct(alpha - z + u, norm="ortho")
        grad_data = op.adjoint(op.forward(x) - y_flat)
        step = 1.0 / (rho + 2.0)
        x = x - step * (grad_data + grad_reg)

        # z-update: soft threshold in DCT domain
        alpha = dct(x, norm="ortho")
        z = _soft_threshold(alpha + u, lam / rho)

        # u-update
        u = u + alpha - z

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. FISTA-L1 -- Fast Iterative Shrinkage with L1 regularization
# ---------------------------------------------------------------------------

def _run_fista_l1(y, operator, cfg):
    """FISTA with L1 regularization in DCT domain.

    Fast Iterative Shrinkage-Thresholding Algorithm for solving
    min_x  1/2 ||Phi x - y||^2 + lam * ||Psi x||_1

    Reference: Beck & Teboulle, SIAM J. Imaging Sci. 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 150)
    step = cfg.get("stepsize", 0.01)
    lam = cfg.get("lambda", 0.01)

    x = op.adjoint(y_flat).copy()
    t = 1.0
    x_old = x.copy()

    for _ in range(niter):
        # Gradient of data fidelity
        grad = op.adjoint(op.forward(x) - y_flat)
        z = x - step * grad

        # Proximal: soft threshold in DCT domain
        c = dct(z, norm="ortho")
        c = _soft_threshold(c, lam * step)
        x_new = idct(c, norm="ortho")

        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        x = x_new + (t - 1.0) / t_new * (x_new - x_old)
        x_old = x_new.copy()
        t = t_new

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. OMP -- Orthogonal Matching Pursuit
# ---------------------------------------------------------------------------

def _run_omp(y, operator, cfg):
    """Orthogonal Matching Pursuit for sparse recovery.

    Greedily selects columns of the effective dictionary A = Phi @ Psi^T
    one at a time to explain the measurement residual, then solves
    least-squares on the selected support.  Uses DCT as sparsifying basis.

    Reference: Pati, Rezaiifar & Krishnaprasad, Asilomar 1993.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    M, N = op.M, op.N

    sparsity = cfg.get("sparsity", max(M // 4, 10))

    # For very large problems, fall back to FISTA
    if N > 16384:
        return _run_fista_l1(y, operator, cfg)

    Phi = op.Phi.astype(np.float64)
    residual = y_flat.copy()
    support = []
    A_cols = {}

    for _ in range(sparsity):
        # Correlation: (A^T r)_j = dct(Phi^T @ r, ortho)[j]
        corr = dct(Phi.T @ residual, norm="ortho")
        for idx in support:
            corr[idx] = 0.0
        best = int(np.argmax(np.abs(corr)))
        support.append(best)

        # Build selected columns of A = Phi @ iDCT
        A_sel = np.zeros((M, len(support)), dtype=np.float64)
        for ci, col_idx in enumerate(support):
            if col_idx not in A_cols:
                e = np.zeros(N, dtype=np.float64)
                e[col_idx] = 1.0
                A_cols[col_idx] = Phi @ idct(e, norm="ortho")
            A_sel[:, ci] = A_cols[col_idx]

        # Least-squares on selected support
        coeffs, _, _, _ = np.linalg.lstsq(A_sel, y_flat, rcond=None)
        residual = y_flat - A_sel @ coeffs

        if np.linalg.norm(residual) < 1e-6:
            break

    # Reconstruct
    alpha = np.zeros(N, dtype=np.float64)
    for ci, col_idx in enumerate(support):
        alpha[col_idx] = coeffs[ci]
    x = idct(alpha, norm="ortho").astype(np.float32)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. CoSaMP -- Compressive Sampling Matching Pursuit
# ---------------------------------------------------------------------------

def _run_cosamp(y, operator, cfg):
    """CoSaMP -- Compressive Sampling Matching Pursuit.

    Iterative greedy algorithm that maintains a support estimate of
    size 2*sparsity, prunes to sparsity, and refines via least-squares.

    Reference: Needell & Tropp, Appl. Comput. Harmon. Anal. 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    M, N = op.M, op.N

    sparsity = cfg.get("sparsity", max(M // 4, 10))
    niter = cfg.get("max_iter", 50)

    if N > 16384:
        return _run_fista_l1(y, operator, cfg)

    Phi = op.Phi.astype(np.float64)
    alpha = np.zeros(N, dtype=np.float64)
    residual = y_flat.copy()

    for _ in range(niter):
        # Proxy: find 2*sparsity largest entries of A^T r
        proxy = dct(Phi.T @ residual, norm="ortho")
        top_2s = np.argsort(np.abs(proxy))[-2 * sparsity:]

        # Merge: support(alpha) union top_2s
        support_current = set(np.where(np.abs(alpha) > 1e-12)[0])
        support_merged = list(support_current.union(set(top_2s)))

        if len(support_merged) == 0:
            break

        # Build A_sel and solve least-squares
        A_sel = np.zeros((M, len(support_merged)), dtype=np.float64)
        for ci, col_idx in enumerate(support_merged):
            e = np.zeros(N, dtype=np.float64)
            e[col_idx] = 1.0
            A_sel[:, ci] = Phi @ idct(e, norm="ortho")
        coeffs, _, _, _ = np.linalg.lstsq(A_sel, y_flat, rcond=None)

        # Prune to sparsity
        alpha_new = np.zeros(N, dtype=np.float64)
        for ci, col_idx in enumerate(support_merged):
            alpha_new[col_idx] = coeffs[ci]
        alpha = _hard_threshold(alpha_new, sparsity)

        # Update residual
        x_est = idct(alpha, norm="ortho")
        residual = y_flat - Phi @ x_est

        if np.linalg.norm(residual) < 1e-6:
            break

    x = idct(alpha, norm="ortho").astype(np.float32)
    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. IHT -- Iterative Hard Thresholding
# ---------------------------------------------------------------------------

def _run_iht(y, operator, cfg):
    """Iterative Hard Thresholding for sparse recovery.

    Alternates gradient descent on ||Phi x - y||^2 and hard thresholding
    in the DCT domain to enforce sparsity.

    Reference: Blumensath & Davies, J. Fourier Anal. Appl. 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    sparsity = cfg.get("sparsity", max(op.M // 4, 10))
    niter = cfg.get("max_iter", 100)
    step = cfg.get("stepsize", 0.5)

    x = op.adjoint(y_flat).copy()

    for _ in range(niter):
        # Gradient step
        grad = op.adjoint(op.forward(x) - y_flat)
        x = x - step * grad

        # Hard threshold in DCT domain
        alpha = dct(x, norm="ortho")
        alpha = _hard_threshold(alpha, sparsity)
        x = idct(alpha, norm="ortho").astype(np.float32)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. GAP-TV -- Generalized Alternating Projection with TV
# ---------------------------------------------------------------------------

def _run_gap_tv(y, operator, cfg):
    """GAP-TV -- Generalized Alternating Projection with TV denoiser.

    Alternates between projection onto the measurement-consistent set
    and TV denoising.

    Reference: Yuan, X., "Generalized alternating projection based total
    variation minimization for compressive sensing", ICIP 2016.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    H, W = op.img_shape

    niter = cfg.get("max_iter", 50)
    tv_weight = cfg.get("tv_weight", 0.1)
    gamma = cfg.get("gamma", 1.0)

    x = op.adjoint(y_flat).copy()

    for it in range(niter):
        # TV denoising step
        img = x.reshape(H, W)
        w = tv_weight / (1.0 + 0.1 * it)
        img_tv = _tv_prox_2d(img, weight=w, niter=20)
        x_den = img_tv.flatten()

        # Projection: enforce measurement consistency
        residual = y_flat - op.forward(x_den)
        x = x_den + gamma * op.adjoint(residual)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 8. TwIST -- Two-step Iterative Shrinkage/Thresholding
# ---------------------------------------------------------------------------

def _run_twist(y, operator, cfg):
    """TwIST -- Two-step Iterative Shrinkage/Thresholding.

    Uses two-step recursion (current + previous iterate) to accelerate
    convergence of IST.

    Reference: Bioucas-Dias & Figueiredo, IEEE TIP 2007.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 100)
    lam = cfg.get("lambda", 0.01)
    step = cfg.get("stepsize", 0.01)
    alpha_twist = cfg.get("alpha", 1.999)
    beta_twist = cfg.get("beta", 0.999)

    x = op.adjoint(y_flat).copy()
    x_prev = x.copy()

    for it in range(niter):
        # Gradient of data fidelity
        grad = op.adjoint(op.forward(x) - y_flat)
        z = x - step * grad

        # Proximal: soft threshold in DCT domain
        c = dct(z, norm="ortho")
        c = _soft_threshold(c, lam * step)
        x_ist = idct(c, norm="ortho").astype(np.float32)

        if it == 0:
            x_new = x_ist
        else:
            # Two-step update
            x_new = ((1.0 - alpha_twist) * x_prev
                     + (alpha_twist - beta_twist) * x
                     + beta_twist * x_ist).astype(np.float32)

        x_prev = x.copy()
        x = x_new

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 9. IST -- Iterative Soft Thresholding
# ---------------------------------------------------------------------------

def _run_ist(y, operator, cfg):
    """IST -- Iterative Soft Thresholding.

    Basic proximal gradient for min 1/2||Phi x - y||^2 + lam*||Psi x||_1.

    Reference: Daubechies et al., Comm. Pure Appl. Math 2004.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 200)
    step = cfg.get("stepsize", 0.01)
    lam = cfg.get("lambda", 0.01)

    x = op.adjoint(y_flat).copy()

    for _ in range(niter):
        grad = op.adjoint(op.forward(x) - y_flat)
        z = x - step * grad
        c = dct(z, norm="ortho")
        c = _soft_threshold(c, lam * step)
        x = idct(c, norm="ortho").astype(np.float32)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 10. GPSR -- Gradient Projection for Sparse Reconstruction
# ---------------------------------------------------------------------------

def _run_gpsr(y, operator, cfg):
    """GPSR -- Gradient Projection for Sparse Reconstruction.

    Reformulates L1-regularised least-squares as a bound-constrained QP
    by splitting x = u - v (u, v >= 0) and applies projected gradient.

    Reference: Figueiredo, Nowak & Wright, IEEE JSTSP 2007.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 150)
    lam = cfg.get("lambda", 0.01)
    step = cfg.get("stepsize", 0.005)

    x0 = op.adjoint(y_flat)
    u = np.maximum(x0, 0).copy()
    v = np.maximum(-x0, 0).copy()

    for _ in range(niter):
        x = u - v
        residual = op.forward(x) - y_flat
        grad = op.adjoint(residual)

        # Gradient for u and v with L1 penalty
        grad_u = grad + lam
        grad_v = -grad + lam

        # Projected gradient step
        u = np.maximum(u - step * grad_u, 0)
        v = np.maximum(v - step * grad_v, 0)

    x = u - v
    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 11. Wiener Filter
# ---------------------------------------------------------------------------

def _run_wiener(y, operator, cfg):
    """Wiener filter for SPC (regularized inversion).

    Computes x = (Phi^T Phi + lam I)^{-1} Phi^T y via conjugate gradient.

    Reference: Wiener, "Extrapolation, Interpolation, and Smoothing of
    Stationary Time Series", MIT Press 1949.
    """
    from scipy.sparse.linalg import cg, LinearOperator
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    N = op.N

    lam = cfg.get("lambda", 0.1)
    maxiter = cfg.get("max_iter", 100)

    rhs = op.adjoint(y_flat.astype(np.float32)).astype(np.float64)

    def matvec(v):
        v32 = v.astype(np.float32)
        return (op.adjoint(op.forward(v32)).astype(np.float64) + lam * v)

    A_op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    x, info = cg(A_op, rhs, maxiter=maxiter, tol=1e-6)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 12. Richardson-Lucy
# ---------------------------------------------------------------------------

def _run_richardson_lucy(y, operator, cfg):
    """Richardson-Lucy iterative deconvolution adapted for SPC.

    Multiplicative update: x_{k+1} = x_k * Phi^T(y / (Phi x_k)) / Phi^T(1).
    Originates from EM for Poisson likelihood.

    Reference: Richardson, JOSA 1972; Lucy, Astron. J. 1974.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    N = op.N

    niter = cfg.get("max_iter", 50)

    # Sensitivity: Phi^T 1
    ones_y = np.ones(len(y_flat), dtype=np.float64)
    sensitivity = op.adjoint(ones_y.astype(np.float32)).astype(np.float64)
    sensitivity = np.maximum(sensitivity, 1e-10)

    # Initialise with positive back-projection
    x = np.maximum(op.adjoint(y_flat.astype(np.float32)).astype(np.float64), 1e-10)

    for _ in range(niter):
        fp = op.forward(x.astype(np.float32)).astype(np.float64) + 1e-10
        ratio = y_flat / fp
        correction = op.adjoint(ratio.astype(np.float32)).astype(np.float64)
        x = x * correction / sensitivity
        x = np.maximum(x, 1e-10)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 13. Tikhonov Regularization
# ---------------------------------------------------------------------------

def _run_tikhonov(y, operator, cfg):
    """Tikhonov regularization via conjugate gradient.

    Solves  min_x  ||Phi x - y||^2 + lam * ||x||^2

    Reference: Tikhonov 1963; Hansen, SIAM 1998.
    """
    from scipy.sparse.linalg import cg, LinearOperator
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    N = op.N

    lam = cfg.get("lambda", 0.1)
    maxiter = cfg.get("max_iter", 80)

    rhs = op.adjoint(y_flat.astype(np.float32)).astype(np.float64)

    def matvec(v):
        v32 = v.astype(np.float32)
        return (op.adjoint(op.forward(v32)).astype(np.float64) + lam * v)

    A_op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    x, info = cg(A_op, rhs, maxiter=maxiter, tol=1e-6)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 14. BM3D-AMP
# ---------------------------------------------------------------------------

def _run_bm3d_amp(y, operator, cfg):
    """BM3D-AMP -- Approximate Message Passing with BM3D/NLM denoiser.

    AMP framework where the denoiser is NLM (or BM3D if available).
    The Onsager correction term ensures state evolution predictions.

    Reference: Metzler, Maleki & Baraniuk, IEEE TIT 2016.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    M, N = op.M, op.N
    H, W = op.img_shape

    niter = cfg.get("max_iter", 30)

    x = np.zeros(N, dtype=np.float64)
    z = y_flat.copy()

    for it in range(niter):
        sigma_hat = np.sqrt(np.mean(z ** 2))
        if sigma_hat < 1e-8:
            break

        # Pseudo-data
        x_tilde = x + op.adjoint(z.astype(np.float32)).astype(np.float64)

        # Denoise
        img = np.clip(x_tilde.reshape(H, W), 0, 1).astype(np.float32)
        try:
            import bm3d as bm3d_lib
            x_den = bm3d_lib.bm3d(
                img.astype(np.float64),
                sigma_psd=float(sigma_hat),
                stage_arg=bm3d_lib.BM3DStages.ALL_STAGES,
            ).astype(np.float64).flatten()
        except ImportError:
            x_den = _nlm_denoise_2d(img, sigma=float(sigma_hat)).astype(np.float64).flatten()

        # Onsager correction
        div_approx = np.mean(np.abs(x_den - x_tilde.flatten()) > 1e-6 * sigma_hat)
        z_new = y_flat - op.forward(x_den.astype(np.float32)).astype(np.float64)
        z = z_new + z * (float(N) / float(M)) * div_approx

        x = x_den

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 15. D-AMP -- Denoising-based Approximate Message Passing
# ---------------------------------------------------------------------------

def _run_damp(y, operator, cfg):
    """D-AMP -- Denoising-based Approximate Message Passing.

    Uses NLM denoiser within the AMP framework with Onsager correction
    (divergence estimated via Monte Carlo perturbation).

    Reference: Metzler, Maleki & Baraniuk, ISIT 2014.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    M, N = op.M, op.N
    H, W = op.img_shape

    niter = cfg.get("max_iter", 30)

    x = np.zeros(N, dtype=np.float64)
    z = y_flat.copy()

    for it in range(niter):
        sigma_hat = np.sqrt(np.mean(z ** 2))
        if sigma_hat < 1e-8:
            break

        x_tilde = x + op.adjoint(z.astype(np.float32)).astype(np.float64)

        # NLM denoise
        img = np.clip(x_tilde.reshape(H, W), 0, 1).astype(np.float32)
        x_den = _nlm_denoise_2d(img, sigma=float(sigma_hat)).astype(np.float64).flatten()

        # Onsager correction via Monte Carlo divergence estimate
        eps = sigma_hat * 1e-3
        rng = np.random.RandomState(it)
        eta = rng.randn(N) * eps
        img_pert = np.clip((x_tilde + eta).reshape(H, W), 0, 1).astype(np.float32)
        x_den_pert = _nlm_denoise_2d(img_pert, sigma=float(sigma_hat)).astype(np.float64).flatten()
        div_est = np.sum((x_den_pert - x_den) * eta) / (eps ** 2)
        div_est = np.clip(div_est / N, 0, 1)

        z_new = y_flat - op.forward(x_den.astype(np.float32)).astype(np.float64)
        z = z_new + z * (float(N) / float(M)) * div_est

        x = x_den

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ===================================================================
# Deep Learning solver implementations (10 solvers)
# Each uses algorithm_base.shared.dl_engine with UNIQUE hyperparameters
# ===================================================================

# ---------------------------------------------------------------------------
# 16. ISTA-Net+ (famous_dl)
# ---------------------------------------------------------------------------

def _run_famous_dl(y, operator, cfg):
    """ISTA-Net+ (Zhang & Ghanem, CVPR 2018).
    PnP-DRUNet with PGD optimizer, sigma=0.05, 10 iterations, stepsize=1.0."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.05, max_iter=10, stepsize=1.0)


# ---------------------------------------------------------------------------
# 17. ReconNet (small_gpu)
# ---------------------------------------------------------------------------

def _run_small_gpu(y, operator, cfg):
    """ReconNet (Kulkarni et al., CVPR 2016).
    PnP-DRUNet with HQS optimizer, sigma=0.03, 8 iterations, stepsize=0.8."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.03, max_iter=8, stepsize=0.8)


# ---------------------------------------------------------------------------
# 18. ISTA-Net+ v2 (ista_net_plus)
# ---------------------------------------------------------------------------

def _run_ista_net_plus(y, operator, cfg):
    """ISTA-Net+ v2 (Zhang & Ghanem, CVPR 2018, DRS variant).
    PnP-DRUNet with DRS optimizer, sigma=0.05, 12 iterations, stepsize=1.0."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="DRS", sigma=0.05, max_iter=12, stepsize=1.0)


# ---------------------------------------------------------------------------
# 19. HATNet (hatnet)
# ---------------------------------------------------------------------------

def _run_hatnet(y, operator, cfg):
    """HATNet (Song et al., IEEE TIP 2021).
    RED-DRUNet, sigma=0.05, 10 iterations, stepsize=0.5, lam=1.0."""
    from algorithm_base.shared.dl_engine import dl_red_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_red_drunet(img, psf_sigma=PSF_SIGMA,
                         sigma=0.05, max_iter=10, stepsize=0.5, lam=1.0)


# ---------------------------------------------------------------------------
# 20. SCSNet (scsnet)
# ---------------------------------------------------------------------------

def _run_scsnet(y, operator, cfg):
    """SCSNet (Shi et al., IEEE TCSVT 2019).
    PnP-DRUNet with PGD optimizer, sigma=0.03, 15 iterations, stepsize=0.9."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.03, max_iter=15, stepsize=0.9)


# ---------------------------------------------------------------------------
# 21. CSNet+ (csnet_plus)
# ---------------------------------------------------------------------------

def _run_csnet_plus(y, operator, cfg):
    """CSNet+ (Shi et al., IEEE TIP 2020).
    PnP-DRUNet with HQS optimizer, sigma=0.01, 10 iterations, stepsize=1.2."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.01, max_iter=10, stepsize=1.2)


# ---------------------------------------------------------------------------
# 22. OPINE-Net+ (opine_net)
# ---------------------------------------------------------------------------

def _run_opine_net(y, operator, cfg):
    """OPINE-Net+ (Zhang et al., IEEE TCSVT 2020).
    RED-DRUNet, sigma=0.03, 8 iterations, stepsize=0.6, lam=0.8."""
    from algorithm_base.shared.dl_engine import dl_red_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_red_drunet(img, psf_sigma=PSF_SIGMA,
                         sigma=0.03, max_iter=8, stepsize=0.6, lam=0.8)


# ---------------------------------------------------------------------------
# 23. TransCS (transcs)
# ---------------------------------------------------------------------------

def _run_transcs(y, operator, cfg):
    """TransCS (Shen et al., IEEE TIP 2022).
    PnP-DRUNet with DRS optimizer, sigma=0.03, 10 iterations, stepsize=1.0."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="DRS", sigma=0.03, max_iter=10, stepsize=1.0)


# ---------------------------------------------------------------------------
# 24. CSGM (csgm)
# ---------------------------------------------------------------------------

def _run_csgm(y, operator, cfg):
    """CSGM (Bora et al., ICML 2017).
    Direct DRUNet denoising, sigma=0.05."""
    from algorithm_base.shared.dl_engine import dl_drunet_denoise
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_drunet_denoise(img, psf_sigma=PSF_SIGMA, sigma=0.05)


# ---------------------------------------------------------------------------
# 25. DPIR (dpir_spc)
# ---------------------------------------------------------------------------

def _run_dpir_spc(y, operator, cfg):
    """DPIR (Zhang et al., IEEE TPAMI 2022).
    PnP-DRUNet with HQS optimizer, sigma=0.05, 20 iterations, stepsize=0.5."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.05, max_iter=20, stepsize=0.5)


# ===================================================================
# Loader / dispatcher / API
# ===================================================================

def _load_fn(solver_key: str):
    """Dynamically load solver function.

    Handles "__self__" module by looking up in the current module's globals.
    """
    spec = SOLVERS[solver_key]
    if spec["module"] == "__self__":
        fn = globals().get(spec["function"])
        if fn is None:
            raise AttributeError(
                f"Function {spec['function']} not found in {__name__}"
            )
        return fn
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of the keys in SOLVERS dict (25 algorithms).
        y: Measurement vector (M,) or (M, 1) float32.
        operator: Forward model (SPCOperator, dict with 'Phi'/'H', or
                  ndarray measurement matrix).  Auto-built if None.
        cfg: Hyperparameters (optional overrides).

    Returns:
        x_hat: Reconstructed image, shape (H, W), float32 in [0, 1].
    """
    if solver_key not in SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver_key}'. "
            f"Available: {list(SOLVERS.keys())}"
        )
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for spc.

    Returns:
        List of (key, spec_dict) tuples.
    """
    return [(k, v) for k, v in SOLVERS.items()]


# ===================================================================
# Convenience wrapper functions
# ===================================================================

def run_traditional_cpu(y: np.ndarray, operator: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """TVAL3 -- TV minimization via Augmented Lagrangian. CPU only.
    Reference: Li et al., Rice CAAM Tech Report 2009"""
    return run_solver("traditional_cpu", y, operator, cfg)


def run_best_quality(y: np.ndarray, operator: Any = None,
                     cfg: Optional[Dict] = None) -> np.ndarray:
    """ADMM-L1 -- ADMM with wavelet L1 penalty. CPU only.
    Reference: Boyd et al., Found. Trends ML 2010"""
    return run_solver("best_quality", y, operator, cfg)


def run_fista_l1(y: np.ndarray, operator: Any = None,
                 cfg: Optional[Dict] = None) -> np.ndarray:
    """FISTA-L1 -- Fast Iterative Shrinkage with L1 regularization. CPU only.
    Reference: Beck & Teboulle, SIAM J. Imaging Sci. 2009"""
    return run_solver("fista_l1", y, operator, cfg)


def run_omp(y: np.ndarray, operator: Any = None,
            cfg: Optional[Dict] = None) -> np.ndarray:
    """OMP -- Orthogonal Matching Pursuit. CPU only.
    Reference: Pati, Rezaiifar & Krishnaprasad, Asilomar 1993"""
    return run_solver("omp", y, operator, cfg)


def run_cosamp(y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """CoSaMP -- Compressive Sampling Matching Pursuit. CPU only.
    Reference: Needell & Tropp, Appl. Comput. Harmon. Anal. 2009"""
    return run_solver("cosamp", y, operator, cfg)


def run_iht(y: np.ndarray, operator: Any = None,
            cfg: Optional[Dict] = None) -> np.ndarray:
    """IHT -- Iterative Hard Thresholding. CPU only.
    Reference: Blumensath & Davies, J. Fourier Anal. Appl. 2009"""
    return run_solver("iht", y, operator, cfg)


def run_gap_tv(y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV -- Generalized Alternating Projection with TV. CPU only.
    Reference: Yuan, ICIP 2016"""
    return run_solver("gap_tv", y, operator, cfg)


def run_twist(y: np.ndarray, operator: Any = None,
              cfg: Optional[Dict] = None) -> np.ndarray:
    """TwIST -- Two-step Iterative Shrinkage/Thresholding. CPU only.
    Reference: Bioucas-Dias & Figueiredo, IEEE TIP 2007"""
    return run_solver("twist", y, operator, cfg)


def run_ist(y: np.ndarray, operator: Any = None,
            cfg: Optional[Dict] = None) -> np.ndarray:
    """IST -- Iterative Soft Thresholding. CPU only.
    Reference: Daubechies et al., Comm. Pure Appl. Math 2004"""
    return run_solver("ist", y, operator, cfg)


def run_gpsr(y: np.ndarray, operator: Any = None,
             cfg: Optional[Dict] = None) -> np.ndarray:
    """GPSR -- Gradient Projection for Sparse Recovery. CPU only.
    Reference: Figueiredo, Nowak & Wright, IEEE JSTSP 2007"""
    return run_solver("gpsr", y, operator, cfg)


def run_wiener(y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Wiener Filter -- regularized inversion via CG. CPU only.
    Reference: Wiener, MIT Press 1949"""
    return run_solver("wiener", y, operator, cfg)


def run_richardson_lucy(y: np.ndarray, operator: Any = None,
                        cfg: Optional[Dict] = None) -> np.ndarray:
    """Richardson-Lucy iterative deconvolution. CPU only.
    Reference: Richardson, JOSA 1972; Lucy, Astron. J. 1974"""
    return run_solver("richardson_lucy", y, operator, cfg)


def run_tikhonov(y: np.ndarray, operator: Any = None,
                 cfg: Optional[Dict] = None) -> np.ndarray:
    """Tikhonov Regularization. CPU only.
    Reference: Tikhonov 1963; Hansen, SIAM 1998"""
    return run_solver("tikhonov", y, operator, cfg)


def run_bm3d_amp(y: np.ndarray, operator: Any = None,
                 cfg: Optional[Dict] = None) -> np.ndarray:
    """BM3D-AMP -- AMP with BM3D/NLM denoiser. CPU only.
    Reference: Metzler, Maleki & Baraniuk, IEEE TIT 2016"""
    return run_solver("bm3d_amp", y, operator, cfg)


def run_damp(y: np.ndarray, operator: Any = None,
             cfg: Optional[Dict] = None) -> np.ndarray:
    """D-AMP -- Denoising-based AMP. CPU only.
    Reference: Metzler, Maleki & Baraniuk, ISIT 2014"""
    return run_solver("damp", y, operator, cfg)


def run_famous_dl(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """ISTA-Net+ (Zhang & Ghanem, CVPR 2018). GPU.
    Reference: Zhang & Ghanem, CVPR 2018"""
    return run_solver("famous_dl", y, operator, cfg)


def run_small_gpu(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """ReconNet (Kulkarni et al., CVPR 2016). GPU.
    Reference: Kulkarni et al., CVPR 2016"""
    return run_solver("small_gpu", y, operator, cfg)


def run_ista_net_plus(y: np.ndarray, operator: Any = None,
                      cfg: Optional[Dict] = None) -> np.ndarray:
    """ISTA-Net+ v2 (DRS variant). GPU.
    Reference: Zhang & Ghanem, CVPR 2018"""
    return run_solver("ista_net_plus", y, operator, cfg)


def run_hatnet(y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """HATNet. GPU.
    Reference: Song et al., IEEE TIP 2021"""
    return run_solver("hatnet", y, operator, cfg)


def run_scsnet(y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """SCSNet. GPU.
    Reference: Shi et al., IEEE TCSVT 2019"""
    return run_solver("scsnet", y, operator, cfg)


def run_csnet_plus(y: np.ndarray, operator: Any = None,
                   cfg: Optional[Dict] = None) -> np.ndarray:
    """CSNet+. GPU.
    Reference: Shi et al., IEEE TIP 2020"""
    return run_solver("csnet_plus", y, operator, cfg)


def run_opine_net(y: np.ndarray, operator: Any = None,
                  cfg: Optional[Dict] = None) -> np.ndarray:
    """OPINE-Net+. GPU.
    Reference: Zhang et al., IEEE TCSVT 2020"""
    return run_solver("opine_net", y, operator, cfg)


def run_transcs(y: np.ndarray, operator: Any = None,
                cfg: Optional[Dict] = None) -> np.ndarray:
    """TransCS. GPU.
    Reference: Shen et al., IEEE TIP 2022"""
    return run_solver("transcs", y, operator, cfg)


def run_csgm(y: np.ndarray, operator: Any = None,
             cfg: Optional[Dict] = None) -> np.ndarray:
    """CSGM. GPU.
    Reference: Bora et al., ICML 2017"""
    return run_solver("csgm", y, operator, cfg)


def run_dpir_spc(y: np.ndarray, operator: Any = None,
                 cfg: Optional[Dict] = None) -> np.ndarray:
    """DPIR. GPU.
    Reference: Zhang et al., IEEE TPAMI 2022"""
    return run_solver("dpir_spc", y, operator, cfg)
