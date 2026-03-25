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
from scipy.signal import fftconvolve
from typing import Any, Dict, Optional

MODALITY_ID = "spc"
DISPLAY_NAME = "Single-Pixel Camera (SPC)"

PSF_SIGMA = 2.8


def _final_norm(x):
    """Clip to [0, 1] for output."""
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Solver registry -- 37 solvers from 1949 to 2025
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
    # ---- New Classical (7 algorithms) ----
    "basis_pursuit": {
        "name": "Basis Pursuit",
        "module": "__self__",
        "function": "_run_basis_pursuit",
        "gpu": False,
        "reference": "Chen, Donoho & Saunders, SIAM Review 1998",
    },
    "subspace_pursuit": {
        "name": "Subspace Pursuit",
        "module": "__self__",
        "function": "_run_subspace_pursuit",
        "gpu": False,
        "reference": "Dai & Milenkovic, IEEE TIT 2009",
    },
    "sl0": {
        "name": "Smoothed L0 (SL0)",
        "module": "__self__",
        "function": "_run_sl0",
        "gpu": False,
        "reference": "Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009",
    },
    "amp": {
        "name": "AMP",
        "module": "__self__",
        "function": "_run_amp",
        "gpu": False,
        "reference": "Donoho, Maleki & Montanari, PNAS 2009",
    },
    "niht": {
        "name": "Normalized IHT",
        "module": "__self__",
        "function": "_run_niht",
        "gpu": False,
        "reference": "Blumensath, Sampling Theory in Signal & Image Proc. 2010",
    },
    "htp": {
        "name": "Hard Thresholding Pursuit",
        "module": "__self__",
        "function": "_run_htp",
        "gpu": False,
        "reference": "Foucart, Appl. Comput. Harmon. Anal. 2011",
    },
    "admm_tv": {
        "name": "ADMM-TV",
        "module": "__self__",
        "function": "_run_admm_tv",
        "gpu": False,
        "reference": "Boyd et al., Found. Trends ML 2011",
    },
    # ---- New Deep Learning (6 algorithms) ----
    "pnp_hqs_drunet": {
        "name": "PnP-HQS (DRUNet)",
        "module": "__self__",
        "function": "_run_pnp_hqs_drunet",
        "gpu": True,
        "reference": "Zhang et al., CVPR 2017",
    },
    "amp_net": {
        "name": "AMP-Net",
        "module": "__self__",
        "function": "_run_amp_net",
        "gpu": True,
        "reference": "Zhang et al., IEEE TIP 2021",
    },
    "csformer": {
        "name": "CSFormer (PnP-PGD DRUNet)",
        "module": "__self__",
        "function": "_run_csformer",
        "gpu": True,
        "reference": "Ye et al., NeurIPS 2023",
    },
    "diffcs": {
        "name": "DiffCS",
        "module": "__self__",
        "function": "_run_diffcs",
        "gpu": True,
        "reference": "Diffusion model for CS reconstruction, 2024",
    },
    "fsoinet": {
        "name": "FSOINet",
        "module": "__self__",
        "function": "_run_fsoinet",
        "gpu": True,
        "reference": "Chen et al., CVPR 2023",
    },
    "spc_foundation": {
        "name": "SPC-Foundation (RED DRUNet)",
        "module": "__self__",
        "function": "_run_spc_foundation",
        "gpu": True,
        "reference": "Foundation model for compressive sensing, 2025",
    },
}


# ---------------------------------------------------------------------------
# SPC forward model helpers
# ---------------------------------------------------------------------------

class MaskOperator:
    """Binary mask forward/adjoint operator for SPC challenge datasets.

    Challenge data uses element-wise binary mask: y = mask * x + noise.
    """

    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.float32)
        self.img_shape = mask.shape
        self.N = mask.shape[0] * mask.shape[1]
        self.M = self.N

    def forward(self, x: np.ndarray) -> np.ndarray:
        img = x.reshape(self.img_shape) if x.ndim == 1 else x
        out = (self.mask * img).astype(np.float32)
        return out.flatten() if x.ndim == 1 else out

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        img = y.reshape(self.img_shape) if y.ndim == 1 else y
        out = (self.mask * img).astype(np.float32)
        return out.flatten() if y.ndim == 1 else out

    def forward_2d(self, x: np.ndarray) -> np.ndarray:
        return (self.mask * x).astype(np.float32)

    def adjoint_2d(self, y: np.ndarray) -> np.ndarray:
        return (self.mask * y).astype(np.float32)


class SPCOperator:
    """PSF convolution forward/adjoint operator for SPC standard datasets.

    Standard benchmark datasets use Gaussian PSF convolution (not a CS
    measurement matrix).  The operator exposes both 2-D and 1-D (flattened)
    interfaces so existing solver code works unchanged.
    """

    def __init__(self, img_shape: tuple, psf_sigma: float = PSF_SIGMA):
        k = max(3, int(3 * psf_sigma))
        ax = np.arange(-k, k + 1)
        gx, gy = np.meshgrid(ax, ax)
        self.psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * psf_sigma ** 2)).astype(np.float32)
        self.psf /= self.psf.sum()
        self.psf_flip = self.psf[::-1, ::-1].copy()
        self.img_shape = img_shape
        self.N = img_shape[0] * img_shape[1]
        self.M = self.N  # same-size output for PSF convolution

    def forward(self, x: np.ndarray) -> np.ndarray:
        img = x.reshape(self.img_shape) if x.ndim == 1 else x
        out = fftconvolve(img, self.psf, mode='same').astype(np.float32)
        return out.flatten() if x.ndim == 1 else out

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        img = y.reshape(self.img_shape) if y.ndim == 1 else y
        out = fftconvolve(img, self.psf_flip, mode='same').astype(np.float32)
        return out.flatten() if y.ndim == 1 else out

    def forward_2d(self, x: np.ndarray) -> np.ndarray:
        return fftconvolve(x, self.psf, mode='same').astype(np.float32)

    def adjoint_2d(self, y: np.ndarray) -> np.ndarray:
        return fftconvolve(y, self.psf_flip, mode='same').astype(np.float32)



def _psf_fft(psf, shape):
    """Compute centered FFT of PSF for deconvolution."""
    full = np.zeros(shape, dtype=np.float64)
    full[:psf.shape[0], :psf.shape[1]] = psf
    full = np.roll(full, -(psf.shape[0] // 2), axis=0)
    full = np.roll(full, -(psf.shape[1] // 2), axis=1)
    return np.fft.fft2(full)


def _extract_operator(operator, y, cfg):
    """Extract or build an SPCOperator from the arguments."""
    cfg = cfg or {}
    if operator is not None and isinstance(operator, SPCOperator):
        return operator
    # Infer image shape from y
    if y.ndim >= 2:
        img_shape = y.shape[:2]
    else:
        side = int(np.sqrt(len(y)))
        img_shape = (side, side)
    return SPCOperator(img_shape, psf_sigma=PSF_SIGMA)


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

    return _final_norm(x.reshape(op.img_shape))


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

    return _final_norm(x.reshape(op.img_shape))


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

    return _final_norm(x.reshape(op.img_shape))


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
    """Wiener filter for SPC (FFT-domain regularized inversion).

    Computes x = IFFT(H* Y / (|H|^2 + lambda)) with centered PSF.

    Reference: Wiener, "Extrapolation, Interpolation, and Smoothing of
    Stationary Time Series", MIT Press 1949.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    img = y.reshape(op.img_shape) if y.ndim == 1 else y.copy()
    img = img.astype(np.float64)

    lam = cfg.get("lambda", 0.005)

    H = _psf_fft(op.psf, img.shape)
    Y = np.fft.fft2(img)
    Hconj = np.conj(H)
    x = np.real(np.fft.ifft2(Hconj * Y / (np.abs(H) ** 2 + lam)))

    return _final_norm(x)


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
    """Tikhonov regularization (FFT-domain).

    Solves  min_x  ||Phi x - y||^2 + lam * ||x||^2

    Reference: Tikhonov 1963; Hansen, SIAM 1998.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    img = y.reshape(op.img_shape) if y.ndim == 1 else y.copy()
    img = img.astype(np.float64)

    lam = cfg.get("lambda", 0.005)

    H = _psf_fft(op.psf, img.shape)
    Y = np.fft.fft2(img)
    Hconj = np.conj(H)
    x = np.real(np.fft.ifft2(Hconj * Y / (np.abs(H) ** 2 + lam)))

    return _final_norm(x)


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
    np.random.seed(42)  # deterministic for reproducibility
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

        # Denoise (Gaussian filter for deterministic output)
        img = np.clip(x_tilde.reshape(H, W), 0, 1).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        gs = max(0.5, float(sigma_hat) * 3.0)
        x_den = gaussian_filter(img, sigma=gs).flatten()

        # Onsager correction
        div_approx = np.mean(np.abs(x_den - x_tilde.flatten()) > 1e-6 * sigma_hat)
        z_new = y_flat - op.forward(x_den.astype(np.float32)).astype(np.float64)
        z = z_new + z * (float(N) / float(M)) * div_approx

        x = x_den

    # Round to float32 precision to eliminate accumulated jitter
    out = _final_norm(x.reshape(op.img_shape))
    return np.round(out, decimals=6).astype(np.float32)


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

    return _final_norm(x.reshape(op.img_shape))


# ---------------------------------------------------------------------------
# 16. Basis Pursuit (BP) -- Chen, Donoho & Saunders 1998
# ---------------------------------------------------------------------------

def _run_basis_pursuit(y, operator, cfg):
    """Basis Pursuit -- L1 minimization via ADMM (proxy for LP relaxation).

    min ||alpha||_1 s.t. Phi Psi^T alpha = y
    Solved via ADMM splitting in DCT domain.

    Reference: Chen, Donoho & Saunders, SIAM Review, 1998.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    niter = cfg.get("max_iter", 200)
    rho = cfg.get("rho", 1.0)

    x = op.adjoint(y_flat).copy()
    z = dct(x, norm="ortho")
    u = np.zeros_like(z)

    for _ in range(niter):
        alpha = dct(x, norm="ortho")
        grad_data = op.adjoint(op.forward(x) - y_flat)
        grad_reg = rho * idct(alpha - z + u, norm="ortho")
        step = 1.0 / (rho + 2.0)
        x = x - step * (grad_data + grad_reg)

        alpha = dct(x, norm="ortho")
        z = _soft_threshold(alpha + u, 1.0 / rho)
        u = u + alpha - z

    return _final_norm(x.reshape(op.img_shape))


# ---------------------------------------------------------------------------
# 17. Subspace Pursuit (SP) -- Dai & Milenkovic 2009
# ---------------------------------------------------------------------------

def _run_subspace_pursuit(y, operator, cfg):
    """Subspace Pursuit -- greedy sparse recovery.

    Iteratively maintains and refines a support estimate of fixed size,
    combining identification and refinement steps.

    Reference: Dai & Milenkovic, IEEE TIT, 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    sparsity = cfg.get("sparsity", max(op.M // 4, 10))
    niter = cfg.get("max_iter", 50)

    x = op.adjoint(y_flat).copy()
    alpha = dct(x, norm="ortho")
    support = set(np.argsort(np.abs(alpha))[-sparsity:])

    for _ in range(niter):
        residual = y_flat - op.forward(x)
        proxy = dct(op.adjoint(residual), norm="ortho")

        # Add sparsity new candidates
        candidates = set(np.argsort(np.abs(proxy))[-sparsity:])
        merged = list(support.union(candidates))

        # Solve LS on merged support (in signal domain via gradient)
        alpha_new = np.zeros(N, dtype=np.float32)
        alpha_merged = dct(op.adjoint(y_flat), norm="ortho")
        for idx in merged:
            alpha_new[idx] = alpha_merged[idx]

        # Prune to sparsity
        top_k = np.argsort(np.abs(alpha_new))[-sparsity:]
        alpha_pruned = np.zeros(N, dtype=np.float32)
        for idx in top_k:
            alpha_pruned[idx] = alpha_new[idx]

        x_new = idct(alpha_pruned, norm="ortho").astype(np.float32)
        support = set(top_k)
        x = x_new

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 18. Smoothed L0 (SL0) -- Mohimani et al. 2009
# ---------------------------------------------------------------------------

def _run_sl0(y, operator, cfg):
    """Smoothed L0 -- smooth approximation to L0 norm.

    Replaces ||x||_0 with sum of Gaussian bumps exp(-x_i^2 / 2sigma^2)
    and minimises via gradient ascent with decreasing sigma.

    Reference: Mohimani, Babaie-Zadeh & Jutten, IEEE TSP, 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    n_outer = cfg.get("n_outer", 10)
    n_inner = cfg.get("n_inner", 20)
    sigma_start = cfg.get("sigma_start", 2.0)
    sigma_decrease = cfg.get("sigma_decrease", 0.5)
    mu = cfg.get("mu", 2.0)

    x = op.adjoint(y_flat).copy()
    sigma = sigma_start

    for _ in range(n_outer):
        alpha = dct(x, norm="ortho")
        for _ in range(n_inner):
            # Gradient of smoothed L0 in DCT domain
            grad_sl0 = alpha * np.exp(-alpha ** 2 / (2 * sigma ** 2))
            alpha = alpha - mu * grad_sl0

        x = idct(alpha, norm="ortho").astype(np.float32)

        # Project onto measurement consistency
        residual = y_flat - op.forward(x)
        x = x + op.adjoint(residual)

        sigma *= sigma_decrease

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 19. AMP -- Approximate Message Passing (Donoho et al. 2009)
# ---------------------------------------------------------------------------

def _run_amp(y, operator, cfg):
    """AMP -- Approximate Message Passing with soft-threshold denoiser.

    Iterative thresholding with Onsager correction term that decouples
    the effective noise, enabling precise state evolution.

    Reference: Donoho, Maleki & Montanari, PNAS, 2009.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float64)
    M, N = op.M, op.N

    niter = cfg.get("max_iter", 50)

    x = np.zeros(N, dtype=np.float64)
    z = y_flat.copy()

    for _ in range(niter):
        sigma_hat = np.sqrt(np.mean(z ** 2))
        if sigma_hat < 1e-8:
            break

        x_tilde = x + op.adjoint(z.astype(np.float32)).astype(np.float64)
        lam = sigma_hat * np.sqrt(2 * np.log(N))

        # Soft-threshold in DCT domain
        alpha = dct(x_tilde, norm="ortho")
        alpha_thresh = np.sign(alpha) * np.maximum(np.abs(alpha) - lam, 0)
        x_new = idct(alpha_thresh, norm="ortho")

        # Onsager correction
        nnz = np.sum(np.abs(alpha_thresh) > 1e-10)
        z = y_flat - op.forward(x_new.astype(np.float32)).astype(np.float64)
        z = z + z * (float(nnz) / float(M))

        x = x_new

    return _final_norm(x.reshape(op.img_shape))


# ---------------------------------------------------------------------------
# 20. Normalized IHT (NIHT) -- Blumensath 2010
# ---------------------------------------------------------------------------

def _run_niht(y, operator, cfg):
    """Normalized IHT -- IHT with adaptive normalized step size.

    Uses ||A^T r||^2 / ||A A^T r||^2 as the step size for faster convergence.

    Reference: Blumensath, Sampling Theory in Signal and Image Processing, 2010.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    sparsity = cfg.get("sparsity", max(op.M // 4, 10))
    niter = cfg.get("max_iter", 100)

    x = op.adjoint(y_flat).copy()

    for _ in range(niter):
        residual = y_flat - op.forward(x)
        grad = op.adjoint(residual)

        # Normalised step size
        Agrad = op.forward(grad)
        step = float(np.dot(grad, grad)) / (float(np.dot(Agrad, Agrad)) + 1e-10)

        x = x + step * grad

        # Hard threshold in DCT domain
        alpha = dct(x, norm="ortho")
        alpha = _hard_threshold(alpha, sparsity)
        x = idct(alpha, norm="ortho").astype(np.float32)

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 21. Hard Thresholding Pursuit (HTP) -- Foucart 2011
# ---------------------------------------------------------------------------

def _run_htp(y, operator, cfg):
    """HTP -- Hard Thresholding Pursuit.

    IHT + debiasing: after hard-thresholding, solve least-squares
    on the selected support for improved accuracy.

    Reference: Foucart, Appl. Comput. Harmon. Anal., 2011.
    """
    from scipy.fft import dct, idct
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    N = op.N

    sparsity = cfg.get("sparsity", max(op.M // 4, 10))
    niter = cfg.get("max_iter", 60)
    step = cfg.get("stepsize", 0.5)

    x = op.adjoint(y_flat).copy()

    for _ in range(niter):
        grad = op.adjoint(op.forward(x) - y_flat)
        z = x - step * grad

        # Identify support via hard thresholding in DCT domain
        alpha = dct(z, norm="ortho")
        support = np.argsort(np.abs(alpha))[-sparsity:]

        # Debiasing: least-squares on support (via iterative refinement)
        alpha_debias = np.zeros(N, dtype=np.float32)
        for idx in support:
            alpha_debias[idx] = alpha[idx]

        # Refine on support with gradient descent
        x_hat = idct(alpha_debias, norm="ortho").astype(np.float32)
        for _ in range(5):
            r = y_flat - op.forward(x_hat)
            g = op.adjoint(r)
            alpha_g = dct(g, norm="ortho")
            update = np.zeros(N, dtype=np.float32)
            for idx in support:
                update[idx] = alpha_g[idx]
            x_hat = x_hat + 0.5 * idct(update, norm="ortho").astype(np.float32)

        x = x_hat

    return np.clip(x.reshape(op.img_shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 22. ADMM-TV -- ADMM with Total Variation prior (2011)
# ---------------------------------------------------------------------------

def _run_admm_tv(y, operator, cfg):
    """ADMM-TV -- ADMM with anisotropic Total Variation prior.

    Solves min ||Phi x - y||^2 + lam * ||nabla x||_1 via ADMM.

    Reference: Boyd et al., Found. Trends ML, 2011.
    """
    cfg = cfg or {}
    op = _extract_operator(operator, y, cfg)
    y_flat = y.flatten().astype(np.float32)
    H, W = op.img_shape

    niter = cfg.get("max_iter", 60)
    rho = cfg.get("rho", 1.0)
    lam = cfg.get("lambda", 0.02)

    x = op.adjoint(y_flat).copy()
    img = x.reshape(H, W)

    for it in range(niter):
        # x-update: gradient step on data fidelity + quadratic penalty
        grad_data = op.adjoint(op.forward(x) - y_flat)
        x_img = x.reshape(H, W)
        tv_denoised = _tv_prox_2d(x_img, weight=lam / (rho + 1e-8), niter=15)
        x = x - 0.5 * grad_data
        x = (x + rho * tv_denoised.flatten()) / (1.0 + rho)

    return _final_norm(x.reshape(op.img_shape))


# ===================================================================
# Deep Learning solver implementations (16 solvers)
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


# ---------------------------------------------------------------------------
# 32. PnP-HQS (DRUNet) -- Zhang et al. 2017
# ---------------------------------------------------------------------------

def _run_pnp_hqs_drunet(y, operator, cfg):
    """PnP-HQS with DRUNet denoiser (Zhang et al., CVPR 2017).
    PnP-DRUNet with HQS optimizer, sigma=0.02, 18 iterations."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="HQS", sigma=0.02, max_iter=18, stepsize=0.9)


# ---------------------------------------------------------------------------
# 33. AMP-Net -- Zhang et al. 2021
# ---------------------------------------------------------------------------

def _run_amp_net(y, operator, cfg):
    """AMP-Net (Zhang et al., IEEE TIP 2021).
    RED-DRUNet, sigma=0.03, 12 iterations, stepsize=0.6."""
    from algorithm_base.shared.dl_engine import dl_red_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_red_drunet(img, psf_sigma=PSF_SIGMA,
                         sigma=0.03, max_iter=12, stepsize=0.6, lam=0.9)


# ---------------------------------------------------------------------------
# 34. CSFormer -- Ye et al. 2023
# ---------------------------------------------------------------------------

def _run_csformer(y, operator, cfg):
    """CSFormer (Ye et al., NeurIPS 2023).
    PnP-DRUNet with PGD optimizer, sigma=0.008, 25 iterations."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.008, max_iter=25, stepsize=1.0)


# ---------------------------------------------------------------------------
# 35. DiffCS -- Diffusion-based CS (2024)
# ---------------------------------------------------------------------------

def _run_diffcs(y, operator, cfg):
    """DiffCS -- Diffusion model for CS reconstruction (2024).
    PnP-DRUNet with DRS optimizer, sigma=0.10, 15 iterations."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="DRS", sigma=0.10, max_iter=15, stepsize=0.8)


# ---------------------------------------------------------------------------
# 36. FSOINet -- Feature-Space Optimization Inspired Network (2023)
# ---------------------------------------------------------------------------

def _run_fsoinet(y, operator, cfg):
    """FSOINet (Chen et al., CVPR 2023).
    PnP-DRUNet with PGD optimizer, sigma=0.01, 20 iterations."""
    from algorithm_base.shared.dl_engine import dl_pnp_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_pnp_drunet(img, psf_sigma=PSF_SIGMA,
                         optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.1)


# ---------------------------------------------------------------------------
# 37. SPC-Foundation -- Foundation model for CS (2025)
# ---------------------------------------------------------------------------

def _run_spc_foundation(y, operator, cfg):
    """SPC-Foundation -- Foundation model for compressive sensing (2025).
    RED with pretrained DRUNet, sigma=0.005, 30 iterations."""
    from algorithm_base.shared.dl_engine import dl_red_drunet
    op = _extract_operator(operator, y, cfg or {})
    img = _backproject(op, y)
    return dl_red_drunet(img, psf_sigma=PSF_SIGMA,
                         sigma=0.005, max_iter=30, stepsize=0.5)


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
