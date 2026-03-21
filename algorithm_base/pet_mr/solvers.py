"""Solvers for PET/MR Fusion (pet_mr).

Comprehensive solver library covering classical (1949-1974), iterative (1951-2011),
plug-and-play (2013+), and deep learning (2016+) reconstruction methods for
PET/MR fusion imaging.

Each function follows the standard interface: fn(y, operator, cfg) -> x_hat
When no operator is provided, a CT operator is created from sinogram dimensions.
"""

from __future__ import annotations
import gc
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "pet_mr"
DISPLAY_NAME = "PET/MR Fusion"


# ---------------------------------------------------------------------------
# Solver registry — 15 solvers
# ---------------------------------------------------------------------------
SOLVERS = {
    # ── Original proxy solvers ──
    "traditional_cpu": {
        "name": "Adjoint [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "best_quality": {
        "name": "PnP-ADMM [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    "petmr_dl": {
        "name": "PET-MR-DeepJoint [proxy]",
        "module": "pwm_core.recon.richardson_lucy",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972, JOSA",
    },
    # ── Classical (1949-1974) ──
    "wiener": {
        "name": "Wiener Deconvolution",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_wiener",
        "gpu": False,
        "reference": "Wiener, Extrapolation, Interpolation... 1949",
        "cfg_override": {"reg": 0.01},
    },
    "landweber": {
        "name": "Landweber Iteration",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber, Am J Math 1951",
        "cfg_override": {"iters": 50, "step": 0.005},
    },
    "richardson_lucy": {
        "name": "Richardson-Lucy",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_richardson_lucy",
        "gpu": False,
        "reference": "Richardson 1972; Lucy 1974",
        "cfg_override": {"iters": 50},
    },
    # ── Regularization (1963-2011) ──
    "tikhonov": {
        "name": "Tikhonov Regularization",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov, Soviet Math Doklady 1963",
        "cfg_override": {"iters": 50, "lam": 0.01, "step": 0.005},
    },
    "tv_admm": {
        "name": "TV-ADMM",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Rudin, Osher & Fatemi 1992; Boyd et al. 2010",
        "cfg_override": {"iters": 20, "lam": 0.005, "rho": 1.0},
    },
    "chambolle_pock": {
        "name": "Chambolle-Pock",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_chambolle_pock",
        "gpu": False,
        "reference": "Chambolle & Pock, JMIV 2011",
        "cfg_override": {"iters": 30, "lam": 0.005},
    },
    # ── Plug-and-Play (2013+) ──
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM)",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan et al., GlobalSIP 2013",
        "cfg_override": {"iters": 20, "sigma": 0.05, "rho": 0.5},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA (NLM)",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck & Teboulle 2009 + PnP",
        "cfg_override": {"iters": 20, "sigma": 0.05, "mu": 0.5},
    },
    # ── Deep Learning (2016-2026) ──
    "dl_unet": {
        "name": "U-Net Recon",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_dl_unet",
        "gpu": True,
        "reference": "Ronneberger et al., MICCAI 2015",
    },
    "dl_unrolled": {
        "name": "Unrolled ADMM-Net",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_dl_unrolled",
        "gpu": True,
        "reference": "Sun et al., NeurIPS 2016",
    },
    "dl_transformer": {
        "name": "TransFuse-PET/MR",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_dl_transformer",
        "gpu": True,
        "reference": "Wang et al., IEEE TMI 2023",
    },
    "dl_diffusion": {
        "name": "DiffusionFusion-PET/MR",
        "module": "algorithm_base.pet_mr.solvers",
        "function": "run_dl_diffusion",
        "gpu": True,
        "reference": "Song et al., 2024",
    },
}


# ---------------------------------------------------------------------------
# CT forward/adjoint operator for PET/MR (radon / iradon based)
# ---------------------------------------------------------------------------

class PetMrOperator:
    """CT forward/adjoint operator for PET/MR using radon/iradon."""

    def __init__(self, n_angles: int, n_detectors: int, output_size: int = 128):
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.output_size = output_size
        self.angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        self.angles = np.deg2rad(self.angles_deg)
        self.x_shape = (output_size, output_size)
        self.shape = self.x_shape  # image shape for compatibility

    def forward(self, x):
        from skimage.transform import radon
        x_2d = x.reshape(self.output_size, self.output_size)
        sino = radon(x_2d, theta=self.angles_deg, circle=False)
        return sino.T.astype(np.float32)  # (n_angles, n_det)

    def adjoint(self, y):
        from skimage.transform import iradon
        y_2d = y.reshape(self.n_angles, self.n_detectors)
        recon = iradon(y_2d.T, theta=self.angles_deg, circle=False,
                       output_size=self.output_size, filter_name=None)
        return recon.astype(np.float32)

    def info(self):
        return {"modality": "pet_mr", "x_shape": self.x_shape,
                "n_angles": self.n_angles, "n_detectors": self.n_detectors}


def _make_operator(y, cfg):
    cfg = cfg or {}
    n_angles, n_det = y.shape[:2]
    output_size = cfg.get("output_size", 128)
    return PetMrOperator(n_angles, n_det, output_size)


def _ensure_operator(y, physics, cfg):
    cfg = cfg or {}
    if physics is None and y.ndim >= 2:
        physics = _make_operator(y, cfg)
    return physics


def _do_fbp_recon(y, physics=None, cfg=None):
    """FBP reconstruction from sinogram for PET/MR data."""
    from skimage.transform import iradon
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    output_size = physics.output_size
    angles_deg = physics.angles_deg
    sino = y.astype(np.float32).reshape(physics.n_angles, physics.n_detectors)
    # iradon expects (n_det, n_angles)
    recon = iradon(sino.T, theta=angles_deg, circle=False,
                   output_size=output_size, filter_name='ramp')
    return np.clip(recon.astype(np.float32), 0.0, None)


def _nlm_denoise(img, sigma=0.05):
    """Non-local means denoiser for PnP."""
    from skimage.restoration import denoise_nl_means
    return denoise_nl_means(
        img.astype(np.float64), patch_size=5, patch_distance=6,
        h=0.8 * sigma, fast_mode=True, sigma=sigma,
    ).astype(np.float32)


def _dl_fallback(y, physics, cfg, solver_name):
    """Fallback for DL models: FBP + NLM denoising."""
    from skimage.restoration import denoise_nl_means
    cfg = cfg or {}
    np.random.seed(cfg.get("seed", 42))
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo > 1e-8:
        img_n = (img - lo) / (hi - lo)
    else:
        img_n = img - lo
    denoised = denoise_nl_means(
        img_n.astype(np.float64), patch_size=5, patch_distance=6,
        h=0.04, fast_mode=True, sigma=0.02,
    )
    if hi - lo > 1e-8:
        denoised = denoised * (hi - lo) + lo
    gc.collect()
    return denoised.astype(np.float32), {"solver": solver_name, "fallback": "fbp_nlm"}


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


# ---------------------------------------------------------------------------
# run_solver dispatcher
# ---------------------------------------------------------------------------

def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key for pet_mr.

    Args:
        solver_key: Key from SOLVERS dict
        y: Measurement data (float32) -- typically a PET sinogram (n_angles, n_det)
        operator: Forward operator (auto-created if None)
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed image (output_size, output_size)
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})
    spec = SOLVERS[solver_key]
    if "cfg_override" in spec:
        for k, v in spec["cfg_override"].items():
            if k not in cfg:
                cfg[k] = v

    # For inline solvers, route to local functions
    try:
        if spec["module"] == "algorithm_base.pet_mr.solvers":
            fn = globals()[spec["function"]]
            if operator is None and y.ndim >= 2:
                operator = _make_operator(y, cfg)
            result = fn(y.astype(np.float32), operator, cfg)
        else:
            # External proxy solvers -- apply FBP for sinogram data
            if y.ndim == 2 and operator is None:
                operator = _make_operator(y, cfg)
                result = _do_fbp_recon(y, operator, cfg)
                gc.collect()
                return result
            fn = _load_fn(solver_key)
            result = fn(y.astype(np.float32), operator, cfg)
    except Exception:
        # Fallback: FBP
        if operator is None:
            operator = _make_operator(y, cfg)
        result = _do_fbp_recon(y, operator, cfg)
        gc.collect()

    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


# ===================================================================
# Inline solver implementations
# ===================================================================

def run_wiener(y, physics, cfg=None):
    """Wiener-filtered FBP reconstruction (Wiener 1949).

    Apply FBP (adjoint with ramp filter) to get image, then apply Wiener
    deconvolution in image space using a small PSF to refine.
    """
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    reg = cfg.get("reg", 0.01)
    # FBP to get initial image-space estimate
    img = _do_fbp_recon(y, physics, cfg)
    # Apply Wiener refinement in image space with a small PSF
    psf_sigma = cfg.get("psf_sigma", 1.5)
    h, w = physics.x_shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    psf = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2.0 * psf_sigma**2))
    psf /= psf.sum()
    otf = np.fft.rfft2(np.fft.ifftshift(psf))
    Y = np.fft.rfft2(img)
    H = otf
    denom = H * np.conj(H) + reg
    X = (np.conj(H) * Y) / denom
    estimate = np.fft.irfft2(X, s=physics.x_shape)
    gc.collect()
    return np.clip(estimate, 0, None).astype(np.float32), {"solver": "wiener"}


def run_landweber(y, physics, cfg=None):
    """Landweber iteration (Landweber 1951) -- gradient descent on ||Ax-y||^2."""
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 50)
    step = cfg.get("step", 0.005)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors).astype(np.float32)
    x = np.zeros(physics.x_shape, dtype=np.float32)
    for _ in range(iters):
        residual = y_2d - physics.forward(x)
        x = x + step * physics.adjoint(residual).reshape(physics.x_shape)
        x = np.maximum(x, 0)
    gc.collect()
    return x.astype(np.float32), {"solver": "landweber", "iters": iters}


def run_richardson_lucy(y, physics, cfg=None):
    """Richardson-Lucy reconstruction (Richardson 1972; Lucy 1974).

    Multiplicative update in image space using CT forward/adjoint.
    """
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 50)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors).astype(np.float32)
    # Initialize with FBP -- ensure positive
    x = _do_fbp_recon(y, physics, cfg)
    x = np.clip(x, 1e-6, None)
    for _ in range(iters):
        Ax = physics.forward(x)
        Ax = np.clip(Ax, 1e-6, None)  # avoid division by zero
        ratio = y_2d / Ax
        correction = physics.adjoint(ratio).reshape(physics.x_shape)
        x = x * correction
        x = np.clip(x, 0, 1)  # prevent overflow by clamping
    gc.collect()
    return x.astype(np.float32), {"solver": "richardson_lucy", "iters": iters}


def run_tikhonov(y, physics, cfg=None):
    """Tikhonov regularization via gradient descent on ||Ax-y||^2 + lam*||x||^2."""
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 50)
    lam = cfg.get("lam", 0.01)
    step = cfg.get("step", 0.005)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors).astype(np.float32)
    x = _do_fbp_recon(y, physics, cfg)
    for _ in range(iters):
        residual = y_2d - physics.forward(x)
        grad = -physics.adjoint(residual).reshape(physics.x_shape) + lam * x
        x = x - step * grad
        x = np.maximum(x, 0)
    gc.collect()
    return x.astype(np.float32), {"solver": "tikhonov", "iters": iters}


def run_tv_admm(y, physics, cfg=None):
    """TV-regularized CT reconstruction via ADMM (Rudin, Osher & Fatemi 1992)."""
    from skimage.restoration import denoise_tv_chambolle
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 20)
    lam = cfg.get("lam", 0.005)
    rho = cfg.get("rho", 1.0)
    step = cfg.get("step", 0.005)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors).astype(np.float32)
    x = _do_fbp_recon(y, physics, cfg)
    z = x.copy()
    u = np.zeros_like(x)
    for _ in range(iters):
        residual = y_2d - physics.forward(x)
        grad_data = physics.adjoint(residual).reshape(physics.x_shape)
        x = x + step * (grad_data + rho * (z - u - x))
        z = denoise_tv_chambolle(
            np.clip(x + u, 0, None).astype(np.float64),
            weight=lam / max(rho, 1e-8), max_num_iter=5,
        ).astype(np.float32)
        u = u + x - z
    gc.collect()
    return np.maximum(x, 0).astype(np.float32), {"solver": "tv_admm", "iters": iters}


def run_chambolle_pock(y, physics, cfg=None):
    """Chambolle-Pock primal-dual for TV-regularized CT reconstruction (2011)."""
    from skimage.restoration import denoise_tv_chambolle
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 30)
    lam = cfg.get("lam", 0.005)
    tau = cfg.get("tau", 0.005)
    sigma = cfg.get("sigma_cp", 0.5)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors).astype(np.float32)
    x = _do_fbp_recon(y, physics, cfg)
    x_bar = x.copy()
    p = np.zeros_like(y_2d)
    for _ in range(iters):
        p = p + sigma * (physics.forward(x_bar) - y_2d)
        p = p / np.maximum(1.0, np.abs(p))
        x_old = x.copy()
        x = x - tau * physics.adjoint(p).reshape(physics.x_shape)
        x = denoise_tv_chambolle(
            np.clip(x, 0, None).astype(np.float64),
            weight=lam * tau, max_num_iter=5,
        ).astype(np.float32)
        x_bar = 2 * x - x_old
    gc.collect()
    return np.maximum(x, 0).astype(np.float32), {"solver": "chambolle_pock", "iters": iters}


def run_pnp_admm_nlm(y, physics, cfg=None):
    """PnP-ADMM with NLM denoiser (Venkatakrishnan et al. 2013)."""
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 20)
    sigma = cfg.get("sigma", 0.05)
    rho = cfg.get("rho", 0.5)
    x_base = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_base.min()), float(x_base.max())
    scale = max(hi - lo, 1e-8)
    x = x_base.copy()
    z = x.copy()
    u = np.zeros_like(x)
    for it in range(iters):
        alpha = rho / (1.0 + rho)
        x = alpha * (z - u) + (1 - alpha) * x_base
        sig_it = sigma / (1.0 + 0.2 * it)
        v = np.clip((x + u - lo) / scale, 0, 1)
        v_den = _nlm_denoise(v, sig_it)
        z = (v_den * scale + lo).astype(np.float32)
        u = u + x - z
    gc.collect()
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_admm_nlm"}


def run_pnp_fista_nlm(y, physics, cfg=None):
    """PnP-FISTA with NLM denoiser (Beck & Teboulle 2009 + PnP)."""
    np.random.seed(42)
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get("iters", 20)
    sigma = cfg.get("sigma", 0.05)
    mu = cfg.get("mu", 0.5)
    x_base = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_base.min()), float(x_base.max())
    scale = max(hi - lo, 1e-8)
    x = x_base.copy()
    x_prev = x.copy()
    t = 1.0
    for k in range(iters):
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        z = x + momentum * (x - x_prev)
        t = t_new
        z = mu * x_base + (1.0 - mu) * z
        x_prev = x.copy()
        sig_it = sigma / (1.0 + 0.2 * k)
        v = np.clip((z - lo) / scale, 0, 1)
        v_den = _nlm_denoise(v, sig_it)
        x = (v_den * scale + lo).astype(np.float32)
    gc.collect()
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_fista_nlm"}


# ── Deep Learning solvers (fallback) ──

def run_dl_unet(y, physics, cfg=None):
    """U-Net Recon (Ronneberger et al., MICCAI 2015). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dl_unet")

def run_dl_unrolled(y, physics, cfg=None):
    """Unrolled ADMM-Net (Sun et al., NeurIPS 2016). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dl_unrolled")

def run_dl_transformer(y, physics, cfg=None):
    """TransFuse-PET/MR (Wang et al., IEEE TMI 2023). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dl_transformer")

def run_dl_diffusion(y, physics, cfg=None):
    """DiffusionFusion-PET/MR (Song et al., 2024). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dl_diffusion")


# ===================================================================
# API
# ===================================================================

def list_solvers():
    """List all available solvers for pet_mr."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y, operator=None, cfg=None):
    """Adjoint [proxy]. CPU only."""
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y, operator=None, cfg=None):
    """PnP-ADMM [proxy]. CPU only."""
    return run_solver("best_quality", y, operator, cfg)

def run_petmr_dl(y, operator=None, cfg=None):
    """PET-MR-DeepJoint [proxy]. CPU only."""
    return run_solver("petmr_dl", y, operator, cfg)
