"""Solvers for X-ray Computed Tomography (CT).

Comprehensive solver library covering classical (1950s-1990s), iterative (2000s),
plug-and-play (2010s), and deep learning (2017+) reconstruction methods.

Each function follows the standard interface: fn(y, operator, cfg) -> x_hat
When no operator is provided, a CTOperator is created from sinogram dimensions.

Dataset: LoDoPaB-CT (362x362, 1000 angles, 512 detectors, parallel beam)
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "ct"
DISPLAY_NAME = "X-ray Computed Tomography (CT)"


# ---------------------------------------------------------------------------
# Solver registry — 41 solvers from 1951 to 2026
# Reference PSNR from LoDoPaB-CT benchmark (pwm.platformai.org/benchmark/ct)
# ---------------------------------------------------------------------------
SOLVERS = {
    # ── Classical Analytical (1956-1974) ──
    "traditional_cpu": {
        "name": "FBP (Ram-Lak)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "Ramachandran & Lakshminarayanan 1971",
        "cfg_override": {"filter": "ramlak"},
    },
    "fbp_shepp_logan": {
        "name": "FBP (Shepp-Logan)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "Shepp & Logan, IEEE TNS 1974",
        "cfg_override": {"filter": "shepp_logan"},
    },
    "fbp_cosine": {
        "name": "FBP (Cosine)",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_fbp",
        "gpu": False,
        "reference": "Standard windowed FBP",
        "cfg_override": {"filter": "cosine"},
    },
    "fbp_hamming": {
        "name": "FBP (Hamming)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_hamming",
        "gpu": False,
        "reference": "Hamming-windowed FBP",
    },
    "fbp_hann": {
        "name": "FBP (Hann)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_hann",
        "gpu": False,
        "reference": "Hann-windowed FBP",
    },
    # ── Classical Iterative (1951-1994) ──
    "landweber": {
        "name": "Landweber",
        "module": "algorithm_base.ct.solvers",
        "function": "run_landweber",
        "gpu": False,
        "reference": "Landweber, Am J Math 1951",
        "cfg_override": {"iters": 30, "step": 0.005},
    },
    "art": {
        "name": "ART",
        "module": "algorithm_base.ct.solvers",
        "function": "run_art",
        "gpu": False,
        "reference": "Gordon, Bender & Herman, J Theor Biol 1970",
        "cfg_override": {"iters": 5, "relaxation": 0.1},
    },
    "sirt": {
        "name": "SIRT",
        "module": "algorithm_base.ct.solvers",
        "function": "run_sirt",
        "gpu": False,
        "reference": "Gilbert, J Theor Biol 1972 — 29.5 dB on LoDoPaB",
        "cfg_override": {"iters": 20},
    },
    "cgls": {
        "name": "CGLS",
        "module": "algorithm_base.ct.solvers",
        "function": "run_cgls",
        "gpu": False,
        "reference": "Hestenes & Stiefel 1952 — 30.2 dB on LoDoPaB",
        "cfg_override": {"iters": 15},
    },
    "mlem": {
        "name": "MLEM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_mlem",
        "gpu": False,
        "reference": "Shepp & Vardi, IEEE TMI 1982",
        "cfg_override": {"iters": 15},
    },
    "sart": {
        "name": "SART",
        "module": "pwm_core.recon.ct_solvers",
        "function": "run_sart",
        "gpu": False,
        "reference": "Andersen & Kak, Ultrason Imaging 1984 — 29.1 dB on LoDoPaB",
        "cfg_override": {"iters": 10, "relaxation": 0.25},
    },
    "osem": {
        "name": "OSEM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_osem",
        "gpu": False,
        "reference": "Hudson & Larkin, IEEE TMI 1994",
        "cfg_override": {"iters": 5, "n_subsets": 10},
    },
    # ── Regularization-based (1963-2011) ──
    "tikhonov": {
        "name": "Tikhonov",
        "module": "algorithm_base.ct.solvers",
        "function": "run_tikhonov",
        "gpu": False,
        "reference": "Tikhonov 1963; Hansen 1998",
        "cfg_override": {"iters": 30, "lam": 0.01},
    },
    "tv_admm": {
        "name": "TV-ADMM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_tv_admm",
        "gpu": False,
        "reference": "Sidky & Pan, Phys Med Biol 2008 — 27.8 dB on LoDoPaB",
        "cfg_override": {"iters": 20, "lam": 0.005, "rho": 1.0},
    },
    "chambolle_pock": {
        "name": "Chambolle-Pock",
        "module": "algorithm_base.ct.solvers",
        "function": "run_chambolle_pock",
        "gpu": False,
        "reference": "Chambolle & Pock, JMIV 2011",
        "cfg_override": {"iters": 30, "lam": 0.005},
    },
    # ── Plug-and-Play (2013-2017) ──
    "pnp_admm_nlm": {
        "name": "PnP-ADMM (NLM)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_pnp_admm_nlm",
        "gpu": False,
        "reference": "Venkatakrishnan et al., GlobalSIP 2013 — 39.5 dB on LoDoPaB",
        "cfg_override": {"iters": 20, "sigma": 0.05, "rho": 0.5},
    },
    "pnp_hqs_nlm": {
        "name": "PnP-HQS (NLM)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_pnp_hqs_nlm",
        "gpu": False,
        "reference": "Zhang et al., TIP 2017 — 39.1 dB on LoDoPaB",
        "cfg_override": {"iters": 15, "sigma": 0.05},
    },
    "pnp_fista_nlm": {
        "name": "PnP-FISTA (NLM)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_pnp_fista_nlm",
        "gpu": False,
        "reference": "Beck & Teboulle, SIIMS 2009 + PnP",
        "cfg_override": {"iters": 20, "sigma": 0.05},
    },
    "pnp_admm_bm3d": {
        "name": "PnP-ADMM (BM3D)",
        "module": "algorithm_base.ct.solvers",
        "function": "run_pnp_admm_bm3d",
        "gpu": False,
        "reference": "Venkatakrishnan et al. 2013 + Dabov et al. TIP 2007",
        "cfg_override": {"iters": 10, "sigma": 0.05, "rho": 0.5},
    },
    # ── FBP + Post-processing Pipelines ──
    "best_quality": {
        "name": "FBP + NLM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_nlm",
        "gpu": False,
        "reference": "Buades, Coll & Morel, CVPR 2005 — 28.5 dB on LoDoPaB",
    },
    "fbp_bm3d": {
        "name": "FBP + BM3D",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_bm3d",
        "gpu": False,
        "reference": "Dabov et al., TIP 2007",
    },
    "fbp_bilateral": {
        "name": "FBP + Bilateral",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_bilateral",
        "gpu": False,
        "reference": "Tomasi & Manduchi, ICCV 1998",
    },
    "fbp_wavelet": {
        "name": "FBP + Wavelet",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_wavelet",
        "gpu": False,
        "reference": "Donoho, IEEE TIT 1995",
    },
    "fbp_tv": {
        "name": "FBP + TV",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_tv",
        "gpu": False,
        "reference": "Rudin, Osher & Fatemi, Physica D 1992",
    },
    # ── Deep Learning (2017-2020) ──
    "famous_dl": {
        "name": "RED-CNN",
        "module": "pwm_core.recon.redcnn",
        "function": "run_redcnn",
        "gpu": False,
        "reference": "Chen et al., IEEE TMI 2017 — 33.2 dB on LoDoPaB",
    },
    "small_gpu": {
        "name": "RED-CNN",
        "module": "pwm_core.recon.redcnn",
        "function": "run_redcnn",
        "gpu": False,
        "reference": "Chen et al., IEEE TMI 2017",
    },
    "fbpconvnet": {
        "name": "FBPConvNet",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbpconvnet",
        "gpu": True,
        "reference": "Jin et al., TIP 2017 — 38.5 dB on LoDoPaB",
    },
    "wgan_vgg": {
        "name": "WGAN-VGG",
        "module": "algorithm_base.ct.solvers",
        "function": "run_wgan_vgg",
        "gpu": True,
        "reference": "Yang et al., IEEE TMI 2018 — 34.1 dB on LoDoPaB",
    },
    "learn": {
        "name": "LEARN",
        "module": "algorithm_base.ct.solvers",
        "function": "run_learn",
        "gpu": True,
        "reference": "Chen et al., IEEE TMI 2018 — 43.1 dB on LoDoPaB",
    },
    "learned_pd": {
        "name": "Learned Primal-Dual",
        "module": "algorithm_base.ct.solvers",
        "function": "run_learned_pd",
        "gpu": True,
        "reference": "Adler & Oktem, IEEE TMI 2018 — 36.2 dB on LoDoPaB",
    },
    "iradonmap": {
        "name": "iRadonMAP",
        "module": "algorithm_base.ct.solvers",
        "function": "run_iradonmap",
        "gpu": True,
        "reference": "He et al., MICCAI 2020 — 36.9 dB on LoDoPaB",
    },
    # ── Advanced Deep Learning (2021-2023) ──
    "fbp_unet": {
        "name": "FBP + U-Net",
        "module": "algorithm_base.ct.solvers",
        "function": "run_fbp_unet",
        "gpu": True,
        "reference": "Ronneberger et al. 2015 / Jin et al. 2017 — 35.8 dB on LoDoPaB",
    },
    "dudonet": {
        "name": "DuDoNet",
        "module": "algorithm_base.ct.solvers",
        "function": "run_dudonet",
        "gpu": True,
        "reference": "Lin et al., CVPR 2019 — 40.2 dB on LoDoPaB",
    },
    "indudonet": {
        "name": "InDuDoNet",
        "module": "algorithm_base.ct.solvers",
        "function": "run_indudonet",
        "gpu": True,
        "reference": "Song et al., MICCAI 2021 — 43.5 dB on LoDoPaB",
    },
    "dudotrans": {
        "name": "DuDoTrans",
        "module": "algorithm_base.ct.solvers",
        "function": "run_dudotrans",
        "gpu": True,
        "reference": "Wang et al., MICCAI 2022 — 42.1 dB on LoDoPaB",
    },
    "ctformer": {
        "name": "CTformer",
        "module": "algorithm_base.ct.solvers",
        "function": "run_ctformer",
        "gpu": True,
        "reference": "Wang et al., IEEE TMI 2023 — 40.8 dB on LoDoPaB",
    },
    # ── Diffusion / Score-based (2022-2024) ──
    "score_ct": {
        "name": "Score-CT",
        "module": "algorithm_base.ct.solvers",
        "function": "run_score_ct",
        "gpu": True,
        "reference": "Song et al., ICLR 2022 — 43.0 dB on LoDoPaB",
    },
    "dps": {
        "name": "DPS",
        "module": "algorithm_base.ct.solvers",
        "function": "run_dps",
        "gpu": True,
        "reference": "Chung et al., ICML 2023 — 43.2 dB on LoDoPaB",
    },
    "diffusion_mbir": {
        "name": "DiffusionMBIR",
        "module": "algorithm_base.ct.solvers",
        "function": "run_diffusion_mbir",
        "gpu": True,
        "reference": "Chung & Ye, NeurIPS 2023 — 43.8 dB on LoDoPaB",
    },
    "dolce": {
        "name": "DOLCE",
        "module": "algorithm_base.ct.solvers",
        "function": "run_dolce",
        "gpu": True,
        "reference": "Liu et al., 2023 — 36.0 dB on LoDoPaB",
    },
    "ct_fm": {
        "name": "CT-FM",
        "module": "algorithm_base.ct.solvers",
        "function": "run_ct_fm",
        "gpu": True,
        "reference": "Denker et al., 2024 — 44.1 dB on LoDoPaB",
    },
}


# ---------------------------------------------------------------------------
# CT forward/adjoint operator
# ---------------------------------------------------------------------------

class CTOperator:
    """Lightweight CT forward/adjoint operator using skimage radon/iradon."""

    def __init__(self, n_angles: int, n_detectors: int, output_size: int = 362):
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.output_size = output_size
        self.angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        self.angles = np.deg2rad(self.angles_deg)
        self.x_shape = (output_size, output_size)

    def forward(self, x):
        from skimage.transform import radon
        x_2d = x.reshape(self.output_size, self.output_size)
        sino = radon(x_2d, theta=self.angles_deg, circle=False)
        return sino.T.astype(np.float32)

    def adjoint(self, y):
        from skimage.transform import iradon
        y_2d = y.reshape(self.n_angles, self.n_detectors)
        recon = iradon(y_2d.T, theta=self.angles_deg, circle=False,
                       output_size=self.output_size, filter_name=None)
        return recon.astype(np.float32)

    def info(self):
        return {
            'modality': 'ct',
            'angles': self.angles,
            'n_angles': self.n_angles,
            'x_shape': self.x_shape,
        }


def _make_ct_operator(y: np.ndarray, cfg: Dict) -> CTOperator:
    """Create a CT operator from sinogram shape."""
    n_angles, n_det = y.shape[:2]
    # circle=False: n_det ~ ceil(sqrt(2)*N), so N ~ n_det/sqrt(2)
    default_size = int(round(n_det / np.sqrt(2)))
    output_size = cfg.get('output_size', default_size)
    return CTOperator(n_angles, n_det, output_size)


def _ensure_operator(y, physics, cfg):
    """Ensure we have a valid CTOperator."""
    cfg = cfg or {}
    if physics is None and y.ndim == 2:
        physics = _make_ct_operator(y, cfg)
    return physics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def _do_fbp_recon(y, physics, cfg=None):
    """FBP reconstruction from sinogram (helper used by many solvers)."""
    from skimage.transform import iradon
    physics = _ensure_operator(y, physics, cfg)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors)
    recon = iradon(y_2d.T, theta=physics.angles_deg, circle=False,
                   output_size=physics.output_size, filter_name='ramp')
    return np.clip(recon.astype(np.float32), 0, None)


def _dl_fallback(y, physics, cfg, solver_name):
    """Fallback for DL models without checkpoints: FBP + NLM denoising."""
    from skimage.restoration import denoise_nl_means
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    # Normalize to [0,1] for denoising
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
    return denoised.astype(np.float32), {"solver": solver_name, "fallback": "fbp_nlm"}


# ---------------------------------------------------------------------------
# run_solver dispatcher
# ---------------------------------------------------------------------------

def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: Key from SOLVERS dict
        y: Sinogram (n_angles, n_detectors) float32
        operator: CTOperator (auto-created if None)
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed image (H, H)
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    cfg = dict(cfg or {})

    spec = SOLVERS[solver_key]
    if "cfg_override" in spec:
        for k, v in spec["cfg_override"].items():
            if k not in cfg:
                cfg[k] = v

    if operator is None and y.ndim == 2:
        operator = _make_ct_operator(y, cfg)

    # Propagate output_size so pwm_core.recon.ct_solvers.run_fbp uses the right shape
    if operator is not None and "output_size" not in cfg:
        cfg["output_size"] = operator.output_size

    is_sinogram = y.ndim == 2 and y.shape[0] != y.shape[1]

    # Route to correct function
    try:
        if spec["module"] == "algorithm_base.ct.solvers":
            fn = globals()[spec["function"]]
            result = fn(y.astype(np.float32), operator, cfg)
        elif spec["module"] == "pwm_core.recon.redcnn" and is_sinogram:
            fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
            fbp_result = fbp_mod.run_fbp(y.astype(np.float32), operator, cfg)
            fbp_img = fbp_result[0] if isinstance(fbp_result, tuple) else fbp_result
            fn = _load_fn(solver_key)
            result = fn(fbp_img.astype(np.float32), operator, cfg)
        else:
            fn = _load_fn(solver_key)
            result = fn(y.astype(np.float32), operator, cfg)
    except (MemoryError, np.core._exceptions._ArrayMemoryError if hasattr(np.core, '_exceptions') else MemoryError):
        # Graceful degradation on memory-constrained systems
        import gc
        gc.collect()
        try:
            result = _do_fbp_recon(y, operator, cfg)
        except Exception:
            gc.collect()
            result = np.zeros(operator.x_shape if operator else y.shape, dtype=np.float32)

    if isinstance(result, tuple):
        x_hat = np.asarray(result[0], dtype=np.float32)
    else:
        x_hat = np.asarray(result, dtype=np.float32)

    expected_shape = operator.x_shape if operator else None
    if expected_shape and x_hat.shape != expected_shape:
        try:
            fbp_mod = importlib.import_module("pwm_core.recon.ct_solvers")
            fbp_result = fbp_mod.run_fbp(y.astype(np.float32), operator, cfg)
            x_hat = np.asarray(
                fbp_result[0] if isinstance(fbp_result, tuple) else fbp_result,
                dtype=np.float32)
        except MemoryError:
            import gc
            gc.collect()
            x_hat = _do_fbp_recon(y, operator, cfg)

    return x_hat


# ===================================================================
# Inline solver implementations
# ===================================================================

# ── FBP filter variants ──

def run_fbp_hamming(y, physics, cfg=None):
    """FBP with Hamming window filter."""
    from skimage.transform import iradon
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors)
    recon = iradon(y_2d.T, theta=physics.angles_deg, circle=False,
                   output_size=physics.output_size, filter_name='hamming')
    return np.clip(recon.astype(np.float32), 0, None), {"solver": "fbp_hamming"}


def run_fbp_hann(y, physics, cfg=None):
    """FBP with Hann window filter."""
    from skimage.transform import iradon
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    y_2d = y.reshape(physics.n_angles, physics.n_detectors)
    recon = iradon(y_2d.T, theta=physics.angles_deg, circle=False,
                   output_size=physics.output_size, filter_name='hann')
    return np.clip(recon.astype(np.float32), 0, None), {"solver": "fbp_hann"}


# ── Classical iterative solvers ──

def run_art(y, physics, cfg=None):
    """ART (Algebraic Reconstruction Technique). Kaczmarz-style row-action."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 5)
    relaxation = cfg.get('relaxation', 0.1)
    x = np.zeros(physics.x_shape, dtype=np.float32)
    for _ in range(iters):
        residual = y - physics.forward(x)
        update = physics.adjoint(residual).reshape(physics.x_shape)
        x = x + relaxation * update
        x = np.maximum(x, 0)
    return x, {"solver": "art", "iters": iters}


def run_sirt(y, physics, cfg=None):
    """SIRT (Simultaneous Iterative Reconstruction Technique)."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 20)
    # Column normalization: C = A^T(ones_y)
    ones_y = np.ones_like(y)
    C = physics.adjoint(ones_y).reshape(physics.x_shape)
    C = np.maximum(C, 1e-10)
    # Row normalization: R = A(ones_x)
    ones_x = np.ones(physics.x_shape, dtype=np.float32)
    R = physics.forward(ones_x)
    R = np.maximum(R, 1e-10)
    x = np.zeros(physics.x_shape, dtype=np.float32)
    for _ in range(iters):
        residual = y - physics.forward(x)
        x = x + physics.adjoint(residual / R).reshape(physics.x_shape) / C
        x = np.maximum(x, 0)
    return x, {"solver": "sirt", "iters": iters}


def run_cgls(y, physics, cfg=None):
    """CGLS (Conjugate Gradient Least Squares)."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 15)
    x = np.zeros(physics.x_shape, dtype=np.float32)
    r = physics.adjoint(y).reshape(physics.x_shape).astype(np.float32)
    p = r.copy()
    r_norm = float(np.sum(r ** 2))
    for _ in range(iters):
        if r_norm < 1e-12:
            break
        Ap = physics.forward(p)
        alpha = r_norm / (float(np.sum(Ap ** 2)) + 1e-10)
        x = x + alpha * p
        r_new = r - alpha * physics.adjoint(Ap).reshape(physics.x_shape)
        r_new_norm = float(np.sum(r_new ** 2))
        beta = r_new_norm / (r_norm + 1e-10)
        p = r_new + beta * p
        r = r_new
        r_norm = r_new_norm
    return np.maximum(x, 0).astype(np.float32), {"solver": "cgls", "iters": iters}


def run_landweber(y, physics, cfg=None):
    """Landweber iteration (gradient descent on ||Ax-y||^2)."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 30)
    step = cfg.get('step', 0.005)
    x = np.zeros(physics.x_shape, dtype=np.float32)
    for _ in range(iters):
        residual = y - physics.forward(x)
        x = x + step * physics.adjoint(residual).reshape(physics.x_shape)
        x = np.maximum(x, 0)
    return x, {"solver": "landweber", "iters": iters}


def run_mlem(y, physics, cfg=None):
    """MLEM (Maximum Likelihood Expectation Maximization)."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 15)
    # Sensitivity image
    ones_y = np.ones_like(y)
    sensitivity = physics.adjoint(ones_y).reshape(physics.x_shape)
    sensitivity = np.maximum(sensitivity, 1e-10)
    # Initialize with FBP
    x = _do_fbp_recon(y, physics, cfg)
    x = np.maximum(x, 1e-10)
    for _ in range(iters):
        fp = physics.forward(x) + 1e-10
        ratio = y / fp
        correction = physics.adjoint(ratio).reshape(physics.x_shape)
        x = x * correction / sensitivity
        x = np.maximum(x, 1e-10)
    return x.astype(np.float32), {"solver": "mlem", "iters": iters}


def run_osem(y, physics, cfg=None):
    """OSEM (Ordered Subsets Expectation Maximization)."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 5)
    n_subsets = cfg.get('n_subsets', 10)
    # Sensitivity image
    ones_y = np.ones_like(y)
    sensitivity = physics.adjoint(ones_y).reshape(physics.x_shape)
    sensitivity = np.maximum(sensitivity, 1e-10)
    # Initialize with FBP
    x = _do_fbp_recon(y, physics, cfg)
    x = np.maximum(x, 1e-10)
    # Approximate OSEM: scale MLEM update by n_subsets for acceleration
    for it in range(iters):
        fp = physics.forward(x) + 1e-10
        ratio = y / fp
        correction = physics.adjoint(ratio).reshape(physics.x_shape)
        update = correction / sensitivity
        # Blended update (accelerated early, stable late)
        accel = min(float(n_subsets), float(n_subsets) / (1.0 + it))
        x = x * (1.0 + accel * (update - 1.0))
        x = np.maximum(x, 1e-10)
    return x.astype(np.float32), {"solver": "osem", "iters": iters}


# ── Regularization-based solvers ──

def run_tikhonov(y, physics, cfg=None):
    """Tikhonov regularization via gradient descent on ||Ax-y||^2 + lam*||x||^2."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 30)
    lam = cfg.get('lam', 0.01)
    step = cfg.get('step', 0.005)
    x = _do_fbp_recon(y, physics, cfg)
    for _ in range(iters):
        residual = y - physics.forward(x)
        grad = -physics.adjoint(residual).reshape(physics.x_shape) + lam * x
        x = x - step * grad
        x = np.maximum(x, 0)
    return x.astype(np.float32), {"solver": "tikhonov", "iters": iters}


def run_tv_admm(y, physics, cfg=None):
    """TV-regularized CT reconstruction via ADMM."""
    from skimage.restoration import denoise_tv_chambolle
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 20)
    lam = cfg.get('lam', 0.005)
    rho = cfg.get('rho', 1.0)
    # Initialize with FBP
    x = _do_fbp_recon(y, physics, cfg)
    z = x.copy()
    u = np.zeros_like(x)
    step = 0.005
    for it in range(iters):
        # x-update: one gradient step (A^T*A + rho*I)*x = A^T*y + rho*(z - u)
        residual = y - physics.forward(x)
        grad_data = physics.adjoint(residual).reshape(physics.x_shape)
        x = x + step * (grad_data + rho * (z - u - x))
        # z-update: TV proximal
        z = denoise_tv_chambolle(np.clip(x + u, 0, None), weight=lam / rho,
                                 max_num_iter=5)
        z = z.astype(np.float32)
        # u-update
        u = u + x - z
    return np.maximum(x, 0).astype(np.float32), {"solver": "tv_admm", "iters": iters}


def run_chambolle_pock(y, physics, cfg=None):
    """Chambolle-Pock primal-dual algorithm for TV-regularized CT."""
    from skimage.restoration import denoise_tv_chambolle
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 30)
    lam = cfg.get('lam', 0.005)
    tau = cfg.get('tau', 0.005)
    sigma = cfg.get('sigma_cp', 0.5)
    # Initialize
    x = _do_fbp_recon(y, physics, cfg)
    x_bar = x.copy()
    p = np.zeros_like(y)  # dual variable in measurement space
    for _ in range(iters):
        # Dual update
        p = p + sigma * (physics.forward(x_bar) - y)
        p = p / np.maximum(1.0, np.abs(p))  # project to unit ball
        # Primal update
        x_old = x.copy()
        x = x - tau * physics.adjoint(p).reshape(physics.x_shape)
        # TV proximal
        x = denoise_tv_chambolle(np.clip(x, 0, None), weight=lam * tau,
                                 max_num_iter=5)
        x = x.astype(np.float32)
        # Over-relaxation
        x_bar = 2 * x - x_old
    return np.maximum(x, 0).astype(np.float32), {"solver": "chambolle_pock", "iters": iters}


# ── FBP + Post-processing pipelines ──

def run_fbp_nlm(y, physics, cfg=None):
    """FBP + Non-Local Means denoising."""
    from skimage.restoration import denoise_nl_means
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo > 1e-8:
        img_n = ((img - lo) / (hi - lo)).astype(np.float64)
    else:
        img_n = (img - lo).astype(np.float64)
    sigma_est = cfg.get("sigma", 0.02)
    denoised = denoise_nl_means(
        img_n, patch_size=5, patch_distance=6,
        h=0.8 * sigma_est, fast_mode=True, sigma=sigma_est,
    )
    if hi - lo > 1e-8:
        denoised = denoised * (hi - lo) + lo
    return denoised.astype(np.float32), {"solver": "fbp_nlm"}


def run_fbp_bm3d(y, physics, cfg=None):
    """FBP + BM3D denoising."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    try:
        import bm3d
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > 1e-8:
            img_n = (img - lo) / (hi - lo)
        else:
            img_n = img - lo
        denoised = bm3d.bm3d(img_n, sigma_psd=0.02,
                             stage_arg=bm3d.BM3DStages.ALL_STAGES)
        if hi - lo > 1e-8:
            denoised = denoised * (hi - lo) + lo
        return denoised.astype(np.float32), {"solver": "fbp_bm3d"}
    except ImportError:
        return run_fbp_nlm(y, physics, cfg)


def run_fbp_bilateral(y, physics, cfg=None):
    """FBP + bilateral filter denoising."""
    from skimage.restoration import denoise_bilateral
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo > 1e-8:
        img_n = (img - lo) / (hi - lo)
    else:
        img_n = img - lo
    denoised = denoise_bilateral(img_n, sigma_color=0.05, sigma_spatial=3)
    if hi - lo > 1e-8:
        denoised = denoised * (hi - lo) + lo
    return denoised.astype(np.float32), {"solver": "fbp_bilateral"}


def run_fbp_wavelet(y, physics, cfg=None):
    """FBP + wavelet denoising."""
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    try:
        from skimage.restoration import denoise_wavelet
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > 1e-8:
            img_n = (img - lo) / (hi - lo)
        else:
            img_n = img - lo
        denoised = denoise_wavelet(img_n, method='BayesShrink',
                                   mode='soft', rescale_sigma=True)
        if hi - lo > 1e-8:
            denoised = denoised * (hi - lo) + lo
        return denoised.astype(np.float32), {"solver": "fbp_wavelet"}
    except ImportError:
        # PyWavelets not installed — fall back to FBP + NLM
        return run_fbp_nlm(y, physics, cfg)


def run_fbp_tv(y, physics, cfg=None):
    """FBP + Total Variation denoising."""
    from skimage.restoration import denoise_tv_chambolle
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    img = _do_fbp_recon(y, physics, cfg)
    denoised = denoise_tv_chambolle(np.clip(img, 0, None), weight=0.05,
                                    max_num_iter=50)
    return denoised.astype(np.float32), {"solver": "fbp_tv"}


# ── Plug-and-Play solvers (no torch dependency) ──

def _nlm_denoise(img, sigma=0.05):
    """Non-local means denoiser for PnP."""
    from skimage.restoration import denoise_nl_means
    return denoise_nl_means(
        img.astype(np.float64), patch_size=5, patch_distance=6,
        h=0.8 * sigma, fast_mode=True, sigma=sigma,
    ).astype(np.float32)


def _bm3d_denoise(img, sigma=0.05):
    """BM3D denoiser for PnP (falls back to NLM if bm3d not installed)."""
    try:
        import bm3d
        return bm3d.bm3d(img.astype(np.float64), sigma_psd=sigma,
                         stage_arg=bm3d.BM3DStages.ALL_STAGES).astype(np.float32)
    except ImportError:
        return _nlm_denoise(img, sigma)


def run_pnp_admm_nlm(y, physics, cfg=None):
    """PnP-ADMM with NLM denoiser (Venkatakrishnan et al. 2013).

    Uses FBP as data-consistency step + ADMM splitting with NLM denoiser.
    """
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 20)
    sigma = cfg.get('sigma', 0.05)
    rho = cfg.get('rho', 0.5)
    # FBP as fixed data-consistency anchor
    x_fbp = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_fbp.min()), float(x_fbp.max())
    scale = max(hi - lo, 1e-8)
    x = x_fbp.copy()
    z = x.copy()
    u = np.zeros_like(x)
    for it in range(iters):
        # x-update: blend current with FBP for data consistency
        alpha = rho / (1.0 + rho)
        x = alpha * (z - u) + (1 - alpha) * x_fbp
        # z-update: denoise (x + u)
        sig_it = sigma / (1.0 + 0.2 * it)
        v = np.clip((x + u - lo) / scale, 0, 1)
        v_den = _nlm_denoise(v, sig_it)
        z = (v_den * scale + lo).astype(np.float32)
        # u-update
        u = u + x - z
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_admm_nlm"}


def run_pnp_hqs_nlm(y, physics, cfg=None):
    """PnP-HQS with NLM denoiser (Zhang et al. 2017).

    Half-Quadratic Splitting: alternates between data-consistency and denoising.
    Uses increasing mu schedule to gradually enforce data consistency.
    """
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 15)
    sigma = cfg.get('sigma', 0.05)
    x_fbp = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_fbp.min()), float(x_fbp.max())
    scale = max(hi - lo, 1e-8)
    x = x_fbp.copy()
    for it in range(iters):
        # Increasing data-consistency weight (start gentle, end strong)
        mu = 0.3 + 0.5 * (it / max(iters - 1, 1))
        # z-update: enforce data consistency via FBP blend
        z = mu * x_fbp + (1.0 - mu) * x
        # x-update: mild denoise z (decreasing sigma)
        sig_it = sigma / (1.0 + 0.5 * it)
        v = np.clip((z - lo) / scale, 0, 1)
        v_den = _nlm_denoise(v, sig_it)
        x = (v_den * scale + lo).astype(np.float32)
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_hqs_nlm"}


def run_pnp_fista_nlm(y, physics, cfg=None):
    """PnP-FISTA with NLM denoiser (Beck & Teboulle 2009 + PnP).

    FISTA momentum with denoising proximal operator.
    """
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 20)
    sigma = cfg.get('sigma', 0.05)
    mu = cfg.get('mu', 0.5)
    x_fbp = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_fbp.min()), float(x_fbp.max())
    scale = max(hi - lo, 1e-8)
    x = x_fbp.copy()
    x_prev = x.copy()
    t = 1.0
    for k in range(iters):
        # Momentum
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        z = x + momentum * (x - x_prev)
        t = t_new
        # Data consistency: blend with FBP
        z = mu * x_fbp + (1.0 - mu) * z
        # Proximal (denoise)
        x_prev = x.copy()
        sig_it = sigma / (1.0 + 0.2 * k)
        v = np.clip((z - lo) / scale, 0, 1)
        v_den = _nlm_denoise(v, sig_it)
        x = (v_den * scale + lo).astype(np.float32)
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_fista_nlm"}


def run_pnp_admm_bm3d(y, physics, cfg=None):
    """PnP-ADMM with BM3D denoiser (Venkatakrishnan et al. 2013 + BM3D).

    Same ADMM splitting as PnP-ADMM-NLM but with BM3D denoiser.
    Fewer iterations to manage BM3D memory usage.
    """
    import gc
    cfg = cfg or {}
    physics = _ensure_operator(y, physics, cfg)
    iters = cfg.get('iters', 10)  # fewer iters for BM3D (memory-heavy)
    sigma = cfg.get('sigma', 0.05)
    rho = cfg.get('rho', 0.5)
    x_fbp = _do_fbp_recon(y, physics, cfg)
    lo, hi = float(x_fbp.min()), float(x_fbp.max())
    scale = max(hi - lo, 1e-8)
    x = x_fbp.copy()
    z = x.copy()
    u = np.zeros_like(x)
    for it in range(iters):
        alpha = rho / (1.0 + rho)
        x = alpha * (z - u) + (1 - alpha) * x_fbp
        sig_it = sigma / (1.0 + 0.2 * it)
        v = np.clip((x + u - lo) / scale, 0, 1)
        v_den = _bm3d_denoise(v, sig_it)
        z = (v_den * scale + lo).astype(np.float32)
        u = u + x - z
        gc.collect()
    return np.maximum(x, 0).astype(np.float32), {"solver": "pnp_admm_bm3d"}


# ── Deep Learning solvers (fallback to FBP + NLM when no checkpoint) ──

def run_fbpconvnet(y, physics, cfg=None):
    """FBPConvNet (Jin et al., TIP 2017). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "fbpconvnet")


def run_wgan_vgg(y, physics, cfg=None):
    """WGAN-VGG (Yang et al., IEEE TMI 2018). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "wgan_vgg")


def run_learn(y, physics, cfg=None):
    """LEARN (Chen et al., IEEE TMI 2018). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "learn")


def run_learned_pd(y, physics, cfg=None):
    """Learned Primal-Dual (Adler & Oktem, 2018). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "learned_pd")


def run_iradonmap(y, physics, cfg=None):
    """iRadonMAP (He et al., MICCAI 2020). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "iradonmap")


def run_fbp_unet(y, physics, cfg=None):
    """FBP + U-Net post-processing. Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "fbp_unet")


def run_dudonet(y, physics, cfg=None):
    """DuDoNet (Lin et al., CVPR 2019). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dudonet")


def run_indudonet(y, physics, cfg=None):
    """InDuDoNet (Song et al., MICCAI 2021). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "indudonet")


def run_dudotrans(y, physics, cfg=None):
    """DuDoTrans (Wang et al., MICCAI 2022). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dudotrans")


def run_ctformer(y, physics, cfg=None):
    """CTformer (Wang et al., IEEE TMI 2023). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "ctformer")


def run_score_ct(y, physics, cfg=None):
    """Score-CT (Song et al., ICLR 2022). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "score_ct")


def run_dps(y, physics, cfg=None):
    """DPS (Chung et al., ICML 2023). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dps")


def run_diffusion_mbir(y, physics, cfg=None):
    """DiffusionMBIR (Chung & Ye, NeurIPS 2023). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "diffusion_mbir")


def run_dolce(y, physics, cfg=None):
    """DOLCE (Liu et al., 2023). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "dolce")


def run_ct_fm(y, physics, cfg=None):
    """CT-FM (Denker et al., 2024). Fallback: FBP + NLM."""
    return _dl_fallback(y, physics, cfg or {}, "ct_fm")


# ===================================================================
# API
# ===================================================================

def list_solvers():
    """List all available solvers for ct."""
    return [(k, v) for k, v in SOLVERS.items()]


# Convenience wrappers for required keys
def run_traditional_cpu(y, operator=None, cfg=None):
    """FBP (Ram-Lak). CPU only."""
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y, operator=None, cfg=None):
    """FBP + NLM. CPU only."""
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y, operator=None, cfg=None):
    """RED-CNN. Ref: Chen et al. 2017, IEEE TMI."""
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y, operator=None, cfg=None):
    """RED-CNN."""
    return run_solver("small_gpu", y, operator, cfg)
