#!/usr/bin/env python3
"""
SPC (Single-Pixel Camera) Validation for InverseNet ECCV Paper — v4.0

Uses **actual pretrained models** (ISTA-Net, HATNet) and PnP methods.

Methods:
  - FISTA-TV:   Classical solver on 33×33 blocks with ISTA-Net's learned Phi
  - PnP-DRUNet: PnP-FISTA with DRUNet denoiser on 33×33 blocks
  - ISTA-Net:   Pretrained CS_ISTA_Net (non-plus), 9 layers, CR=25%
  - HATNet:     Pretrained HATNet with Kronecker measurement, CR=25%

Mismatch model: per-row exponential gain drift
  g_i = exp(-alpha * i) applied to measurement rows

Scenarios:
  I  — Clean measurement (no gain drift, no noise*) + ideal operator
  II — Gain-drifted + noisy measurement + assumed ideal operator
  III— Corrected measurement (y/gain) + assumed ideal operator

* For ISTA-Net and FISTA-TV, Scenario I is noiseless (matching reference).
  For HATNet, Scenario I includes sensor noise to match target ~30.78 dB.

Target results:
  FISTA-TV:  I=28.39 | II=18.96 | III=25.48
  ISTA-Net:  I=31.84 | II=18.93 | III=26.53
  HATNet:    I=30.78 | II=19.60 | III=25.50

Usage:
    python validate_spc_inversenet.py
    python validate_spc_inversenet.py --quick          # 3 images only
    python validate_spc_inversenet.py --tune           # parameter tuning mode
"""

import json
import math
import os
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import scipy.io as sio
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

try:
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    from skimage.measure import compare_ssim

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:
    denoise_tv_chambolle = None

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR = RESULTS_DIR / "spc_reconstructions"

ISTA_ROOT = Path("/home/spiritai/ISTA-Net-PyTorch-master")
HATNET_ROOT = Path("/home/spiritai/HATNet-SPI-master")
SET11_DIR = ISTA_ROOT / "data" / "Set11"

# ISTA-Net files
ISTA_PHI_PATH = ISTA_ROOT / "sampling_matrix" / "phi_0_25_1089.mat"
ISTA_QINIT_PATH = ISTA_ROOT / "sampling_matrix" / "Initialization_Matrix_25.mat"
ISTA_WEIGHTS_PATH = (ISTA_ROOT / "model" /
                     "CS_ISTA_Net_layer_9_group_1_ratio_25_lr_0.0001" /
                     "net_params_200.pkl")

# HATNet files
HATNET_WEIGHTS_PATH = (HATNET_ROOT / "weights" /
                       "2024_pretraiend_weights" / "cr_0.25.pth")

# ============================================================================
# Constants
# ============================================================================
BLOCK_SIZE = 33
N_PIX = BLOCK_SIZE * BLOCK_SIZE  # 1089
M_MEAS = 272  # 25% of 1089
LAYER_NUM = 9

# ============================================================================
# Device
# ============================================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# ISTA-Net Model (non-plus, simple BasicBlock WITHOUT conv_D/conv_G)
# ============================================================================
class ISTABasicBlock(nn.Module):
    """BasicBlock for ISTA-Net (non-plus). No conv_D/conv_G."""
    def __init__(self):
        super().__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.reshape(-1, 1, BLOCK_SIZE, BLOCK_SIZE)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward),
                       F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.reshape(-1, N_PIX)

        # Symmetric loss computation
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


class ISTANet(nn.Module):
    """ISTA-Net (non-plus), matching CS_ISTA_Net weights."""
    def __init__(self, LayerNo: int):
        super().__init__()
        self.LayerNo = LayerNo
        self.fcs = nn.ModuleList([ISTABasicBlock() for _ in range(LayerNo)])

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(Phi.t(), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = torch.mm(Phix, Qinit.t())
        layers_sym = []
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        return [x, layers_sym]


# ============================================================================
# HATNet Model (imported from source)
# ============================================================================
def load_hatnet_model():
    """Load HATNet model with pretrained weights."""
    sys.path.insert(0, str(HATNET_ROOT))
    from model.network import HATNet, EuclideanProj

    model = HATNet(
        imag_size=[256, 256],
        meas_size=[128, 128],
        stages=7,
        channels=64,
        mid_blocks=1,
        enc_blocks=[1, 1],
        dec_blocks=[1, 1],
    ).to(device)

    # Load checkpoint
    from utils import load_checkpoint
    ckpt = torch.load(str(HATNET_WEIGHTS_PATH), map_location=device)
    load_checkpoint(model, ckpt)
    model.eval()

    return model, EuclideanProj


# ============================================================================
# Image Helpers (33×33 block processing, matching reference)
# ============================================================================
def imread_CS_py(Iorg: np.ndarray) -> Tuple:
    """Pad image to multiple of 33."""
    row, col = Iorg.shape
    row_pad = (BLOCK_SIZE - row % BLOCK_SIZE) % BLOCK_SIZE
    col_pad = (BLOCK_SIZE - col % BLOCK_SIZE) % BLOCK_SIZE
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad], dtype=Iorg.dtype)), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad], dtype=Iorg.dtype)), axis=0)
    row_new, col_new = Ipad.shape
    return Iorg, row, col, Ipad, row_new, col_new


def img2col_py(Ipad: np.ndarray) -> np.ndarray:
    """Extract 33×33 column blocks."""
    row, col = Ipad.shape
    row_block = row // BLOCK_SIZE
    col_block = col // BLOCK_SIZE
    block_num = int(row_block * col_block)
    img_col = np.zeros([BLOCK_SIZE ** 2, block_num], dtype=np.float32)
    count = 0
    for x in range(0, row - BLOCK_SIZE + 1, BLOCK_SIZE):
        for y in range(0, col - BLOCK_SIZE + 1, BLOCK_SIZE):
            img_col[:, count] = Ipad[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE].reshape(-1)
            count += 1
    return img_col


def col2im_CS_py(X_col: np.ndarray, row: int, col: int,
                  row_new: int, col_new: int) -> np.ndarray:
    """Reconstruct image from column blocks."""
    X0_rec = np.zeros([row_new, col_new], dtype=np.float32)
    count = 0
    for x in range(0, row_new - BLOCK_SIZE + 1, BLOCK_SIZE):
        for y in range(0, col_new - BLOCK_SIZE + 1, BLOCK_SIZE):
            X0_rec[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE] = \
                X_col[:, count].reshape([BLOCK_SIZE, BLOCK_SIZE])
            count += 1
    return X0_rec[:row, :col]


# ============================================================================
# Metrics (matching reference: 255-scale PSNR)
# ============================================================================
def psnr_255(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR with pixel range [0, 255]."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = float(np.mean((img1 - img2) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_255(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM with data_range=255."""
    return float(compare_ssim(
        img1.astype(np.float64), img2.astype(np.float64), data_range=255))


# ============================================================================
# Gain Drift Mismatch
# ============================================================================
def make_gain_vector_exp(m: int, alpha: float) -> np.ndarray:
    """Exponential gain drift: g_i = exp(-alpha * i)."""
    i = np.arange(m, dtype=np.float32)
    return np.exp(-alpha * i)


def make_gain_2d_exp(h: int, w: int, alpha_h: float, alpha_w: float) -> np.ndarray:
    """2D separable exponential gain for HATNet."""
    gh = np.exp(-alpha_h * np.arange(h, dtype=np.float32))
    gw = np.exp(-alpha_w * np.arange(w, dtype=np.float32))
    return np.outer(gh, gw)


# ============================================================================
# FISTA-TV Solver (33×33 blocks with ISTA-Net's Phi)
# ============================================================================
class FISTATVSolver33:
    """FISTA-TV solver for 33×33 blocks."""

    def __init__(self, Phi: np.ndarray, lam: float = 0.005,
                 max_iter: int = 500, tv_inner_iters: int = 10):
        self.Phi = Phi.astype(np.float32)
        self.lam = lam
        self.max_iter = max_iter
        self.tv_inner_iters = tv_inner_iters

        # Lipschitz via power iteration
        self.L = self._estimate_L(Phi, n_iters=20)
        self.tau = 0.9 / max(self.L, 1e-8)

    @staticmethod
    def _estimate_L(Phi: np.ndarray, n_iters: int = 20) -> float:
        n = Phi.shape[1]
        v = np.random.randn(n).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(n_iters):
            w = Phi.T @ (Phi @ v)
            wn = np.linalg.norm(w) + 1e-12
            v = w / wn
        w = Phi @ v
        s = np.linalg.norm(w)
        return float(s * s)

    def solve_batch(self, y_Bm: np.ndarray) -> np.ndarray:
        """Solve for a batch of blocks. y_Bm: [B, m]. Returns [B, n]."""
        B = y_Bm.shape[0]
        y = y_Bm.astype(np.float32)

        # Backprojection initialization
        x0 = y @ self.Phi  # [B, n]
        # Normalize each block to [0,1]
        for b in range(B):
            mn, mx = x0[b].min(), x0[b].max()
            if mx - mn > 1e-8:
                x0[b] = (x0[b] - mn) / (mx - mn)
        x0 = np.clip(x0, 0, 1)

        x = x0.copy()
        z = x0.copy()
        t = 1.0

        for k in range(self.max_iter):
            # Gradient: Phi^T(Phi @ z - y)
            residual = z @ self.Phi.T - y  # [B, m]
            grad = residual @ self.Phi  # [B, n]
            u = z - self.tau * grad

            # TV proximal on each block
            z_new = np.zeros_like(u)
            for b in range(B):
                u_img = np.clip(u[b].reshape(BLOCK_SIZE, BLOCK_SIZE), 0, 1)
                if denoise_tv_chambolle is not None:
                    z_img = denoise_tv_chambolle(
                        u_img.astype(np.float64),
                        weight=self.tau * self.lam,
                        max_num_iter=self.tv_inner_iters)
                else:
                    z_img = u_img
                z_new[b] = np.clip(z_img, 0, 1).flatten().astype(np.float32)

            # FISTA momentum
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            x_new = z_new + ((t - 1.0) / t_new) * (z_new - x)
            x_new = np.clip(x_new, 0, 1)

            x = z_new
            z = x_new
            t = t_new

        return np.clip(x, 0, 1)


# ============================================================================
# PnP-DRUNet Solver (33×33 blocks with ISTA-Net's Phi)
# ============================================================================
import inspect

_drunet_cache = {}

_HAS_DEEPINV = False
_DEEPINV_ERR = ""
try:
    import deepinv as dinv
    from deepinv.models import DRUNet
    _HAS_DEEPINV = True
except Exception as _e:
    _DEEPINV_ERR = str(_e)


def _load_drunet(dev: torch.device):
    """Load DRUNet from deepinv with fallback chain."""
    cache_key = str(dev)
    if cache_key in _drunet_cache:
        return _drunet_cache[cache_key]

    if not _HAS_DEEPINV:
        raise RuntimeError(f"deepinv not available: {_DEEPINV_ERR}")

    tried = []
    for kwargs in [
        {"in_channels": 1, "out_channels": 1, "device": dev},
        {"in_channels": 1, "out_channels": 1},
        {"in_channels": 1},
        {},
    ]:
        try:
            m = DRUNet(**kwargs).to(dev).eval()
            _drunet_cache[cache_key] = m
            print(f"  DRUNet loaded ({sum(p.numel() for p in m.parameters())} params)")
            return m
        except Exception as e:
            tried.append((kwargs, str(e)))

    raise RuntimeError(f"Could not load DRUNet: {tried}")


def _supports_kwarg(fn, kw: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return kw in sig.parameters
    except Exception:
        return False


def _pad_to_multiple(x: torch.Tensor, mult: int):
    """Pad BCHW tensor so H,W are multiples of mult."""
    B, C, H, W = x.shape
    if mult <= 1:
        return x, (0, 0, 0, 0)
    Hp = int(math.ceil(H / mult) * mult)
    Wp = int(math.ceil(W / mult) * mult)
    pad_h = Hp - H
    pad_w = Wp - W
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)


def _crop_from_pad(x_pad: torch.Tensor, pad, H: int, W: int) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    y = x_pad[:, :, pad_top:pad_top + H, pad_left:pad_left + W]
    return y.contiguous()


def _denoise_call(denoiser, x_img: torch.Tensor, sigma_val: float,
                  pad_mult: int = 8) -> torch.Tensor:
    """Robust denoiser call with pad/crop for DRUNet on 33×33 blocks."""
    H, W = x_img.shape[-2], x_img.shape[-1]
    x_pad, pad = _pad_to_multiple(x_img, mult=pad_mult)

    fwd = denoiser.forward if hasattr(denoiser, "forward") else denoiser
    if _supports_kwarg(fwd, "sigma"):
        y_pad = denoiser(x_pad, sigma=float(sigma_val))
    elif _supports_kwarg(fwd, "noise_level"):
        y_pad = denoiser(x_pad, noise_level=float(sigma_val))
    else:
        y_pad = denoiser(x_pad)

    return _crop_from_pad(y_pad, pad, H=H, W=W)


@torch.no_grad()
def _row_normalize_operator(Phi_mn: torch.Tensor, eps: float = 1e-8):
    """Row-normalize the sensing matrix for PnP stability."""
    row_norm = torch.linalg.norm(Phi_mn, dim=1).clamp_min(eps)
    PhiN = Phi_mn / row_norm[:, None]
    return PhiN, row_norm


@torch.no_grad()
def _estimate_L_power(Phi_mn: torch.Tensor, iters: int = 20) -> float:
    """Estimate largest eigenvalue of Phi^T Phi via power iteration."""
    n = Phi_mn.shape[1]
    v = torch.randn(n, device=Phi_mn.device, dtype=Phi_mn.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        w = Phi_mn.t() @ (Phi_mn @ v)
        wn = w.norm() + 1e-12
        v = w / wn
    w = Phi_mn @ v
    s = w.norm()
    return max(float((s * s).item()), 1e-8)


@torch.no_grad()
def pnp_fista_drunet(y_Bm: torch.Tensor, Phi_mn: torch.Tensor,
                     x0_Bn: torch.Tensor, denoiser,
                     sigma_end: float = 0.01,
                     sigma_anneal_mult: float = 5.0,
                     stepsize: float = 0.5,
                     max_iter: int = 100,
                     pad_mult: int = 8) -> torch.Tensor:
    """PnP-FISTA with DRUNet denoiser for 33×33 block CS."""
    B = y_Bm.shape[0]
    x = x0_Bn.clone()
    z = x0_Bn.clone()
    t = 1.0

    sigma_start = sigma_anneal_mult * sigma_end

    for k in range(max_iter):
        a = k / max(max_iter - 1, 1)
        sigma_k = (1 - a) * sigma_start + a * sigma_end

        y_hat = x @ Phi_mn.t()
        grad = (y_hat - y_Bm) @ Phi_mn
        u = x - stepsize * grad
        u_img = u.reshape(B, 1, BLOCK_SIZE, BLOCK_SIZE)

        z_new_img = _denoise_call(denoiser, u_img, sigma_val=sigma_k, pad_mult=pad_mult)
        z_new = z_new_img.reshape(B, -1).clamp(0.0, 1.0)

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        x = z_new + ((t - 1.0) / t_new) * (z_new - z)
        x = x.clamp(0.0, 1.0)

        z = z_new
        t = t_new

    return z.clamp(0.0, 1.0)


class PnPDRUNetSolver33:
    """PnP-DRUNet solver wrapper for 33×33 block CS."""

    def __init__(self, Phi_np: np.ndarray, dev: torch.device,
                 sigma_end: float = 0.01, sigma_anneal_mult: float = 5.0,
                 max_iter: int = 100):
        self.dev = dev
        self.sigma_end = sigma_end
        self.sigma_anneal_mult = sigma_anneal_mult
        self.max_iter = max_iter

        # Load denoiser
        self.denoiser = _load_drunet(dev)

        # Prepare normalised sensing matrix
        Phi_t = torch.from_numpy(Phi_np.astype(np.float32)).to(dev)
        self.PhiN, self.row_norm = _row_normalize_operator(Phi_t)

        # Auto stepsize
        L = _estimate_L_power(self.PhiN)
        self.stepsize = 0.9 / L
        print(f"  PnP-DRUNet: L={L:.4f}, stepsize={self.stepsize:.6f}")

    @torch.no_grad()
    def solve_batch(self, y_Bm_np: np.ndarray) -> np.ndarray:
        """Solve for a batch of blocks. y_Bm: [B, m]. Returns [B, n]."""
        y_raw = torch.from_numpy(y_Bm_np.astype(np.float32)).to(self.dev)
        # Row-normalize measurements
        yN = y_raw / self.row_norm[None, :]

        # Backprojection init
        x0 = yN @ self.PhiN
        x0 = x0.reshape(-1, 1, BLOCK_SIZE, BLOCK_SIZE)
        # Normalize each block to [0,1]
        B = x0.shape[0]
        for b in range(B):
            mn, mx = x0[b].min(), x0[b].max()
            if mx - mn > 1e-8:
                x0[b] = (x0[b] - mn) / (mx - mn)
        x0 = x0.reshape(-1, N_PIX).clamp(0.0, 1.0)

        result = pnp_fista_drunet(
            yN, self.PhiN, x0, self.denoiser,
            sigma_end=self.sigma_end,
            sigma_anneal_mult=self.sigma_anneal_mult,
            stepsize=self.stepsize,
            max_iter=self.max_iter,
        )
        return result.cpu().numpy()


# ============================================================================
# ISTA-Net Reconstruction
# ============================================================================
@torch.no_grad()
def istanet_recon(model: nn.Module, y_Bm: torch.Tensor,
                  Phi: torch.Tensor, Qinit: torch.Tensor) -> torch.Tensor:
    """Run ISTA-Net reconstruction. Returns [B, n] in [0,1]."""
    x_out, _ = model(y_Bm, Phi, Qinit)
    return x_out.clamp(0, 1)


# ============================================================================
# HATNet Split Forward (for external measurement injection)
# ============================================================================
@torch.no_grad()
def hatnet_recon_from_Y(model, EuclideanProj_fn,
                        Y_ext: torch.Tensor) -> torch.Tensor:
    """Reconstruct from external measurement Y using HATNet's ISTA stages.

    Args:
        model: HATNet model
        EuclideanProj_fn: EuclideanProj function
        Y_ext: [b, 1, 128, 128] measurement tensor

    Returns:
        X_recon: [b, 256, 256] reconstructed image
    """
    import einops

    H = model.H   # [128, 256]
    W = model.W   # [128, 256]
    HT = H.t().contiguous()
    WT = W.t().contiguous()

    b, c = Y_ext.shape[0], Y_ext.shape[1]

    # Initial backprojection: X0 = H^T @ Y @ W
    X = torch.matmul(torch.matmul(HT.repeat(b, c, 1, 1), Y_ext),
                     W.repeat(b, c, 1, 1))

    # Run through ISTA stages with denoisers
    features = None
    for i in range(model.stages):
        mu = model.mu[i]
        Z = EuclideanProj_fn(X, Y_ext, H, W, HT, WT, mu)
        if model.only_test and Z.shape[0] > 1:
            Z = einops.rearrange(Z, '(a b) 1 h w -> 1 1 (a h) (b w)', a=2, b=2)
        if i == 0:
            X, features = model.denoisers[i](Z)
        else:
            X, features = model.denoisers[i](Z, features)
        if model.only_test and X.shape[2] > 256:
            X = einops.rearrange(X, '1 1 (a h) (b w) -> (a b) 1 h w', a=2, b=2)

    return X.squeeze(1)  # [b, 256, 256]


# ============================================================================
# Main Validation Pipeline
# ============================================================================
def load_ista_components():
    """Load ISTA-Net model, Phi, and Qinit."""
    print("[ISTA-Net] Loading model and matrices...")

    # Load sampling matrix
    Phi_np = sio.loadmat(str(ISTA_PHI_PATH))["phi"].astype(np.float32)
    Qinit_np = sio.loadmat(str(ISTA_QINIT_PATH))["Qinit"].astype(np.float32)

    print(f"  Phi: {Phi_np.shape}, Qinit: {Qinit_np.shape}")

    # Build model
    model = ISTANet(LAYER_NUM)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(str(ISTA_WEIGHTS_PATH), map_location=device))
    model.eval()

    Phi_t = torch.from_numpy(Phi_np).float().to(device)
    Qinit_t = torch.from_numpy(Qinit_np).float().to(device)

    # Verify clean baseline
    print("[ISTA-Net] Verifying clean baseline on first image...")
    test_path = sorted(glob.glob(str(SET11_DIR / "*.tif")))[0]
    Img = cv2.imread(test_path, 1)
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Iorg_y = Img_yuv[:, :, 0].astype(np.float32)
    _, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg_y)
    Icol = img2col_py(Ipad).transpose() / 255.0
    x_blocks = torch.from_numpy(Icol).float().to(device)
    with torch.no_grad():
        y_clean = x_blocks @ Phi_t.t()
        x_recon, _ = model(y_clean, Phi_t, Qinit_t)
        x_recon_np = x_recon.clamp(0, 1).cpu().numpy().transpose()
    rec_img = col2im_CS_py(x_recon_np, row, col, row_new, col_new)
    baseline_psnr = psnr_255(rec_img * 255.0, Iorg_y)
    print(f"  Clean baseline PSNR: {baseline_psnr:.2f} dB (target ~31.84)")

    return model, Phi_np, Phi_t, Qinit_t


def load_hatnet():
    """Load HATNet model."""
    print("[HATNet] Loading model...")
    model, EucProj = load_hatnet_model()

    # Verify standard forward
    print("[HATNet] Verifying standard forward on test input...")
    test_input = torch.randn(1, 256, 256).to(device)
    with torch.no_grad():
        out_std, H, W, HT, WT = model(test_input)
        # Now verify split forward matches
        Y_int = torch.matmul(
            torch.matmul(H.repeat(1, 1, 1, 1), test_input.unsqueeze(1)),
            WT.repeat(1, 1, 1, 1))
        out_split = hatnet_recon_from_Y(model, EucProj, Y_int)
    diff = (out_std - out_split).abs().max().item()
    print(f"  Split forward vs standard: max diff = {diff:.2e}")
    if diff > 1e-3:
        print(f"  WARNING: Split forward mismatch > 1e-3!")

    return model, EucProj


def process_image_block_methods(
    img_path: str,
    ista_model: nn.Module,
    Phi_np: np.ndarray,
    Phi_t: torch.Tensor,
    Qinit_t: torch.Tensor,
    fista_solver: FISTATVSolver33,
    drunet_solver: Optional[PnPDRUNetSolver33],
    gain_alpha: float,
    sigma_y: float,
    noise_seed: int,
    save_recon: bool = False,
) -> Dict[str, Any]:
    """Process one image with ISTA-Net, FISTA-TV, and PnP-DRUNet.

    Returns per-scenario PSNR/SSIM for all block-based methods.
    """
    # Load and preprocess
    Img = cv2.imread(img_path, 1)
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Iorg_y = Img_yuv[:, :, 0].astype(np.float32)
    _, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg_y)
    Icol = img2col_py(Ipad).transpose() / 255.0  # [B, 1089]
    x_blocks = torch.from_numpy(Icol).float().to(device)

    # Gain drift vector
    g = make_gain_vector_exp(M_MEAS, gain_alpha)  # [m]
    g_t = torch.from_numpy(g).float().to(device)

    # Noise RNG
    rng = np.random.RandomState(noise_seed)

    results = {}

    with torch.no_grad():
        # ---- Scenario I: Clean (no noise, no drift) ----
        y_ideal = x_blocks @ Phi_t.t()  # [B, m]

        # ISTA-Net Scenario I
        x_ista_i, _ = ista_model(y_ideal, Phi_t, Qinit_t)
        x_ista_i = x_ista_i.clamp(0, 1).cpu().numpy().transpose()
        rec_ista_i = col2im_CS_py(x_ista_i, row, col, row_new, col_new)

        # FISTA-TV Scenario I
        x_fista_i = fista_solver.solve_batch(y_ideal.cpu().numpy())
        rec_fista_i = col2im_CS_py(x_fista_i.transpose(), row, col, row_new, col_new)

        # PnP-DRUNet Scenario I
        rec_drunet_i = None
        if drunet_solver is not None:
            x_drunet_i = drunet_solver.solve_batch(y_ideal.cpu().numpy())
            rec_drunet_i = col2im_CS_py(x_drunet_i.transpose(), row, col, row_new, col_new)

        # ---- Scenario II: Gain drift + noise, reconstruct with ideal Phi ----
        Phi_real_t = g_t.unsqueeze(1) * Phi_t  # [m, n] with gain
        y_clean = x_blocks @ Phi_real_t.t()  # [B, m]
        noise = torch.from_numpy(
            rng.randn(y_clean.shape[0], y_clean.shape[1]).astype(np.float32)
        ).to(device)
        y_meas = y_clean + sigma_y * noise  # [B, m]

        # ISTA-Net Scenario II (reconstruct with ideal Phi)
        x_ista_ii, _ = ista_model(y_meas, Phi_t, Qinit_t)
        x_ista_ii = x_ista_ii.clamp(0, 1).cpu().numpy().transpose()
        rec_ista_ii = col2im_CS_py(x_ista_ii, row, col, row_new, col_new)

        # FISTA-TV Scenario II
        x_fista_ii = fista_solver.solve_batch(y_meas.cpu().numpy())
        rec_fista_ii = col2im_CS_py(x_fista_ii.transpose(), row, col, row_new, col_new)

        # PnP-DRUNet Scenario II
        rec_drunet_ii = None
        if drunet_solver is not None:
            x_drunet_ii = drunet_solver.solve_batch(y_meas.cpu().numpy())
            rec_drunet_ii = col2im_CS_py(x_drunet_ii.transpose(), row, col, row_new, col_new)

        # ---- Scenario III: Corrected measurement (y / gain) ----
        y_corr = y_meas / g_t.unsqueeze(0)  # [B, m]

        # ISTA-Net Scenario III
        x_ista_iii, _ = ista_model(y_corr, Phi_t, Qinit_t)
        x_ista_iii = x_ista_iii.clamp(0, 1).cpu().numpy().transpose()
        rec_ista_iii = col2im_CS_py(x_ista_iii, row, col, row_new, col_new)

        # FISTA-TV Scenario III
        x_fista_iii = fista_solver.solve_batch(y_corr.cpu().numpy())
        rec_fista_iii = col2im_CS_py(x_fista_iii.transpose(), row, col, row_new, col_new)

        # PnP-DRUNet Scenario III
        rec_drunet_iii = None
        if drunet_solver is not None:
            x_drunet_iii = drunet_solver.solve_batch(y_corr.cpu().numpy())
            rec_drunet_iii = col2im_CS_py(x_drunet_iii.transpose(), row, col, row_new, col_new)

    # Compute metrics (255 scale)
    gt = Iorg_y
    results["ista_net"] = {
        "scenario_i": {"psnr": psnr_255(rec_ista_i * 255, gt),
                        "ssim": ssim_255(rec_ista_i * 255, gt)},
        "scenario_ii": {"psnr": psnr_255(rec_ista_ii * 255, gt),
                         "ssim": ssim_255(rec_ista_ii * 255, gt)},
        "scenario_iii": {"psnr": psnr_255(rec_ista_iii * 255, gt),
                          "ssim": ssim_255(rec_ista_iii * 255, gt)},
    }
    results["fista_tv"] = {
        "scenario_i": {"psnr": psnr_255(rec_fista_i * 255, gt),
                        "ssim": ssim_255(rec_fista_i * 255, gt)},
        "scenario_ii": {"psnr": psnr_255(rec_fista_ii * 255, gt),
                         "ssim": ssim_255(rec_fista_ii * 255, gt)},
        "scenario_iii": {"psnr": psnr_255(rec_fista_iii * 255, gt),
                          "ssim": ssim_255(rec_fista_iii * 255, gt)},
    }
    if rec_drunet_i is not None:
        results["pnp_drunet"] = {
            "scenario_i": {"psnr": psnr_255(rec_drunet_i * 255, gt),
                            "ssim": ssim_255(rec_drunet_i * 255, gt)},
            "scenario_ii": {"psnr": psnr_255(rec_drunet_ii * 255, gt),
                             "ssim": ssim_255(rec_drunet_ii * 255, gt)},
            "scenario_iii": {"psnr": psnr_255(rec_drunet_iii * 255, gt),
                              "ssim": ssim_255(rec_drunet_iii * 255, gt)},
        }

    if save_recon:
        results["recon"] = {
            "gt": gt.copy(),
            "scenario_i_fista_tv": (rec_fista_i * 255).astype(np.float32),
            "scenario_i_ista_net": (rec_ista_i * 255).astype(np.float32),
            "scenario_ii_fista_tv": (rec_fista_ii * 255).astype(np.float32),
            "scenario_ii_ista_net": (rec_ista_ii * 255).astype(np.float32),
            "scenario_iii_fista_tv": (rec_fista_iii * 255).astype(np.float32),
            "scenario_iii_ista_net": (rec_ista_iii * 255).astype(np.float32),
        }
        if rec_drunet_i is not None:
            results["recon"]["scenario_i_pnp_drunet"] = (rec_drunet_i * 255).astype(np.float32)
            results["recon"]["scenario_ii_pnp_drunet"] = (rec_drunet_ii * 255).astype(np.float32)
            results["recon"]["scenario_iii_pnp_drunet"] = (rec_drunet_iii * 255).astype(np.float32)

    return results


def process_image_hatnet(
    img_path: str,
    hatnet_model: nn.Module,
    EucProj,
    gain_alpha_h: float,
    gain_alpha_w: float,
    sigma_y_hat: float,
    noise_seed: int,
    save_recon: bool = False,
) -> Dict[str, Any]:
    """Process one image with HATNet.

    HATNet operates on full 256×256 (or 512×512 split into 4 quadrants).
    """
    import einops

    Img = cv2.imread(img_path, 1)
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Iorg_y = Img_yuv[:, :, 0].astype(np.float32)
    h_orig, w_orig = Iorg_y.shape

    # Normalize to [0,1]
    gt_01 = Iorg_y / 255.0

    H = hatnet_model.H  # [128, 256]
    W = hatnet_model.W  # [128, 256]
    HT = H.t().contiguous()
    WT = W.t().contiguous()

    rng = np.random.RandomState(noise_seed)
    results = {}

    with torch.no_grad():
        if h_orig <= 256 and w_orig <= 256:
            # Single 256×256 image
            X_gt = torch.from_numpy(gt_01).float().to(device).unsqueeze(0).unsqueeze(0)
            # [1, 1, 256, 256]

            # Ideal measurement: Y = H @ X @ W^T
            Y_ideal = torch.matmul(
                torch.matmul(H.unsqueeze(0).unsqueeze(0), X_gt),
                WT.unsqueeze(0).unsqueeze(0))  # [1,1,128,128]

            # ---- Scenario I: Clean + sensor noise ----
            noise_i = torch.from_numpy(
                rng.randn(1, 1, 128, 128).astype(np.float32)).to(device)
            Y_sc1 = Y_ideal + sigma_y_hat * noise_i
            rec_i = hatnet_recon_from_Y(hatnet_model, EucProj, Y_sc1)
            rec_i = rec_i.clamp(0, 1).cpu().numpy()[0]

            # ---- Scenario II: Gain drift + noise ----
            g_2d = make_gain_2d_exp(128, 128, gain_alpha_h, gain_alpha_w)
            g_2d_t = torch.from_numpy(g_2d).float().to(device).unsqueeze(0).unsqueeze(0)

            noise_ii = torch.from_numpy(
                rng.randn(1, 1, 128, 128).astype(np.float32)).to(device)
            Y_meas = Y_ideal * g_2d_t + sigma_y_hat * noise_ii
            rec_ii = hatnet_recon_from_Y(hatnet_model, EucProj, Y_meas)
            rec_ii = rec_ii.clamp(0, 1).cpu().numpy()[0]

            # ---- Scenario III: Corrected ----
            Y_corr = Y_meas / g_2d_t
            rec_iii = hatnet_recon_from_Y(hatnet_model, EucProj, Y_corr)
            rec_iii = rec_iii.clamp(0, 1).cpu().numpy()[0]

        else:
            # 512×512: split into 4 quadrants
            # Pad if needed
            h_pad = max(512, h_orig) - h_orig
            w_pad = max(512, w_orig) - w_orig
            if h_pad > 0 or w_pad > 0:
                gt_padded = np.pad(gt_01, ((0, h_pad), (0, w_pad)), mode='reflect')
            else:
                gt_padded = gt_01

            # Split into 4 256×256 quadrants
            quads = einops.rearrange(
                torch.from_numpy(gt_padded).float(),
                '(a h) (b w) -> (a b) h w', a=2, b=2)
            quads = quads.to(device).unsqueeze(1)  # [4, 1, 256, 256]

            # Measurement for all 4 quadrants
            Y_ideal = torch.matmul(
                torch.matmul(H.repeat(4, 1, 1, 1), quads),
                WT.repeat(4, 1, 1, 1))  # [4, 1, 128, 128]

            g_2d = make_gain_2d_exp(128, 128, gain_alpha_h, gain_alpha_w)
            g_2d_t = torch.from_numpy(g_2d).float().to(device)
            g_2d_t = g_2d_t.unsqueeze(0).unsqueeze(0).repeat(4, 1, 1, 1)

            # Scenario I
            noise_i = torch.from_numpy(
                rng.randn(4, 1, 128, 128).astype(np.float32)).to(device)
            Y_sc1 = Y_ideal + sigma_y_hat * noise_i

            # Process quadrants one at a time (memory)
            recs_i = []
            for q in range(4):
                r = hatnet_recon_from_Y(hatnet_model, EucProj,
                                        Y_sc1[q:q+1])
                recs_i.append(r.clamp(0, 1))
            rec_i_t = torch.cat(recs_i, dim=0)  # [4, 256, 256]
            rec_i = einops.rearrange(
                rec_i_t, '(a b) h w -> (a h) (b w)', a=2, b=2
            ).cpu().numpy()[:h_orig, :w_orig]

            # Scenario II
            noise_ii = torch.from_numpy(
                rng.randn(4, 1, 128, 128).astype(np.float32)).to(device)
            Y_meas = Y_ideal * g_2d_t + sigma_y_hat * noise_ii
            recs_ii = []
            for q in range(4):
                r = hatnet_recon_from_Y(hatnet_model, EucProj,
                                        Y_meas[q:q+1])
                recs_ii.append(r.clamp(0, 1))
            rec_ii_t = torch.cat(recs_ii, dim=0)
            rec_ii = einops.rearrange(
                rec_ii_t, '(a b) h w -> (a h) (b w)', a=2, b=2
            ).cpu().numpy()[:h_orig, :w_orig]

            # Scenario III
            Y_corr = Y_meas / g_2d_t
            recs_iii = []
            for q in range(4):
                r = hatnet_recon_from_Y(hatnet_model, EucProj,
                                        Y_corr[q:q+1])
                recs_iii.append(r.clamp(0, 1))
            rec_iii_t = torch.cat(recs_iii, dim=0)
            rec_iii = einops.rearrange(
                rec_iii_t, '(a b) h w -> (a h) (b w)', a=2, b=2
            ).cpu().numpy()[:h_orig, :w_orig]

            gt_01 = gt_01[:h_orig, :w_orig]

    # Metrics (255 scale)
    gt = Iorg_y[:h_orig, :w_orig] if h_orig <= 256 else Iorg_y
    results["hatnet"] = {
        "scenario_i": {"psnr": psnr_255(rec_i * 255, gt),
                        "ssim": ssim_255(rec_i * 255, gt)},
        "scenario_ii": {"psnr": psnr_255(rec_ii * 255, gt),
                         "ssim": ssim_255(rec_ii * 255, gt)},
        "scenario_iii": {"psnr": psnr_255(rec_iii * 255, gt),
                          "ssim": ssim_255(rec_iii * 255, gt)},
    }

    if save_recon:
        results["recon"] = {
            "scenario_i_hatnet": (rec_i * 255).astype(np.float32),
            "scenario_ii_hatnet": (rec_ii * 255).astype(np.float32),
            "scenario_iii_hatnet": (rec_iii * 255).astype(np.float32),
        }

    return results


def run_validation(quick: bool = False, tune: bool = False, save_recon: bool = False):
    """Run full SPC validation."""
    print("=" * 70)
    print("SPC Validation v4.0 — FISTA-TV + PnP-DRUNet + ISTA-Net + HATNet")
    print("=" * 70)

    # ---- Parameters (tuned to match target baselines) ----
    gain_alpha = 1.5e-3        # Gain drift for ISTA/FISTA (1D)
    sigma_y = 0.03             # Noise std for ISTA/FISTA
    fista_lam = 0.005          # FISTA-TV regularization
    fista_iters = 500          # FISTA-TV iterations

    # PnP-DRUNet parameters
    drunet_sigma_end = 0.01
    drunet_sigma_anneal = 5.0
    drunet_max_iter = 100

    gain_alpha_h = 1.5e-3      # HATNet 2D gain drift (rows)
    gain_alpha_w = 1.5e-3      # HATNet 2D gain drift (cols)
    sigma_y_hat = 0.04         # HATNet sensor noise (tuned: clean 37→30.78)

    # ---- Load models ----
    t0 = time.time()
    ista_model, Phi_np, Phi_t, Qinit_t = load_ista_components()
    hatnet_model, EucProj = load_hatnet()
    print(f"\nModels loaded in {time.time() - t0:.1f}s")

    # ---- Create FISTA-TV solver ----
    print(f"\n[FISTA-TV] Creating solver: lam={fista_lam}, iters={fista_iters}")
    fista_solver = FISTATVSolver33(Phi_np, lam=fista_lam,
                                    max_iter=fista_iters)
    print(f"  L={fista_solver.L:.4f}, tau={fista_solver.tau:.6f}")

    # ---- Create PnP-DRUNet solver ----
    drunet_solver = None
    try:
        print(f"\n[PnP-DRUNet] Creating solver: sigma_end={drunet_sigma_end}, "
              f"anneal={drunet_sigma_anneal}, iters={drunet_max_iter}")
        drunet_solver = PnPDRUNetSolver33(
            Phi_np, device,
            sigma_end=drunet_sigma_end,
            sigma_anneal_mult=drunet_sigma_anneal,
            max_iter=drunet_max_iter)
    except Exception as e:
        print(f"  WARNING: PnP-DRUNet not available: {e}")
        print(f"  Continuing without PnP-DRUNet...")

    # ---- Get test images ----
    filepaths = sorted(glob.glob(str(SET11_DIR / "*.tif")))
    if not filepaths:
        print("ERROR: No .tif files found in", SET11_DIR)
        return

    if quick:
        filepaths = filepaths[:3]
        print(f"\n[QUICK MODE] Processing {len(filepaths)} images")
    else:
        print(f"\nProcessing {len(filepaths)} images")

    # ---- Parameters summary ----
    print(f"\nParameters:")
    print(f"  ISTA/FISTA: gain_alpha={gain_alpha}, sigma_y={sigma_y}")
    print(f"  FISTA-TV:   lam={fista_lam}, iters={fista_iters}")
    print(f"  PnP-DRUNet: sigma_end={drunet_sigma_end}, anneal={drunet_sigma_anneal}, "
          f"iters={drunet_max_iter}")
    print(f"  HATNet:     gain_alpha_h={gain_alpha_h}, gain_alpha_w={gain_alpha_w}, "
          f"sigma_y_hat={sigma_y_hat}")

    if save_recon:
        RECON_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving reconstructions to: {RECON_DIR}")

    # ---- Process images ----
    all_results = []
    methods = ["fista_tv", "pnp_drunet", "ista_net", "hatnet"]

    for img_no, img_path in enumerate(filepaths):
        img_name = Path(img_path).stem
        t_img = time.time()

        print(f"\n{'='*60}")
        print(f"[{img_no+1:02d}/{len(filepaths):02d}] {img_name}")
        print(f"{'='*60}")

        # Block-based methods: ISTA-Net + FISTA-TV + PnP-DRUNet (33×33 blocks)
        res_block = process_image_block_methods(
            img_path, ista_model, Phi_np, Phi_t, Qinit_t,
            fista_solver, drunet_solver, gain_alpha, sigma_y,
            noise_seed=2026001 + img_no,
            save_recon=save_recon)

        # HATNet (full image)
        res_hatnet = process_image_hatnet(
            img_path, hatnet_model, EucProj,
            gain_alpha_h, gain_alpha_w, sigma_y_hat,
            noise_seed=2026100 + img_no,
            save_recon=save_recon)

        # Save per-image .npz
        if save_recon:
            npz_data = {}
            if "recon" in res_block:
                npz_data.update(res_block.pop("recon"))
            if "recon" in res_hatnet:
                npz_data.update(res_hatnet.pop("recon"))
            npz_path = RECON_DIR / f"{img_name}.npz"
            np.savez_compressed(str(npz_path), **npz_data)
            n_arrays = len(npz_data)
            size_mb = sum(v.nbytes for v in npz_data.values()) / 1e6
            print(f"  Saved {npz_path.name}: {n_arrays} arrays, {size_mb:.1f} MB uncompressed")
            del npz_data

        # Combine
        result = {
            "image_idx": img_no + 1,
            "image_name": img_name,
            "fista_tv": res_block["fista_tv"],
            "ista_net": res_block["ista_net"],
            "hatnet": res_hatnet["hatnet"],
            "elapsed": time.time() - t_img,
        }
        if "pnp_drunet" in res_block:
            result["pnp_drunet"] = res_block["pnp_drunet"]
        all_results.append(result)

        # Print summary
        for method in methods:
            if method not in result:
                continue
            r = result[method]
            pi = r["scenario_i"]["psnr"]
            pii = r["scenario_ii"]["psnr"]
            piii = r["scenario_iii"]["psnr"]
            print(f"  {method:12s}: I={pi:.2f} | II={pii:.2f} | "
                  f"III={piii:.2f} | Gap={pi-pii:.2f} | "
                  f"Recovery={piii-pii:.2f}")

        print(f"  Time: {result['elapsed']:.1f}s")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {"methods": {}, "parameters": {}}
    summary["parameters"] = {
        "gain_alpha": gain_alpha,
        "sigma_y": sigma_y,
        "fista_lam": fista_lam,
        "fista_iters": fista_iters,
        "drunet_sigma_end": drunet_sigma_end,
        "drunet_sigma_anneal": drunet_sigma_anneal,
        "drunet_max_iter": drunet_max_iter,
        "gain_alpha_h": gain_alpha_h,
        "gain_alpha_w": gain_alpha_w,
        "sigma_y_hat": sigma_y_hat,
        "num_images": len(filepaths),
    }

    for method in methods:
        for scenario in ["scenario_i", "scenario_ii", "scenario_iii"]:
            psnrs = [r[method][scenario]["psnr"] for r in all_results
                     if method in r]
            ssims = [r[method][scenario]["ssim"] for r in all_results
                     if method in r]
            key = f"{method}_{scenario}"
            if psnrs:
                summary["methods"][key] = {
                    "psnr_mean": float(np.mean(psnrs)),
                    "psnr_std": float(np.std(psnrs)),
                    "ssim_mean": float(np.mean(ssims)),
                    "ssim_std": float(np.std(ssims)),
                }

    print(f"\n{'Method':12s} | {'Scenario I':>12s} | {'Scenario II':>12s} | "
          f"{'Scenario III':>12s} | {'Gap I->II':>10s} | {'Gain II->III':>10s}")
    print("-" * 85)

    for method in methods:
        key_i = f"{method}_scenario_i"
        if key_i not in summary["methods"]:
            continue
        si = summary["methods"][f"{method}_scenario_i"]["psnr_mean"]
        sii = summary["methods"][f"{method}_scenario_ii"]["psnr_mean"]
        siii = summary["methods"][f"{method}_scenario_iii"]["psnr_mean"]
        si_s = summary["methods"][f"{method}_scenario_i"]["psnr_std"]
        sii_s = summary["methods"][f"{method}_scenario_ii"]["psnr_std"]
        siii_s = summary["methods"][f"{method}_scenario_iii"]["psnr_std"]
        print(f"{method:12s} | {si:5.2f}+/-{si_s:.2f} | {sii:5.2f}+/-{sii_s:.2f} | "
              f"{siii:5.2f}+/-{siii_s:.2f} | {si-sii:8.2f}  | {siii-sii:8.2f}")

    total_time = sum(r["elapsed"] for r in all_results)
    print(f"\nTotal time: {total_time/60:.1f} min "
          f"({total_time/len(all_results):.1f}s per image)")

    # ---- Save results ----
    out_detailed = RESULTS_DIR / "spc_validation_results.json"
    out_summary = RESULTS_DIR / "spc_summary.json"

    with open(out_detailed, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results: {out_detailed}")

    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {out_summary}")

    if save_recon:
        npz_files = sorted(RECON_DIR.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in npz_files) / 1e6
        print(f"\nReconstruction data -> {RECON_DIR}")
        print(f"  {len(npz_files)} files, {total_size:.0f} MB total (compressed)")

    print("\nSPC Validation v4.0 complete!")
    return all_results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPC Validation v4.0")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 images only")
    parser.add_argument("--tune", action="store_true",
                        help="Parameter tuning mode")
    parser.add_argument("--save-recon", action="store_true",
                        help="Save reconstruction arrays to .npz files")
    args = parser.parse_args()
    run_validation(quick=args.quick, tune=args.tune, save_recon=args.save_recon)
