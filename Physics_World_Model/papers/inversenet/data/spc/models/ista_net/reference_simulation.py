#!/usr/bin/env python3
# ============================================================
# File: ista_blockcs_gain_noise_ista_and_pnp_fista_norm_fixpad.py
#
# Fixes:
#   1) DRUNet channel mismatch -> force grayscale DRUNet
#   2) DRUNet spatial mismatch on 33x33 -> pad-to-multiple-of-8 then crop
#   3) PyTorch .view() stride error -> use .reshape() and make crop contiguous
#
# Includes:
#   - Row-normalized operator for PnP stability
#   - Auto stepsize via spectral norm estimate
#   - PnP-FISTA with sigma annealing
# ============================================================

import os
import glob
import math
import json
import inspect
from time import time
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

# ---- deepinv ----
_HAS_DEEPINV = False
_DEEPINV_ERR = ""
try:
    import deepinv as dinv
    from deepinv.models import DnCNN, DRUNet
    _HAS_DEEPINV = True
except Exception as e:
    _DEEPINV_ERR = str(e)

# ============================================================
# Args
# ============================================================
parser = ArgumentParser(description="Block-CS gain+noise simulation with ISTA-Net + improved PnP-FISTA (pad/crop for DRUNet)")

# ISTA-Net args
parser.add_argument("--epoch_num", type=int, default=200)
parser.add_argument("--layer_num", type=int, default=9)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--group_num", type=int, default=1)
parser.add_argument("--cs_ratio", type=int, default=25, help="from {1,4,10,25,30,40,50}")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--matrix_dir", type=str, default="/home/spiritai/ISTA-Net-PyTorch-master/sampling_matrix")
parser.add_argument("--model_dir", type=str, default="/home/spiritai/ISTA-Net-PyTorch-master/model")
parser.add_argument("--data_dir", type=str, default="/home/spiritai/ISTA-Net-PyTorch-master/data")
parser.add_argument("--test_name", type=str, default="Set11")

# output
parser.add_argument("--out_root", type=str, default="/home/spiritai/ISTA-Net-PyTorch-master/gain_noise_runs/run_fista_fixpad_use")

# run switches
parser.add_argument("--run_ista", action="store_true")
parser.add_argument("--run_pnp", action="store_true")

# simulation params
parser.add_argument("--sigma_y", type=float, default=0.03)
parser.add_argument("--drift_type", type=str, default="exp", choices=["exp", "linear"])
parser.add_argument("--base_alpha", type=float, default=2e-3)
parser.add_argument("--base_linear_end", type=float, default=0.85)
parser.add_argument("--alpha_jitter_rel", type=float, default=0.5)
parser.add_argument("--end_jitter_abs", type=float, default=0.03)
parser.add_argument("--gain_seed0", type=int, default=2026001)

parser.add_argument("--gain_clamp_min", type=float, default=0.15)
parser.add_argument("--gain_clamp_max", type=float, default=1.0)

# PnP grid
parser.add_argument("--pnp_sigma_grid", type=str, default="0.01,0.02,0.03")
parser.add_argument("--pnp_stepsize_grid", type=str, default="0,0.7,1.0")   # 0 means use tau_auto
parser.add_argument("--pnp_max_iter_grid", type=str, default="200,400")
parser.add_argument("--pnp_select_by", type=str, default="psnr", choices=["psnr", "resid"])

parser.add_argument("--pnp_denoiser", type=str, default="drunet", choices=["dncnn", "drunet"])
parser.add_argument("--pnp_init", type=str, default="backproj", choices=["backproj", "ista"])

# annealing: sigma_start = anneal_mult * sigma_end
parser.add_argument("--pnp_sigma_anneal_mult", type=float, default=3.0)

# power iteration for Lipschitz
parser.add_argument("--pnp_power_iter", type=int, default=20)

# padding multiple for U-Net denoisers
parser.add_argument("--pnp_pad_mult", type=int, default=8)

parser.set_defaults(run_ista=True, run_pnp=True)
args = parser.parse_args()

# ============================================================
# CUDA setup
# ============================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# ISTA-Net components
# ============================================================
ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
m_meas = ratio_dict[args.cs_ratio]
n_pix = 1089
block_size = 33

Phi_data_Name = os.path.join(args.matrix_dir, f"phi_0_{args.cs_ratio}_1089.mat")
if not os.path.exists(Phi_data_Name):
    raise FileNotFoundError(f"Phi file not found: {Phi_data_Name}")
Phi_np = sio.loadmat(Phi_data_Name)["phi"].astype(np.float32)

Qinit_Name = os.path.join(args.matrix_dir, f"Initialization_Matrix_{args.cs_ratio}.mat")

def compute_Qinit_from_training(Phi_input_np: np.ndarray, data_dir: str) -> np.ndarray:
    Training_path = os.path.join(data_dir, "Training_Data.mat")
    if not os.path.exists(Training_path):
        raise FileNotFoundError(f"Training_Data.mat missing at {Training_path}")
    Training_data = sio.loadmat(Training_path)
    X_data = Training_data["labels"].transpose()
    Y_data = np.dot(Phi_input_np, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit_np = np.dot(X_YT, np.linalg.inv(Y_YT))
    return Qinit_np.astype(np.float32)

if os.path.exists(Qinit_Name):
    Qinit_np = sio.loadmat(Qinit_Name)["Qinit"].astype(np.float32)
else:
    Qinit_np = compute_Qinit_from_training(Phi_np, args.data_dir)
    sio.savemat(Qinit_Name, {"Qinit": Qinit_np})

class BasicBlock(nn.Module):
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
        x_input = x.reshape(-1, 1, block_size, block_size)  # reshape safer than view

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.reshape(-1, n_pix)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]

class ISTANet(nn.Module):
    def __init__(self, LayerNo: int):
        super().__init__()
        self.LayerNo = LayerNo
        self.fcs = nn.ModuleList([BasicBlock() for _ in range(LayerNo)])

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(Phi.t(), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = torch.mm(Phix, Qinit.t())
        layers_sym = []
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        return [x, layers_sym]

model = ISTANet(args.layer_num)
model = nn.DataParallel(model).to(device)

model_dir = f"{args.model_dir}/CS_ISTA_Net_layer_{args.layer_num}_group_{args.group_num}_ratio_{args.cs_ratio}_lr_{args.learning_rate:.4f}"
ckpt_path = f"{model_dir}/net_params_{args.epoch_num}.pkl"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"ISTA-Net checkpoint not found: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

Phi_assumed = torch.from_numpy(Phi_np).float().to(device)     # [m,n]
Qinit_assumed = torch.from_numpy(Qinit_np).float().to(device) # [n,m]

# ============================================================
# Helpers
# ============================================================
def imread_CS_py(Iorg: np.ndarray, block_size: int = 33):
    row, col = Iorg.shape
    row_pad = (block_size - np.mod(row, block_size)) % block_size
    col_pad = (block_size - np.mod(col, block_size)) % block_size
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad], dtype=Iorg.dtype)), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad], dtype=Iorg.dtype)), axis=0)
    row_new, col_new = Ipad.shape
    return Iorg, row, col, Ipad, row_new, col_new

def img2col_py(Ipad: np.ndarray, block_size: int = 33) -> np.ndarray:
    row, col = Ipad.shape
    row_block = row // block_size
    col_block = col // block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size**2, block_num], dtype=np.float32)
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count += 1
    return img_col

def col2im_CS_py(X_col: np.ndarray, row: int, col: int, row_new: int, col_new: int, block_size: int = 33) -> np.ndarray:
    X0_rec = np.zeros([row_new, col_new], dtype=np.float32)
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            count += 1
    return X0_rec[:row, :col]

def psnr_255(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = float(np.mean((img1 - img2) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

def normalize_01_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)

# ============================================================
# Gain drift
# ============================================================
def make_gain_vector_exp(m: int, alpha: float, device: torch.device) -> torch.Tensor:
    i = torch.arange(m, device=device, dtype=torch.float32)
    return torch.exp(-alpha * i)

def make_gain_vector_linear(m: int, end: float, device: torch.device) -> torch.Tensor:
    return torch.linspace(1.0, float(end), steps=m, device=device, dtype=torch.float32)

def make_gain_all(*, B: int, m: int, device: torch.device, drift_type: str,
                  base_alpha: float, base_linear_end: float, alpha_jitter_rel: float,
                  end_jitter_abs: float, seed0: int) -> torch.Tensor:
    rng = np.random.RandomState(seed0)
    gains = []
    for _ in range(B):
        if drift_type == "exp":
            jitter = (rng.rand() * 2 - 1) * alpha_jitter_rel
            alpha_i = float(base_alpha * (1.0 + jitter))
            g = make_gain_vector_exp(m, alpha=alpha_i, device=device)
        else:
            jitter = (rng.rand() * 2 - 1) * end_jitter_abs
            end_i = float(np.clip(base_linear_end + jitter, 0.1, 2.0))
            g = make_gain_vector_linear(m, end=end_i, device=device)
        gains.append(g)
    return torch.stack(gains, dim=0)

# ============================================================
# ISTA-Net recon
# ============================================================
@torch.no_grad()
def istanet_recon_blocks(y_Bm: torch.Tensor, Phi_mn: torch.Tensor, Qinit_nm: torch.Tensor) -> torch.Tensor:
    x_out, _ = model(y_Bm, Phi_mn, Qinit_nm)
    return x_out

# ============================================================
# PnP utilities
# ============================================================
def _supports_kwarg(fn, kw: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return kw in sig.parameters
    except Exception:
        return False

def _pad_to_multiple(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """
    Pad BCHW tensor so H,W are multiples of mult. Returns padded tensor and pad tuple (l,r,t,b).
    """
    if x.dim() != 4:
        raise ValueError(f"_pad_to_multiple expects BCHW, got {tuple(x.shape)}")
    B, C, H, W = x.shape
    if mult <= 1:
        return x, (0,0,0,0)
    Hp = int(math.ceil(H / mult) * mult)
    Wp = int(math.ceil(W / mult) * mult)
    pad_h = Hp - H
    pad_w = Wp - W
    # symmetric-ish padding: (left,right,top,bottom)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)

def _crop_from_pad(x_pad: torch.Tensor, pad: Tuple[int,int,int,int], H: int, W: int) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    # crop back to original H,W centered; make contiguous to avoid .view() stride issues
    y = x_pad[:, :, pad_top:pad_top+H, pad_left:pad_left+W]
    return y.contiguous()

def _denoise_call(denoiser, x_img: torch.Tensor, sigma_val: float, pad_mult: int) -> torch.Tensor:
    """
    Robust denoiser call + pad/crop so output matches input spatial size.
    Fixes DRUNet failing on 33x33.
    """
    H, W = x_img.shape[-2], x_img.shape[-1]
    x_pad, pad = _pad_to_multiple(x_img, mult=int(pad_mult))

    fwd = denoiser.forward if hasattr(denoiser, "forward") else denoiser
    if _supports_kwarg(fwd, "sigma"):
        y_pad = denoiser(x_pad, sigma=float(sigma_val))
    elif _supports_kwarg(fwd, "noise_level"):
        y_pad = denoiser(x_pad, noise_level=float(sigma_val))
    else:
        y_pad = denoiser(x_pad)

    # Some denoisers can slightly change size if they use valid conv; force crop.
    y = _crop_from_pad(y_pad, pad, H=H, W=W)
    return y

def _load_denoiser(name: str, device: torch.device):
    if not _HAS_DEEPINV:
        raise RuntimeError(f"deepinv not available: {_DEEPINV_ERR}")

    if name == "dncnn":
        return DnCNN(in_channels=1, out_channels=1, pretrained="download", device=device).eval()

    # DRUNet: FORCE GRAYSCALE
    tried = []
    for kwargs in [
        {"in_channels": 1, "out_channels": 1, "device": device},
        {"in_channels": 1, "out_channels": 1},
        {"in_channels": 1},
        {},
    ]:
        try:
            m = DRUNet(**kwargs).to(device).eval()
            return m
        except Exception as e:
            tried.append((kwargs, str(e)))

    print("[WARN] Could not build DRUNet grayscale; falling back to DnCNN.")
    for kw, err in tried[:3]:
        print("   tried", kw, "->", err)
    return DnCNN(in_channels=1, out_channels=1, pretrained="download", device=device).eval()

@torch.no_grad()
def row_normalize_operator(y_Bm: torch.Tensor, Phi_mn: torch.Tensor, eps: float = 1e-8):
    row_norm = torch.linalg.norm(Phi_mn, dim=1).clamp_min(eps)        # [m]
    PhiN = Phi_mn / row_norm[:, None]
    yN = y_Bm / row_norm[None, :]
    return yN, PhiN

@torch.no_grad()
def estimate_L_power(Phi_mn: torch.Tensor, iters: int = 20, eps: float = 1e-12) -> float:
    n = Phi_mn.shape[1]
    v = torch.randn(n, device=Phi_mn.device, dtype=Phi_mn.dtype)
    v = v / (v.norm() + eps)
    for _ in range(int(iters)):
        w = Phi_mn.t() @ (Phi_mn @ v)
        wn = w.norm() + eps
        v = w / wn
    w = Phi_mn @ v
    s = w.norm()
    L = float((s * s).item())
    return max(L, 1e-8)

@torch.no_grad()
def backproj_init_blocks(y_Bm: torch.Tensor, Phi_mn: torch.Tensor) -> torch.Tensor:
    x0 = y_Bm @ Phi_mn
    x0 = x0.reshape(-1, 1, block_size, block_size)  # reshape safer than view
    x0 = normalize_01_tensor(x0).reshape(-1, n_pix)
    return x0.clamp(0.0, 1.0)

@torch.no_grad()
def pnp_fista_run(
    *,
    y_Bm: torch.Tensor,
    Phi_mn: torch.Tensor,
    x0_Bn: torch.Tensor,
    denoiser,
    sigma_end: float,
    sigma_anneal_mult: float,
    stepsize: float,
    max_iter: int,
    pad_mult: int,
):
    B = y_Bm.shape[0]
    x = x0_Bn.clone()
    z = x0_Bn.clone()
    t = 1.0

    sigma_start = float(sigma_anneal_mult) * float(sigma_end)

    for k in range(int(max_iter)):
        a = k / max(int(max_iter) - 1, 1)
        sigma_k = (1 - a) * sigma_start + a * float(sigma_end)

        y_hat = x @ Phi_mn.t()
        grad = (y_hat - y_Bm) @ Phi_mn
        u = x - float(stepsize) * grad
        u_img = u.reshape(B, 1, block_size, block_size)  # reshape safer than view

        z_new_img = _denoise_call(denoiser, u_img, sigma_val=float(sigma_k), pad_mult=int(pad_mult))
        z_new = z_new_img.reshape(B, -1).clamp(0.0, 1.0)  # reshape fixes stride/view error

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        x = z_new + ((t - 1.0) / t_new) * (z_new - z)
        x = x.clamp(0.0, 1.0)

        z = z_new
        t = t_new

    return z.clamp(0.0, 1.0)

@torch.no_grad()
def pnp_grid_search_blocks(
    *,
    y_Bm: torch.Tensor,
    Phi_mn: torch.Tensor,
    x0_Bn: torch.Tensor,
    x_gt_Bn: Optional[torch.Tensor],
    select_by: str,
    sigma_grid: List[float],
    stepsize_grid: List[float],
    max_iter_grid: List[int],
    denoiser_name: str,
    power_iters: int,
    sigma_anneal_mult: float,
    pad_mult: int,
):
    denoiser = _load_denoiser(denoiser_name, device=device)
    metric_psnr = dinv.loss.metric.PSNR() if _HAS_DEEPINV else None

    # normalize operator for stability
    yN, PhiN = row_normalize_operator(y_Bm, Phi_mn)

    # auto Lipschitz
    L = estimate_L_power(PhiN, iters=power_iters)
    tau_auto = 0.9 / L

    best_score = -1e18
    best_cfg = None
    best_x = None

    for sd in sigma_grid:
        for st_scale in stepsize_grid:
            tau = tau_auto if float(st_scale) == 0.0 else float(st_scale) * tau_auto
            for it in max_iter_grid:
                x_hat = pnp_fista_run(
                    y_Bm=yN,
                    Phi_mn=PhiN,
                    x0_Bn=x0_Bn,
                    denoiser=denoiser,
                    sigma_end=float(sd),
                    sigma_anneal_mult=float(sigma_anneal_mult),
                    stepsize=float(tau),
                    max_iter=int(it),
                    pad_mult=int(pad_mult),
                )

                if select_by == "psnr" and x_gt_Bn is not None and metric_psnr is not None:
                    x_hat_img = x_hat.reshape(-1, 1, block_size, block_size)
                    x_gt_img = x_gt_Bn.reshape(-1, 1, block_size, block_size)
                    ps = metric_psnr(x_hat_img, x_gt_img)
                    score = float(ps.mean().item())
                else:
                    resid = torch.mean((x_hat @ PhiN.t() - yN) ** 2).item()
                    score = -float(resid)

                if score > best_score:
                    best_score = score
                    best_cfg = {
                        "sigma_end": float(sd),
                        "tau": float(tau),
                        "tau_auto": float(tau_auto),
                        "L_est": float(L),
                        "stepsize_scale": float(st_scale),
                        "max_iter": int(it),
                        "score": float(score),
                        "select_by": select_by,
                        "denoiser": denoiser_name,
                        "algo": "PnP-FISTA rownorm+anneal+pad",
                        "sigma_anneal_mult": float(sigma_anneal_mult),
                        "pad_mult": int(pad_mult),
                    }
                    best_x = x_hat.detach().clone()

    return best_x, {"best_cfg": best_cfg}

# ============================================================
# Main
# ============================================================
def main():
    if not args.run_ista and not args.run_pnp:
        print("[WARN] Nothing to run. Use --run_ista and/or --run_pnp.")
        return

    if args.run_pnp and not _HAS_DEEPINV:
        raise RuntimeError(
            "You requested --run_pnp but deepinv is not available.\n"
            f"deepinv import error: {_DEEPINV_ERR}"
        )

    sigma_grid = [float(x) for x in args.pnp_sigma_grid.split(",") if x.strip()]
    stepsize_grid = [float(x) for x in args.pnp_stepsize_grid.split(",") if x.strip()]
    max_iter_grid = [int(x) for x in args.pnp_max_iter_grid.split(",") if x.strip()]

    out_root = Path(args.out_root)
    pack_dir = out_root / "pack"
    recon_dir = out_root / "recon"
    pack_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    test_dir = os.path.join(args.data_dir, args.test_name)
    filepaths = sorted(glob.glob(os.path.join(test_dir, "*.tif")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No .tif found in {test_dir}")

    names = [Path(p).name for p in filepaths]
    ImgNum = len(filepaths)

    gain_all = make_gain_all(
        B=ImgNum, m=m_meas, device=device,
        drift_type=args.drift_type,
        base_alpha=args.base_alpha,
        base_linear_end=args.base_linear_end,
        alpha_jitter_rel=args.alpha_jitter_rel,
        end_jitter_abs=args.end_jitter_abs,
        seed0=args.gain_seed0,
    ).clamp(args.gain_clamp_min, args.gain_clamp_max)

    x_list, y_list, y_clean_list, y_ideal_list = [], [], [], []
    y_corr_list = []

    metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "assumed_measured": {"ISTA": [], "PnP": []},
        "real_measured":    {"ISTA": [], "PnP": []},
        "assumed_ideal":    {"ISTA": [], "PnP": []},
    }

    print("\n[SIM+RECON] Start")
    print(f"[SIM] cs_ratio={args.cs_ratio} m={m_meas} block={block_size} sigma_y={args.sigma_y}")
    print(f"[RUN] run_ista={args.run_ista} run_pnp={args.run_pnp}")
    if args.run_pnp:
        print(f"[PnP] denoiser={args.pnp_denoiser} select_by={args.pnp_select_by} pad_mult={args.pnp_pad_mult}")
        print(f"[PnP] sigma_grid={sigma_grid} stepsize_grid={stepsize_grid} max_iter_grid={max_iter_grid}")

    with torch.no_grad():
        for img_no, img_path in enumerate(filepaths):
            img_name = Path(img_path).name
            stem = Path(img_path).stem
            img_out_dir = recon_dir / stem
            img_out_dir.mkdir(parents=True, exist_ok=True)

            Img = cv2.imread(img_path, 1)
            if Img is None:
                raise RuntimeError(f"cv2.imread failed: {img_path}")

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Iorg_y = Img_yuv[:, :, 0].astype(np.float32)

            Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg_y, block_size=block_size)
            Icol = img2col_py(Ipad, block_size).transpose() / 255.0
            x_blocks = torch.from_numpy(Icol).float().to(device)

            y_ideal = x_blocks @ Phi_assumed.t()

            g = gain_all[img_no].view(1, -1)
            Phi_real = (g.t() * Phi_assumed)
            y_clean = x_blocks @ Phi_real.t()
            y_meas = y_clean + float(args.sigma_y) * torch.randn_like(y_clean)
            y_corr = y_meas / g

            x_list.append(x_blocks.detach().cpu())
            y_list.append(y_meas.detach().cpu())
            y_clean_list.append(y_clean.detach().cpu())
            y_ideal_list.append(y_ideal.detach().cpu())
            y_corr_list.append(y_corr.detach().cpu())

            # ISTA (network-compatible)
            if args.run_ista:
                ista_cases = [
                    ("assumed_measured", y_meas,  Phi_assumed),
                    ("real_measured",    y_corr,  Phi_assumed),
                    ("assumed_ideal",    y_ideal, Phi_assumed),
                ]
                for tag, y_used, Phi_used in ista_cases:
                    xhat = istanet_recon_blocks(y_used, Phi_used, Qinit_assumed).clamp(0, 1)
                    xhat_np = xhat.detach().cpu().numpy().transpose()
                    rec_y = col2im_CS_py(xhat_np, row, col, row_new, col_new, block_size)

                    rec_psnr = psnr_255(rec_y * 255.0, Iorg)
                    rec_ssim = ssim(rec_y * 255.0, Iorg.astype(np.float64), data_range=255)
                    metrics[tag]["ISTA"].append({"name": img_name, "psnr": float(rec_psnr), "ssim": float(rec_ssim)})

                print(f"[{img_no+1:02d}/{ImgNum:02d}] {img_name} | assumed_measured: ISTA {metrics['assumed_measured']['ISTA'][-1]['psnr']:.2f}")
                print(f"[{img_no+1:02d}/{ImgNum:02d}] {img_name} | real_measured:    ISTA {metrics['real_measured']['ISTA'][-1]['psnr']:.2f}")
                print(f"[{img_no+1:02d}/{ImgNum:02d}] {img_name} | assumed_ideal:    ISTA {metrics['assumed_ideal']['ISTA'][-1]['psnr']:.2f}")

            # PnP
            if args.run_pnp:
                pnp_cases = [
                    ("assumed_measured", y_meas,  Phi_assumed),
                    ("real_measured",    y_meas,  Phi_real),
                    ("assumed_ideal",    y_ideal, Phi_assumed),
                ]

                for tag, y_used, Phi_used in pnp_cases:
                    t0 = time()

                    if args.pnp_init == "ista" and args.run_ista and tag != "real_measured":
                        x0 = istanet_recon_blocks(y_used, Phi_used, Qinit_assumed).clamp(0, 1)
                    else:
                        x0 = backproj_init_blocks(y_used, Phi_used)

                    xhat_pnp, info = pnp_grid_search_blocks(
                        y_Bm=y_used,
                        Phi_mn=Phi_used,
                        x0_Bn=x0,
                        x_gt_Bn=x_blocks if args.pnp_select_by == "psnr" else None,
                        select_by=args.pnp_select_by,
                        sigma_grid=sigma_grid,
                        stepsize_grid=stepsize_grid,
                        max_iter_grid=max_iter_grid,
                        denoiser_name=args.pnp_denoiser,
                        power_iters=args.pnp_power_iter,
                        sigma_anneal_mult=args.pnp_sigma_anneal_mult,
                        pad_mult=args.pnp_pad_mult,
                    )

                    xhat_np = xhat_pnp.detach().cpu().numpy().transpose()
                    rec_y = col2im_CS_py(xhat_np, row, col, row_new, col_new, block_size)

                    rec_psnr = psnr_255(rec_y * 255.0, Iorg)
                    rec_ssim = ssim(rec_y * 255.0, Iorg.astype(np.float64), data_range=255)
                    metrics[tag]["PnP"].append({"name": img_name, "psnr": float(rec_psnr), "ssim": float(rec_ssim), "best_cfg": info.get("best_cfg", None)})

                    t1 = time()
                    print(f"[{img_no+1:02d}/{ImgNum:02d}] {img_name} | {tag}: PnP {rec_psnr:.2f}  time={t1-t0:.2f}s")

                    if info.get("best_cfg", None) is not None:
                        (img_out_dir / f"{stem}_{tag}_pnp_best_cfg.json").write_text(json.dumps(info["best_cfg"], indent=2), encoding="utf-8")

    # Save pack
    torch.save(x_list, pack_dir / "x.pt")
    torch.save(y_list, pack_dir / "y.pt")
    torch.save(y_clean_list, pack_dir / "y_clean.pt")
    torch.save(y_ideal_list, pack_dir / "y_ideal.pt")
    torch.save(y_corr_list, pack_dir / "y_corr.pt")
    torch.save(gain_all.detach().cpu(), pack_dir / "gain_all.pt")
    (pack_dir / "names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    meta: Dict[str, Any] = {
        "mode": "ista_block_cs_gain_noise_ista_and_pnp_fista_norm_fixpad",
        "cs_ratio": int(args.cs_ratio),
        "m": int(m_meas),
        "n": int(n_pix),
        "block": [block_size, block_size],
        "simulation": {"sigma_y": float(args.sigma_y)},
        "pnp": {
            "algo": "PnP-FISTA rownorm+anneal+pad",
            "denoiser": args.pnp_denoiser,
            "sigma_anneal_mult": args.pnp_sigma_anneal_mult,
            "power_iter": args.pnp_power_iter,
            "pad_mult": int(args.pnp_pad_mult),
        },
    }
    (pack_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (out_root / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n[DONE] Saved:", out_root / "summary_metrics.json")

if __name__ == "__main__":
    main()
