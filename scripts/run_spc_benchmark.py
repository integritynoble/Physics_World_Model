#!/usr/bin/env python3
"""SPC Set11 Benchmark — ISTA-Net+ / HATNet / ADMM-DCT-TV on real Set11 images.

Runs 3 solvers at 4 sampling rates (4%, 10%, 25%, 50%) on the 11-image
Set11 benchmark (256×256 grayscale). Updates pwm/reports/spc.md.

Usage:
    python scripts/run_spc_benchmark.py
    python scripts/run_spc_benchmark.py --rates 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import scipy.io as sio
import cv2

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

ISTA_ROOT = "/home/spiritai/ISTA-Net-PyTorch-master"
HATNET_ROOT = "/home/spiritai/HATNet-SPI-master"

SET11_DIR = os.path.join(ISTA_ROOT, "data", "Set11")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# ── Constants ────────────────────────────────────────────────────────────
BLOCK_SIZE = 33
N_PIX = BLOCK_SIZE ** 2  # 1089
ALL_RATES = [4, 10, 25, 50]
RATE_TO_MEAS = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

# HATNet meas_size per compression ratio
HATNET_MEAS_SIZE = {
    0.04: [51, 51],
    0.10: [81, 81],
    0.25: [128, 128],
    0.50: [181, 181],
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

try:
    from skimage.metrics import structural_similarity as compute_ssim
except ImportError:
    from skimage.measure import compare_ssim as compute_ssim

# ── Helpers ──────────────────────────────────────────────────────────────

def psnr_255(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_255(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(compute_ssim(
        img1.astype(np.float64), img2.astype(np.float64), data_range=255.0
    ))


def load_set11_images():
    """Load Set11 images as grayscale Y-channel, return list of (name, img_uint8)."""
    filepaths = sorted(Path(SET11_DIR).glob("*.tif"))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No .tif files found in {SET11_DIR}")
    images = []
    for fp in filepaths:
        img = cv2.imread(str(fp), 1)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {fp}")
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_y = img_yuv[:, :, 0]  # uint8
        images.append((fp.stem, img_y))
    return images


def imread_CS_py(Iorg, block_size=BLOCK_SIZE):
    row, col = Iorg.shape
    row_pad = (block_size - np.mod(row, block_size)) % block_size
    col_pad = (block_size - np.mod(col, block_size)) % block_size
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad], dtype=Iorg.dtype)), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad], dtype=Iorg.dtype)), axis=0)
    row_new, col_new = Ipad.shape
    return Iorg, row, col, Ipad, row_new, col_new


def img2col_py(Ipad, block_size=BLOCK_SIZE):
    row, col = Ipad.shape
    block_num = (row // block_size) * (col // block_size)
    img_col = np.zeros([block_size ** 2, block_num], dtype=np.float32)
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape(-1)
            count += 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new, block_size=BLOCK_SIZE):
    X0_rec = np.zeros([row_new, col_new], dtype=np.float32)
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape(block_size, block_size)
            count += 1
    return X0_rec[:row, :col]


# ══════════════════════════════════════════════════════════════════════════
# Solver 1: ISTA-Net+ (block-CS, 33×33)
# ══════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, BLOCK_SIZE, BLOCK_SIZE)
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = (x_input + x_G).view(-1, N_PIX)
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D
        return [x_pred, symloss]


class ISTANetPlus(nn.Module):
    def __init__(self, layer_num=9):
        super().__init__()
        self.LayerNo = layer_num
        self.fcs = nn.ModuleList([BasicBlock() for _ in range(layer_num)])

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(Phi.t(), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = torch.mm(Phix, Qinit.t())
        layers_sym = []
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        return [x, layers_sym]


def load_ista_net(cs_ratio, device):
    """Load ISTA-Net+ model and measurement matrix for given cs_ratio."""
    # Load Phi
    phi_path = os.path.join(ISTA_ROOT, "sampling_matrix", f"phi_0_{cs_ratio}_1089.mat")
    Phi_np = sio.loadmat(phi_path)["phi"].astype(np.float32)

    # Load Qinit
    qinit_path = os.path.join(ISTA_ROOT, "sampling_matrix", f"Initialization_Matrix_{cs_ratio}.mat")
    Qinit_np = sio.loadmat(qinit_path)["Qinit"].astype(np.float32)

    # Load model
    model = ISTANetPlus(layer_num=9)
    model = nn.DataParallel(model).to(device)
    ckpt_path = os.path.join(ISTA_ROOT, "model",
                             f"CS_ISTA_Net_plus_layer_9_group_1_ratio_{cs_ratio}_lr_0.0001",
                             "net_params_200.pkl")
    if not os.path.exists(ckpt_path):
        # Fallback to non-plus model
        ckpt_path = os.path.join(ISTA_ROOT, "model",
                                 f"CS_ISTA_Net_layer_9_group_1_ratio_{cs_ratio}_lr_0.0001",
                                 "net_params_200.pkl")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    Phi_t = torch.from_numpy(Phi_np).float().to(device)
    Qinit_t = torch.from_numpy(Qinit_np).float().to(device)
    return model, Phi_t, Qinit_t, Phi_np


@torch.no_grad()
def run_ista_net(model, Phi_t, Qinit_t, Phi_np, img_y, device):
    """Run ISTA-Net+ on a single image. Returns reconstruction (same size as img_y)."""
    Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(img_y.astype(np.float32))
    Icol = img2col_py(Ipad).transpose() / 255.0  # (B, 1089) in [0,1]
    x_blocks = torch.from_numpy(Icol).float().to(device)

    # Forward measurement + reconstruct
    y_blocks = x_blocks @ Phi_t.t()
    x_hat, _ = model(y_blocks, Phi_t, Qinit_t)
    x_hat = x_hat.clamp(0, 1)

    # Reassemble
    xhat_np = x_hat.cpu().numpy().transpose()  # (1089, B)
    rec = col2im_CS_py(xhat_np, row, col, row_new, col_new)
    return rec * 255.0  # back to [0, 255]


# ══════════════════════════════════════════════════════════════════════════
# Solver 2: HATNet (full-image, 256×256)
# ══════════════════════════════════════════════════════════════════════════

def _load_checkpoint_hatnet(model, pretrained_dict):
    """Minimal load_checkpoint without albumentations dependency."""
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys()}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    print(f'  Model params: {len(model_dict)}, Pretrained: {len(pretrained_model_dict)}, Loaded: {len(load_dict)}')


def load_hatnet(cr, device):
    """Load HATNet model for given compression ratio."""
    if HATNET_ROOT not in sys.path:
        sys.path.insert(0, HATNET_ROOT)
    from model.network import HATNet

    meas_size = HATNET_MEAS_SIZE[cr]
    model = HATNet(
        imag_size=[256, 256],
        meas_size=meas_size,
        img_channels=1,
        channels=64,
        mid_blocks=1,
        enc_blocks=[1, 1],
        dec_blocks=[1, 1],
        stages=7,
        matrix_train=True,
    ).to(device)

    weight_path = os.path.join(HATNET_ROOT, "weights", "2024_pretraiend_weights", f"cr_{cr}.pth")
    pretrained_dict = torch.load(weight_path, map_location=device)
    _load_checkpoint_hatnet(model, pretrained_dict)
    model.eval()
    return model


@torch.no_grad()
def run_hatnet(model, img_y, device):
    """Run HATNet on a single 256×256 image."""
    import einops

    gt = img_y.astype(np.float32)
    h, w = gt.shape

    # HATNet expects 256×256; handle larger/smaller via blocking
    if h == 256 and w == 256:
        inp = einops.rearrange(gt, '(a h) (b w) -> (a b) h w', a=1, b=1)
    else:
        inp = einops.rearrange(gt, '(a h) (b w) -> (a b) h w', a=2, b=2)

    x = torch.from_numpy(inp / 255.0).float().to(device)
    out, _, _, _, _ = model(x)
    out = out.cpu().numpy()

    batch = out.shape[0]
    if batch > 1:
        pic = einops.rearrange(out, '(a b) h w -> (a h) (b w)', a=2, b=2)
    else:
        pic = einops.rearrange(out, '(a b) h w -> (a h) (b w)', a=1, b=1)

    return pic * 255.0  # back to [0, 255]


# ══════════════════════════════════════════════════════════════════════════
# Solver 3: ADMM-DCT-TV (block-CS, classical)
# ══════════════════════════════════════════════════════════════════════════

from pwm_core.recon.cs_solvers import admm_tv


def run_admm_dct_tv(Phi_np, img_y):
    """Run ADMM-DCT-TV on a single image using the same Phi as ISTA-Net."""
    Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(img_y.astype(np.float32))
    Icol = img2col_py(Ipad)  # (1089, B)
    n_blocks = Icol.shape[1]

    Phi_d = Phi_np.astype(np.float64)
    recon_col = np.zeros_like(Icol)

    for b in range(n_blocks):
        x_block = Icol[:, b] / 255.0  # [0,1]
        y_block = Phi_d @ x_block.astype(np.float64)
        x_hat = admm_tv(y_block, Phi_d, (BLOCK_SIZE, BLOCK_SIZE),
                        mu_tv=0.002, mu_dct=0.008, rho=1.0,
                        max_iters=500, tv_inner_iters=15, non_negative=True)
        recon_col[:, b] = np.clip(x_hat.flatten(), 0, 1) * 255.0

    rec = col2im_CS_py(recon_col, row, col, row_new, col_new)
    return rec


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPC Set11 Benchmark")
    parser.add_argument("--rates", nargs="+", type=int, default=ALL_RATES,
                        help="Sampling rates to test (default: 4 10 25 50)")
    args = parser.parse_args()
    rates = args.rates

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"SPC Set11 Benchmark — ISTA-Net+ / HATNet / ADMM-DCT-TV")
    print(f"Rates: {rates}%  Device: {device}")
    print("=" * 80)

    # ── Load Set11 images ────────────────────────────────────────────────
    images = load_set11_images()
    print(f"Set11: {len(images)} images")
    for name, img in images:
        print(f"  {name}: {img.shape}")

    # ── Results storage ──────────────────────────────────────────────────
    all_results = {}

    for rate in rates:
        cr = rate / 100.0
        m = RATE_TO_MEAS[rate]

        print(f"\n{'=' * 80}")
        print(f"Sampling rate: {rate}% (M={m}/block, CR={cr})")
        print("=" * 80)

        # ── Load ISTA-Net+ ───────────────────────────────────────────────
        print(f"\n  Loading ISTA-Net+ (cs_ratio={rate}) ...")
        ista_model, Phi_t, Qinit_t, Phi_np = load_ista_net(rate, device)

        # ── Load HATNet ──────────────────────────────────────────────────
        print(f"  Loading HATNet (cr={cr}) ...")
        hatnet_model = load_hatnet(cr, device)

        rate_results = {"ista_net": [], "hatnet": [], "admm_dct_tv": []}

        for img_name, img_y in images:
            print(f"\n  [{img_name}]")

            # ISTA-Net+
            t0 = time.time()
            rec_ista = run_ista_net(ista_model, Phi_t, Qinit_t, Phi_np, img_y, device)
            t_ista = time.time() - t0
            p_ista = psnr_255(rec_ista, img_y.astype(np.float32))
            s_ista = ssim_255(rec_ista, img_y.astype(np.float32))
            rate_results["ista_net"].append({"name": img_name, "psnr": round(p_ista, 2), "ssim": round(s_ista, 4), "time": round(t_ista, 2)})
            print(f"    ISTA-Net+:   PSNR={p_ista:.2f} dB  SSIM={s_ista:.4f}  time={t_ista:.2f}s")

            # HATNet
            t0 = time.time()
            rec_hat = run_hatnet(hatnet_model, img_y, device)
            t_hat = time.time() - t0
            p_hat = psnr_255(rec_hat, img_y.astype(np.float32))
            s_hat = ssim_255(rec_hat, img_y.astype(np.float32))
            rate_results["hatnet"].append({"name": img_name, "psnr": round(p_hat, 2), "ssim": round(s_hat, 4), "time": round(t_hat, 2)})
            print(f"    HATNet:      PSNR={p_hat:.2f} dB  SSIM={s_hat:.4f}  time={t_hat:.2f}s")

            # ADMM-DCT-TV
            t0 = time.time()
            rec_admm = run_admm_dct_tv(Phi_np, img_y)
            t_admm = time.time() - t0
            p_admm = psnr_255(rec_admm, img_y.astype(np.float32))
            s_admm = ssim_255(rec_admm, img_y.astype(np.float32))
            rate_results["admm_dct_tv"].append({"name": img_name, "psnr": round(p_admm, 2), "ssim": round(s_admm, 4), "time": round(t_admm, 2)})
            print(f"    ADMM-DCT-TV: PSNR={p_admm:.2f} dB  SSIM={s_admm:.4f}  time={t_admm:.2f}s")

        # Averages
        for solver_name in ["ista_net", "hatnet", "admm_dct_tv"]:
            entries = rate_results[solver_name]
            avg_psnr = round(mean([e["psnr"] for e in entries]), 2)
            avg_ssim = round(mean([e["ssim"] for e in entries]), 4)
            avg_time = round(mean([e["time"] for e in entries]), 2)
            rate_results[f"{solver_name}_avg"] = {"psnr": avg_psnr, "ssim": avg_ssim, "time": avg_time}

        all_results[f"{rate}pct"] = rate_results

        # Clean up GPU
        del ista_model, hatnet_model, Phi_t, Qinit_t
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    solver_names = ["ISTA-Net+", "HATNet", "ADMM-DCT-TV"]
    solver_keys = ["ista_net", "hatnet", "admm_dct_tv"]

    print(f"\n{'=' * 80}")
    print("SUMMARY — Average PSNR (dB) / SSIM across 11 Set11 images")
    print("=" * 80)

    header = f"{'Rate':<10}" + "".join(f" {sn:>16}" for sn in solver_names)
    print(f"\nPSNR (dB):")
    print(header)
    print("-" * len(header))
    for rate in rates:
        key = f"{rate}pct"
        row = f"{rate}%{'':<7}"
        for sk in solver_keys:
            row += f" {all_results[key][f'{sk}_avg']['psnr']:>16.2f}"
        print(row)

    print(f"\nSSIM:")
    print(header)
    print("-" * len(header))
    for rate in rates:
        key = f"{rate}pct"
        row = f"{rate}%{'':<7}"
        for sk in solver_keys:
            row += f" {all_results[key][f'{sk}_avg']['ssim']:>16.4f}"
        print(row)

    # Per-image at 25%
    if 25 in rates:
        print(f"\n{'=' * 80}")
        print("PER-IMAGE PSNR at 25% sampling rate")
        print("=" * 80)
        header2 = f"{'Image':<16}" + "".join(f" {sn:>16}" for sn in solver_names)
        print(header2)
        print("-" * len(header2))
        for i, (img_name, _) in enumerate(images):
            row = f"{img_name:<16}"
            for sk in solver_keys:
                row += f" {all_results['25pct'][sk][i]['psnr']:>16.2f}"
            print(row)

    # ── Save results JSON ────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    results_path = os.path.join(RUNS_DIR, "spc_set11_benchmark_results.json")
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()[:12]
    except Exception:
        pass

    save_data = {
        "benchmark": "spc_set11",
        "results": all_results,
        "env": {
            "pwm_version": git_sha,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "date": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved: {results_path}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
