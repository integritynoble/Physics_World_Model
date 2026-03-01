#!/usr/bin/env python3
"""CT Benchmark — GPU Methods on Modal (T4 GPU).

Runs FBP, TV-ADMM, and PnP-ADMM with DRUNet on all three CT tiers using
a T4 GPU on Modal.  The GPU-accelerated fan-beam projector uses PyTorch
grid_sample for fast bilinear interpolation.

GPU algorithms:
  fbp_gpu       — Feldkamp FBP (same formula as CPU, GPU-accelerated)
  tv_admm_gpu   — TV-ADMM (gradient descent + TV prox) on GPU
  pnp_drunet    — PnP-ADMM with DRUNet denoiser (requires Modal volume)

Usage:
    modal run scripts/modal_run_ct_benchmark.py               # all tiers, all algos
    modal run scripts/modal_run_ct_benchmark.py --tier public # public only
    modal run scripts/modal_run_ct_benchmark.py --algo fbp_gpu

Prerequisites:
    modal setup
    modal volume ls pwm-models /checkpoint/DRUNet/  # verify DRUNet is present

Output (on local machine after Modal run completes):
    results/ct/benchmark_gpu_<timestamp>.json
    results/ct/benchmark_gpu_<timestamp>.csv
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import modal

# ── Modal setup ───────────────────────────────────────────────────────────────

app = modal.App("pwm-ct-benchmark")
vol = modal.Volume.from_name("pwm-models")

# Image: PyTorch + deepinv (for DRUNet) + numerics
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "h5py",
        "scikit-image",
        "deepinv",
    )
)

# ── CT geometry constants (must match generate_dataset.py) ───────────────────

IMAGE_SIZE  = 362
D_SO        = 800.0
D_SD        = 568.0
D_SSD       = D_SO + D_SD    # 1368.0
N_DET       = 736
DET_SPACING = 1.496
MU_SCALE    = 0.058
FBP_CAL     = 1.655           # empirical calibration (D_SSD Feldkamp + Hann)

# ══════════════════════════════════════════════════════════════════════════════
# GPU fan-beam operators (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════


def _gpu_project(x_t, angles_rad, device, n_steps=300):
    """Fan-beam forward projection on GPU.

    x_t:       torch.Tensor (H, W) on device, pixel-density units
    angles_rad: 1-D array or tensor of view angles
    Returns:   torch.Tensor (n_views, N_DET) in pixel-density × px (line integral)
    """
    import torch

    H, W = x_t.shape
    cy, cx = H / 2.0, W / 2.0
    n_views = len(angles_rad)

    det_pos = (torch.arange(N_DET, device=device, dtype=torch.float32) - N_DET / 2.0) * DET_SPACING

    # t parameter along each ray
    t_vals = torch.linspace(0.0, 1.0, n_steps, device=device, dtype=torch.float32)

    sino = torch.zeros(n_views, N_DET, device=device, dtype=torch.float32)
    x_img = x_t.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W) for grid_sample

    for i, angle in enumerate(angles_rad):
        angle = float(angle)
        import math
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        src_y = -D_SO * sin_a + cy
        src_x =  D_SO * cos_a + cx
        det_y =  D_SD * sin_a + det_pos * cos_a + cy   # (N_DET,)
        det_x = -D_SD * cos_a + det_pos * sin_a + cx   # (N_DET,)

        ray_y = det_y - src_y   # (N_DET,)
        ray_x = det_x - src_x   # (N_DET,)
        ray_len = torch.sqrt(ray_y ** 2 + ray_x ** 2)   # (N_DET,)

        # (N_DET, n_steps)
        sample_y = src_y + ray_y.unsqueeze(1) * t_vals.unsqueeze(0)
        sample_x = src_x + ray_x.unsqueeze(1) * t_vals.unsqueeze(0)

        # Normalize to [-1, 1] for grid_sample (align_corners=False)
        grid_y = (2.0 * sample_y / H - 1.0)   # (N_DET, n_steps)
        grid_x = (2.0 * sample_x / W - 1.0)   # (N_DET, n_steps)
        # grid_sample expects (N, Hout, Wout, 2) → treat N_DET as batch
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # (N_DET, 1, n_steps, 2)

        x_rep = x_img.expand(N_DET, -1, -1, -1)   # (N_DET, 1, H, W)
        vals = torch.nn.functional.grid_sample(
            x_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # (N_DET, 1, 1, n_steps)
        vals = vals.squeeze(1).squeeze(1)   # (N_DET, n_steps)

        step_size = ray_len / n_steps   # (N_DET,)
        sino[i] = (vals.sum(dim=1) * step_size)

    return sino   # (n_views, N_DET), pixel-density units


def _gpu_backproject(sino_t, angles_rad, device):
    """Fan-beam backprojection (adjoint) on GPU.

    sino_t:    torch.Tensor (n_views, N_DET) on device, pixel-density units
    Returns:   torch.Tensor (IMAGE_SIZE, IMAGE_SIZE)
    """
    import torch
    import math

    H = W = IMAGE_SIZE
    n_views = len(angles_rad)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    py = yy - H / 2.0   # (H, W)
    px = xx - W / 2.0   # (H, W)

    img = torch.zeros(H, W, device=device, dtype=torch.float64)

    for i, angle in enumerate(angles_rad):
        angle = float(angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        sy = -D_SO * sin_a
        sx =  D_SO * cos_a
        vy = py - sy   # (H, W)
        vx = px - sx   # (H, W)

        U = vy * sin_a - vx * cos_a   # (H, W)
        V = vy * cos_a + vx * sin_a   # (H, W)

        U_safe    = torch.where(U > 1e-6, U, torch.full_like(U, 1e-6))
        det_j_val = D_SSD * V / U_safe
        det_idx   = det_j_val / DET_SPACING + N_DET / 2.0

        valid_mask = (U > 0) & (det_idx >= 0) & (det_idx < N_DET - 1)
        d0    = det_idx.floor().long().clamp(0, N_DET - 2)
        d1    = d0 + 1
        frac  = (det_idx - d0.float()).double()

        row   = sino_t[i].double()
        bval  = (1.0 - frac) * row[d0] + frac * row[d1]
        bval  = torch.where(valid_mask, bval, torch.zeros_like(bval))

        img  += bval

    return (img / n_views).float()


def _gpu_step_size(angles_rad, device):
    """Estimate Lipschitz constant via 4-step power iteration on GPU."""
    import torch
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    u = torch.randn(IMAGE_SIZE, IMAGE_SIZE, device=device, generator=rng, dtype=torch.float32)
    u = u / u.norm()
    nrm = 1.0
    for _ in range(4):
        Au  = _gpu_project(u, angles_rad, device) * MU_SCALE
        u   = _gpu_backproject(Au, angles_rad, device) * MU_SCALE
        nrm = float(u.norm())
        if nrm > 0:
            u = u / nrm
    return 1.0 / max(nrm, 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# Sinogram preprocessing
# ══════════════════════════════════════════════════════════════════════════════


def _preprocess_sino_np(sino_nepers):
    """Clamp saturated rays and smooth (NumPy)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    sino_pp = np.clip(sino_nepers.astype(np.float64), -0.1, 6.5)
    return gaussian_filter1d(sino_pp, sigma=1.0, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm: FBP (GPU)
# ══════════════════════════════════════════════════════════════════════════════


def fbp_gpu(sino_nepers, angles_rad, device):
    """Fan-beam FBP with Feldkamp pre-weight and Hann-filtered ramp, GPU."""
    import torch
    import numpy as np

    sino_pp = _preprocess_sino_np(sino_nepers)          # float64, nepers
    sino    = sino_pp / MU_SCALE                        # float64, pixel-density
    n_views, n_det = sino.shape

    det_j = (np.arange(n_det) - n_det / 2.0) * DET_SPACING
    cos_beta = D_SSD / np.sqrt(D_SSD ** 2 + det_j ** 2)
    sino_w   = sino * cos_beta[np.newaxis, :]

    n_fft   = int(2 ** np.ceil(np.log2(2 * n_det)))
    freq    = np.fft.rfftfreq(n_fft, d=DET_SPACING)
    ramp    = 2.0 * np.abs(freq)
    hann    = 0.5 + 0.5 * np.cos(np.pi * freq / freq.max())
    kernel  = ramp * hann

    sino_f        = np.fft.rfft(sino_w, n=n_fft, axis=1)
    sino_f       *= kernel[np.newaxis, :]
    sino_filtered = np.fft.irfft(sino_f, n=n_fft, axis=1)[:, :n_det]

    # Backprojection with proper 1/U² Feldkamp weight (not via _gpu_backproject adjoint)
    sino_t = torch.tensor(sino_filtered, device=device, dtype=torch.float32)
    import math
    H = W = IMAGE_SIZE
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    py = (yy - H / 2.0).double()
    px = (xx - W / 2.0).double()

    recon_w = torch.zeros(H, W, device=device, dtype=torch.float64)
    sino_td = sino_t.double()

    for i, angle in enumerate(angles_rad):
        angle = float(angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        sy = -D_SO * sin_a
        sx =  D_SO * cos_a
        vy = py - sy
        vx = px - sx
        U  = vy * sin_a - vx * cos_a
        V  = vy * cos_a + vx * sin_a

        U_safe    = torch.where(U > 1e-6, U, torch.full_like(U, 1e-6))
        det_j_val = D_SSD * V / U_safe
        det_idx   = det_j_val / DET_SPACING + n_det / 2.0

        valid = (U > 0) & (det_idx >= 0) & (det_idx < n_det - 1)
        d0    = det_idx.floor().long().clamp(0, n_det - 2)
        d1    = d0 + 1
        frac  = (det_idx - d0.double())

        row   = sino_td[i]
        val   = (1.0 - frac) * row[d0] + frac * row[d1]
        val   = torch.where(valid, val, torch.zeros_like(val))

        weight = (D_SO / U_safe) ** 2
        recon_w += weight * val

    recon_w *= (math.pi / (2.0 * n_views)) * FBP_CAL
    return recon_w.clamp(0.0, 1.0).float().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm: TV-ADMM (GPU)
# ══════════════════════════════════════════════════════════════════════════════


def _tv_prox_gpu(v, lam, n_iter=30):
    """Chambolle TV proximal on GPU."""
    import torch
    tau = 0.25
    p   = torch.zeros(2, *v.shape, device=v.device, dtype=v.dtype)

    for _ in range(n_iter):
        # primal: z = v - lam * div(p)
        dy = torch.zeros_like(p[0])
        dx = torch.zeros_like(p[1])
        dy[1:, :]  = p[0][1:, :]  - p[0][:-1, :]
        dy[0,  :]  = p[0][0,  :]
        dy[-1, :]  = -p[0][-2, :]
        dx[:, 1:]  = p[1][:, 1:]  - p[1][:, :-1]
        dx[:,  0]  = p[1][:,  0]
        dx[:, -1]  = -p[1][:, -2]
        div_p = dy + dx

        z  = v - lam * div_p
        # gradient of z
        gz_y = torch.zeros_like(z)
        gz_x = torch.zeros_like(z)
        gz_y[:-1, :] = z[1:, :] - z[:-1, :]
        gz_x[:, :-1] = z[:, 1:] - z[:, :-1]

        p_new = p + tau * torch.stack([gz_y, gz_x])
        norm  = torch.clamp(torch.sqrt(p_new[0] ** 2 + p_new[1] ** 2), min=1.0)
        p     = p_new / norm.unsqueeze(0)

    dy = torch.zeros_like(p[0])
    dx = torch.zeros_like(p[1])
    dy[1:, :]  = p[0][1:, :]  - p[0][:-1, :]
    dy[0,  :]  = p[0][0,  :]
    dy[-1, :]  = -p[0][-2, :]
    dx[:, 1:]  = p[1][:, 1:]  - p[1][:, :-1]
    dx[:,  0]  = p[1][:,  0]
    dx[:, -1]  = -p[1][:, -2]
    return v - lam * (dy + dx)


def tv_admm_gpu(sino_nepers, angles_rad, device, n_iter=20, lam=0.006, step=None):
    """TV-ADMM on GPU: min 0.5||A(x)MU - y||² + λ·TV(x)."""
    import torch
    import numpy as np

    y_np = _preprocess_sino_np(sino_nepers)
    y    = torch.tensor(y_np, device=device, dtype=torch.float32)

    if step is None:
        step = _gpu_step_size(angles_rad, device)

    # Init with FBP
    x = torch.tensor(
        fbp_gpu(sino_nepers, angles_rad, device), device=device, dtype=torch.float32
    )

    for k in range(n_iter):
        Ax   = _gpu_project(x, angles_rad, device) * MU_SCALE
        res  = Ax - y
        grad = _gpu_backproject(res * MU_SCALE, angles_rad, device)
        x    = x - step * grad
        x    = _tv_prox_gpu(x, lam * step, n_iter=20)
        x    = x.clamp(0.0, 1.0)

    return x.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm: PnP-ADMM with DRUNet (GPU)
# ══════════════════════════════════════════════════════════════════════════════


def pnp_drunet(sino_nepers, angles_rad, device, denoiser,
               n_iter=15, sigma_start=0.06, sigma_end=0.01, step=None):
    """PnP-ADMM with DRUNet denoiser on GPU.

    denoiser: deepinv DRUNet model already on `device`
    """
    import torch
    import numpy as np

    y_np = _preprocess_sino_np(sino_nepers)
    y    = torch.tensor(y_np, device=device, dtype=torch.float32)

    if step is None:
        step = _gpu_step_size(angles_rad, device)

    x = torch.tensor(
        fbp_gpu(sino_nepers, angles_rad, device), device=device, dtype=torch.float32
    )

    sigma_schedule = torch.linspace(sigma_start, sigma_end, n_iter)

    for k in range(n_iter):
        # Gradient step
        Ax   = _gpu_project(x, angles_rad, device) * MU_SCALE
        res  = Ax - y
        grad = _gpu_backproject(res * MU_SCALE, angles_rad, device)
        x    = x - step * grad
        x    = x.clamp(0.0, 1.0)

        # DRUNet denoising step: input (1, 1, H, W), sigma as scalar
        sigma = float(sigma_schedule[k])
        with torch.no_grad():
            inp = x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
            x   = denoiser(inp, sigma).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    return x.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Metrics (GPU-compatible, computed in NumPy)
# ══════════════════════════════════════════════════════════════════════════════


def _compute_psnr(x_hat, x_true):
    import numpy as np
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _compute_ssim(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(x_hat.astype("float32"), x_true.astype("float32"), data_range=1.0))


def _compute_consistency(x_hat, sino_measured, angles_rad, device):
    import torch
    import numpy as np
    y_norm = float(np.linalg.norm(sino_measured))
    if y_norm < 1e-8:
        return 0.0
    x_t   = torch.tensor(x_hat, device=device, dtype=torch.float32)
    y_hat = _gpu_project(x_t, angles_rad, device).cpu().numpy() * MU_SCALE
    diff  = float(np.linalg.norm(y_hat - sino_measured.astype("float64")))
    return float(1.0 - diff / y_norm)


def _composite_score(psnr, ssim, cons):
    psnr_norm = min(1.0, (psnr - 10.0) / 40.0)
    return 0.4 * psnr_norm + 0.4 * ssim + 0.2 * cons


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",   # T4 GPU — user requirement; do NOT change to A10/A100
    volumes={"/models": vol},
    timeout=7200,
    memory=16384,
)
def run_ct_tier_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    """Process one CT tier on T4 GPU, return list of per-scene metric dicts."""
    import torch
    import h5py
    import json
    import time
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tier}] Device: {device}  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    # Load DRUNet if pnp_drunet requested
    denoiser = None
    if "pnp_drunet" in algos:
        try:
            import deepinv as dinv
            drunet_path = "/models/checkpoint/DRUNet/drunet_deepinv_gray_finetune_26k.pth"
            denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, nb=4).to(device)
            ckpt = torch.load(drunet_path, map_location=device, weights_only=False)
            denoiser.load_state_dict(ckpt)
            denoiser.eval()
            print("[DRUNet] Loaded successfully")
        except Exception as exc:
            print(f"[DRUNet] Failed to load: {exc}")
            denoiser = None

    rows = []
    f = h5py.File(io.BytesIO(h5_bytes), "r")
    scene_keys = sorted(f.keys())

    for sk in scene_keys:
        grp           = f[sk]
        x_true        = grp["x_true"][()].astype(np.float32)
        sino_measured = grp["sinogram_measured"][()].astype(np.float32)
        angles_rad    = grp["angles_nominal"][()].astype(np.float64)
        meta          = json.loads(grp.attrs.get("metadata", "{}"))
        scene_name    = meta.get("scene", sk)
        n_views       = int(len(angles_rad))

        print(f"\n  [{tier}] Scene {sk}  ({scene_name})  n_views={n_views}")

        for algo in algos:
            t0 = time.time()
            try:
                if algo == "fbp_gpu":
                    x_hat = fbp_gpu(sino_measured, angles_rad, device)
                elif algo == "tv_admm_gpu":
                    x_hat = tv_admm_gpu(sino_measured, angles_rad, device)
                elif algo == "pnp_drunet":
                    if denoiser is None:
                        print(f"    [{algo}] SKIP — DRUNet not loaded")
                        continue
                    x_hat = pnp_drunet(sino_measured, angles_rad, device, denoiser)
                else:
                    print(f"    [{algo}] Unknown algorithm, skipping")
                    continue
            except Exception as exc:
                print(f"    [{algo}] ERROR: {exc}")
                import traceback; traceback.print_exc()
                continue

            elapsed = time.time() - t0
            psnr    = _compute_psnr(x_hat, x_true)
            ssim    = _compute_ssim(x_hat, x_true)
            cons    = _compute_consistency(x_hat, sino_measured, angles_rad, device)
            score   = _composite_score(psnr, ssim, cons)

            print(f"    [{algo:12s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                  f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")

            rows.append({
                "tier":        tier,
                "scene":       sk,
                "scene_name":  scene_name,
                "algo":        algo,
                "n_views":     n_views,
                "psnr_db":     round(psnr, 4),
                "ssim":        round(ssim, 4),
                "consistency": round(cons, 4),
                "score":       round(score, 4),
                "time_s":      round(elapsed, 2),
            })

    f.close()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main(
    tier:  str = "all",
    algo:  str = "all",
):
    """Run CT GPU benchmark on Modal T4.

    --tier  public|dev|hidden|all   (default: all)
    --algo  fbp_gpu|tv_admm_gpu|pnp_drunet|all  (default: all)
    """
    import csv
    import json
    import time
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT    = PROJECT_ROOT / "datasets" / "benchmark" / "ct"
    RESULTS_DIR  = PROJECT_ROOT / "results" / "ct"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    ALL_ALGOS = ["fbp_gpu", "tv_admm_gpu", "pnp_drunet"]

    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo  == "all" else [a.strip() for a in algo.split(",")]

    print("CT GPU Benchmark — Modal T4")
    print(f"Tiers: {tiers}  |  Algos: {algos}")

    # Submit all tiers in parallel
    futures = {}
    for t in tiers:
        h5_path = DATA_ROOT / t / f"ct_challenge_{t}.h5"
        if not h5_path.exists():
            print(f"  [SKIP] {h5_path} not found")
            continue
        with open(h5_path, "rb") as fh:
            h5_bytes = fh.read()
        futures[t] = run_ct_tier_gpu.remote(h5_bytes, t, algos)

    # Collect results
    all_rows = []
    for t, future in futures.items():
        rows = future.get()
        all_rows.extend(rows)

    # Save outputs
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"benchmark_gpu_{ts}.json"
    out_csv  = RESULTS_DIR / f"benchmark_gpu_{ts}.csv"

    result_doc = {
        "timestamp": ts,
        "tiers":     tiers,
        "algos":     algos,
        "gpu":       "T4",
        "scenes":    all_rows,
    }
    with open(out_json, "w") as fj:
        json.dump(result_doc, fj, indent=2)
    print(f"\nResults saved → {out_json}")

    if all_rows:
        with open(out_csv, "w", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV saved     → {out_csv}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — mean metrics per (tier, algo)")
    print("=" * 70)
    print(f"{'tier':8s}  {'algo':14s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Cons':>6s}  {'Score':>6s}")
    print("-" * 70)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rows in sorted(acc.items()):
        psnr  = sum(r["psnr_db"]    for r in rows) / len(rows)
        ssim  = sum(r["ssim"]        for r in rows) / len(rows)
        cons  = sum(r["consistency"] for r in rows) / len(rows)
        score = sum(r["score"]       for r in rows) / len(rows)
        print(f"{t:8s}  {a:14s}  {psnr:7.2f}  {ssim:6.4f}  {cons:6.4f}  {score:6.4f}")
    print("=" * 70)
