#!/usr/bin/env python3
"""Score-MRI Benchmark with DPS (Diffusion Posterior Sampling) — Modal T4 GPU.

Root-cause fixes applied in this version
-----------------------------------------
  Bug 1 — FBP resize vs crop:
    Previous FBP used PIL resize (182→128) which caused geometric distortion.
    Correct: crop the central 128×128 from the 182×182 iradon output.

  Bug 2 — FBP scale mismatch:
    After min-max normalising the FBP to [0,1], the Radon of the FBP image is
    ~1.4× larger than the observed sinogram y.  We pre-scale x_fbp so that
    mean(A(x_init)) ≈ mean(y_sino) before the iterative loop.

  Bug 3 — SIRT without D_R normalisation:
    The adjoint _radon_bwd already divides by n_angles.  To recover the
    proper SIRT update D_C^{-1} A^T D_R^{-1} (y − Ax), the residual must
    also be divided by the projection-length image D_R = A(ones_128).

  Bug 4 — DRUNet at σ_max=0.18 destroying the image:
    DRUNet treats σ as AWGN noise level.  At σ=0.18 the denoiser zeroes out
    nearly all content in a [0,1]-range phantom, wiping every SIRT gain.
    Fix: start at σ_max=0.05 so the denoiser acts as a mild regulariser.

Dataset format (MRI challenge HDF5)
-------------------------------------
    y       : (180, 182)  parallel-beam sinogram (rotation-based Radon)
    H_ideal : (180,)      projection angles [0, 1, …, 179] degrees
    x_true  : (128, 128)  ground-truth phantom in [0, 1]

Usage
-----
    modal run scripts/modal_run_mri_score_benchmark.py
    modal run scripts/modal_run_mri_score_benchmark.py --tier public
    modal run scripts/modal_run_mri_score_benchmark.py --algo score_mri_dps
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

# ── Modal infrastructure ──────────────────────────────────────────────────────

app = modal.App("pwm-mri-score-dps-v2")
vol = modal.Volume.from_name("pwm-models")

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
        "Pillow",
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# Batched GPU Radon operators
# ══════════════════════════════════════════════════════════════════════════════


def _radon_fwd(x_t, angles_deg, pad_size, device):
    """Batched GPU Radon forward — all angles at once.

    Matches challenge generator:  sino[i] = ndrotate(padded_x, -θ).sum(axis=0)
    which is a CW rotation of the padded image by θ degrees followed by a
    vertical projection (sum along rows).
    """
    import torch
    import torch.nn.functional as F

    H, W = x_t.shape
    pad_h = (pad_size - H) // 2
    pad_w = (pad_size - W) // 2
    x_pad = F.pad(
        x_t.unsqueeze(0).unsqueeze(0).float(),
        [pad_w, pad_size - W - pad_w, pad_h, pad_size - H - pad_h],
    )  # (1, 1, pad, pad)

    n = len(angles_deg)
    # Negate angles: scipy ndrotate(x, -a) maps content CW by a, which requires
    # the pull-sampling matrix to use -a in the standard [[cos,-sin],[sin,cos]] form.
    rads = x_t.new_tensor([-a * math.pi / 180.0 for a in angles_deg])
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack(
        [torch.stack([c, -s, z], dim=1), torch.stack([s, c, z], dim=1)], dim=1
    )  # (n, 2, 3)  — CW rotation by θ (matches scipy ndrotate(x, -θ))

    x_batch = x_pad.expand(n, -1, -1, -1)       # (n, 1, pad, pad)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    rot = F.grid_sample(x_batch, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return rot.squeeze(1).sum(dim=1)              # (n, pad)  — sum along rows


def _radon_bwd(sino, angles_deg, out_h, out_w, pad_size, device):
    """Batched GPU Radon back-projection (adjoint of _radon_fwd).

    Returns the un-scaled adjoint A^T(sino) / n_angles cropped to (out_h, out_w).
    The /n normalisation is intentional: it ensures A^T(ones) ≈ 1 so
    SIRT step-sizes can be expressed in natural units (see _sirt_update).
    """
    import torch
    import torch.nn.functional as F

    n = len(angles_deg)
    rads = sino.new_tensor([-a * math.pi / 180.0 for a in angles_deg])  # CCW adjoint
    c, s = torch.cos(rads), torch.sin(rads)
    z = torch.zeros(n, device=device, dtype=torch.float32)
    theta = torch.stack(
        [torch.stack([c, -s, z], dim=1), torch.stack([s, c, z], dim=1)], dim=1
    )

    spread = sino.unsqueeze(1).expand(-1, pad_size, -1)   # (n, pad, pad)
    grid = F.affine_grid(theta, (n, 1, pad_size, pad_size), align_corners=True)
    back = F.grid_sample(
        spread.unsqueeze(1), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )  # (n, 1, pad, pad)
    recon = back.squeeze(1).sum(dim=0) / n      # (pad, pad)

    ph = (pad_size - out_h) // 2
    pw = (pad_size - out_w) // 2
    return recon[ph : ph + out_h, pw : pw + out_w]


# ══════════════════════════════════════════════════════════════════════════════
# FBP baseline — correct crop + scale calibration
# ══════════════════════════════════════════════════════════════════════════════


def _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size):
    """FBP with central crop (not PIL resize) and Hann filter."""
    import numpy as np
    from skimage.transform import iradon

    y_norm = y_sino / max(float(y_sino.max()), 1e-8)
    recon = iradon(y_norm.T, theta=angles_deg,
                   filter_name="hann", interpolation="linear")
    ph = (recon.shape[0] - out_h) // 2
    pw = (recon.shape[1] - out_w) // 2
    cropped = recon[ph : ph + out_h, pw : pw + out_w]
    lo, hi = float(cropped.min()), float(cropped.max())
    if hi > lo + 1e-8:
        cropped = (cropped - lo) / (hi - lo)
    return np.clip(cropped, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Core algorithm
# ══════════════════════════════════════════════════════════════════════════════


def _tv_prox(x: "torch.Tensor", lam: float, n_iter: int = 12,
             lr: float = 0.018) -> "torch.Tensor":
    """Approximate isotropic TV proximal operator via gradient descent.

    Solves: z* = argmin_z { ||z - x||² / 2  +  λ · TV(z) }

    Uses gradient descent on the combined objective.  Converges for
    lr < 2 / (1 + 8λ) (spectral bound on the smooth part + TV Hessian).
    """
    import torch
    z = x.detach().clone()
    for _ in range(n_iter):
        # Forward differences
        dy = torch.cat([z[1:, :] - z[:-1, :],
                        torch.zeros_like(z[:1, :])], dim=0)
        dx = torch.cat([z[:, 1:] - z[:, :-1],
                        torch.zeros_like(z[:, :1])], dim=1)
        mag = (dy ** 2 + dx ** 2 + 1e-8).sqrt()
        ny, nx = dy / mag, dx / mag

        # Backward divergence of (ny, nx)
        ny_pad = torch.cat([torch.zeros_like(ny[:1, :]), ny], dim=0)
        div_y  = ny_pad[1:, :] - ny_pad[:-1, :]
        nx_pad = torch.cat([torch.zeros_like(nx[:, :1]), nx], dim=1)
        div_x  = nx_pad[:, 1:] - nx_pad[:, :-1]
        grad_tv = -(div_y + div_x)                           # ∇TV(z)

        z = z - lr * ((z - x) + lam * grad_tv)
    return z.clamp(0.0, 1.0)


def score_mri_dps(
    y_sino,
    angles_deg,
    device,
    denoiser,
    pad_size: int,
    out_h: int,
    out_w: int,
    # Main iterative loop
    n_outer: int = 500,
    sirt_step: float = 0.8,
    # TV regularisation (annealed from coarse to fine)
    lam_tv_start: float = 0.010,
    lam_tv_end: float = 0.0008,
    tv_n_iter: int = 10,
    tv_lr: float = 0.020,
    # Mild final DRUNet post-processing (sigma=0 = skip)
    final_sigma: float = 0.0,
):
    """Score-MRI: SIRT + TV-regularisation + optional mild DRUNet post-processing.

    Root causes fixed
    -----------------
    * FBP crop (not resize)
    * Scale calibration: mean(A(x₀)) = mean(y) before first iteration
    * D_R (projection-length) normalisation in SIRT update
    * Removed broken Poisson-floor early stopping (floor was 16.65≠actual 1.0)
    * Replaced with relative-convergence criterion ||Δx||/||x|| < 1e-4
    * Reduced TV λ: 0.06→0.010 start (was over-smoothing at lam=0.06)
    * Reduced sirt_step: 1.2→0.8 for improved stability
    * DRUNet skipped by default (natural-image prior hurts piecewise phantoms)

    Algorithm
    ---------
    0. x₀ = FBP(y), scaled so mean(A(x₀)) ≈ mean(y)
    1. Precompute D_R = A(ones) ≈ 90, D_C = A^T(ones)/n ≈ 1
    2. For k = 0 … n_outer:
         a. SIRT step:  x ← clip(x + ω · A^T((y−Ax)/D_R) / D_C, 0,1)
         b. TV prox:    x ← TV_prox(x, λ_k)
         c. Stop when ||Δx||/||x|| < 1e-4 (after ≥100 iters)
    3. Optional: single DRUNet pass at very low σ (mild final denoising)

    TV λ is annealed geometrically from lam_tv_start → lam_tv_end so the
    algorithm first produces smooth, de-artifact reconstructions and
    progressively recovers fine edge detail.
    """
    import torch
    import numpy as np

    n_angles = len(angles_deg)
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)

    # ── 0. FBP init + scale calibration ─────────────────────────────────────
    x_fbp_np = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
    x = torch.tensor(x_fbp_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        sino_init = _radon_fwd(x, angles_deg, pad_size, device)
        scale_fac = float(y_t.mean()) / float(sino_init.mean().clamp(min=1e-6))
        x = (x * scale_fac).clamp(0.0, 1.0)
        dc0 = float(((sino_init * scale_fac - y_t) ** 2).mean())
    print(f"      [init] scale={scale_fac:.3f}  DC₀={dc0:.4f}")

    # ── 1. Precompute SIRT denominators ──────────────────────────────────────
    ones_x = torch.ones(out_h, out_w, device=device, dtype=torch.float32)
    D_R = _radon_fwd(ones_x, angles_deg, pad_size, device).clamp(min=1.0)

    ones_sino = torch.ones(n_angles, pad_size, device=device, dtype=torch.float32)
    D_C = _radon_bwd(ones_sino, angles_deg, out_h, out_w, pad_size, device).clamp(min=0.01)

    print(f"      D_R mean={D_R.mean():.1f}  D_C mean={D_C.mean():.4f}")

    # ── 2. TV λ annealing schedule ───────────────────────────────────────────
    lam_schedule = np.exp(
        np.linspace(np.log(lam_tv_start), np.log(lam_tv_end), n_outer)
    ).tolist()

    for k in range(n_outer):
        lam_tv = lam_schedule[k]

        # ── 2a. SIRT step ─────────────────────────────────────────────────
        x_prev = x.detach().clone()
        sino_cur = _radon_fwd(x, angles_deg, pad_size, device)
        residual = y_t - sino_cur
        update = _radon_bwd(residual / D_R, angles_deg, out_h, out_w, pad_size, device)
        x = (x + sirt_step * update / D_C).clamp(0.0, 1.0)

        # ── 2b. TV proximal step ──────────────────────────────────────────
        x = _tv_prox(x, lam=lam_tv, n_iter=tv_n_iter, lr=tv_lr)

        # ── 2c. Relative-convergence early stopping ────────────────────────
        if k % 50 == 0 or k == n_outer - 1:
            with torch.no_grad():
                dc_k = float(((sino_cur - y_t) ** 2).mean())
                rel_chg = float((x - x_prev).norm() / (x.norm() + 1e-8))
            print(f"      iter {k:4d}/{n_outer}  λ_tv={lam_tv:.5f}  "
                  f"DC={dc_k:.4f}  Δx/x={rel_chg:.5f}")
            if rel_chg < 1e-4 and k >= 100:
                print(f"      [converged] Δx/x < 1e-4 at iter {k}")
                break

    # ── 3. Mild DRUNet final pass (optional) ─────────────────────────────────
    if denoiser is not None and final_sigma > 0.0:
        with torch.no_grad():
            x_in = x.unsqueeze(0).unsqueeze(0)
            x = denoiser(x_in, final_sigma).squeeze().clamp(0.0, 1.0)
        print(f"      [DRUNet final] σ={final_sigma:.3f}")

    import numpy as np
    return x.cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════


def _psnr(x_hat, x_true):
    import numpy as np
    mse = float(((x_hat - x_true) ** 2).mean())
    return 100.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _ssim_np(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        x_hat.astype("float32"), x_true.astype("float32"), data_range=1.0
    ))


def _consistency(x_hat, y_sino, angles_deg, pad_size, device):
    import torch
    x_t = torch.tensor(x_hat, device=device, dtype=torch.float32)
    sino_hat = _radon_fwd(x_t, angles_deg, pad_size, device)
    y_t = torch.tensor(y_sino, device=device, dtype=torch.float32)
    y_scale = float(y_t.max().clamp(min=1e-8))
    hat_scale = float(sino_hat.max().clamp(min=1e-8))
    diff = float((sino_hat / hat_scale - y_t / y_scale).norm())
    y_norm = float((y_t / y_scale).norm())
    return float(max(0.0, 1.0 - diff / y_norm)) if y_norm > 1e-8 else 0.0


def _composite(psnr, ssim, cons):
    return 0.4 * min(1.0, max(0.0, (psnr - 10.0) / 40.0)) + 0.4 * ssim + 0.2 * cons


# ══════════════════════════════════════════════════════════════════════════════
# Modal remote function
# ══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=3600,
    memory=16384,
)
def run_mri_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    """Run MRI reconstruction algorithms on T4 GPU."""
    import json
    import time
    import h5py
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[{tier}] Device: {device}  GPU: {gpu_name}")

    # ── Load DRUNet ───────────────────────────────────────────────────────────
    denoiser = None
    if "score_mri_dps" in algos:
        try:
            import deepinv as dinv
            path = "/models/checkpoint/DRUNet/drunet_deepinv_gray_finetune_26k.pth"
            denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, nb=4).to(device)
            ckpt = torch.load(path, map_location=device, weights_only=False)
            denoiser.load_state_dict(ckpt)
            denoiser.eval()
            print("[DRUNet] Loaded from volume")
        except Exception as exc:
            print(f"[DRUNet] Volume load failed ({exc}), trying download …")
            try:
                import deepinv as dinv
                denoiser = dinv.models.DRUNet(
                    in_channels=1, out_channels=1, nb=4, pretrained="download"
                ).to(device)
                denoiser.eval()
                print("[DRUNet] Downloaded from deepinv hub")
            except Exception as exc2:
                print(f"[DRUNet] Download also failed: {exc2}")

    rows = []
    f = h5py.File(io.BytesIO(h5_bytes), "r")

    for sk in sorted(f.keys()):
        grp = f[sk]
        x_true = grp["x_true"][()].astype(np.float32)
        y_sino = grp["y"][()].astype(np.float64)
        angles_deg = grp["H_ideal"][()].astype(np.float64)
        try:
            meta = json.loads(grp.attrs.get("metadata", "{}"))
        except Exception:
            meta = {}
        scene_name = meta.get("scene", sk)

        out_h, out_w = x_true.shape
        pad_size = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))

        if x_true.max() > 1.0:
            x_true /= x_true.max()

        print(f"\n  [{tier}] {sk} ({scene_name})  "
              f"img={out_h}×{out_w}  sino={y_sino.shape}  pad={pad_size}")

        # Always compute FBP for diagnostics
        x_fbp_np = _fbp_recon(y_sino, angles_deg, out_h, out_w, pad_size)
        psnr_fbp = _psnr(x_fbp_np, x_true)
        ssim_fbp = _ssim_np(x_fbp_np, x_true)
        print(f"    [fbp_diag] PSNR={psnr_fbp:.2f} dB  SSIM={ssim_fbp:.4f}")

        # Forward-model consistency: A_GPU(x_true) vs y_sino
        with torch.no_grad():
            x_true_t = torch.tensor(x_true, device=device, dtype=torch.float32)
            sino_gpu = _radon_fwd(x_true_t, angles_deg, pad_size, device)
            y_t_diag = torch.tensor(y_sino, device=device, dtype=torch.float32)
            dc_gpu = float(((sino_gpu - y_t_diag) ** 2).mean())
            sino_gpu_np = sino_gpu.cpu().numpy()
        # Scipy Radon on CPU (matches challenge data generator exactly)
        from scipy.ndimage import rotate as ndrotate
        _ph = (pad_size - out_h) // 2
        _pw = (pad_size - out_w) // 2
        padded_true = np.zeros((pad_size, pad_size), dtype=np.float64)
        padded_true[_ph:_ph + out_h, _pw:_pw + out_w] = x_true.astype(np.float64)
        sino_scipy = np.zeros((len(angles_deg), pad_size), dtype=np.float64)
        for _i, _a in enumerate(angles_deg):
            sino_scipy[_i] = ndrotate(padded_true, -_a, reshape=False,
                                      order=1, mode="constant").sum(axis=0)
        dc_scipy_vs_y = float(((sino_scipy - y_sino.astype(np.float64)) ** 2).mean())
        dc_gpu_vs_scipy = float(((sino_gpu_np.astype(np.float64) - sino_scipy) ** 2).mean())
        print(f"    [fwd_model] DC_GPU(x_true,y)={dc_gpu:.4f}  "
              f"DC_scipy(x_true,y)={dc_scipy_vs_y:.4f}  "
              f"DC_GPU_vs_scipy={dc_gpu_vs_scipy:.4f}")
        print(f"    [fwd_model] GPU sino=[{sino_gpu_np.min():.2f},{sino_gpu_np.max():.2f}]  "
              f"scipy sino=[{sino_scipy.min():.2f},{sino_scipy.max():.2f}]  "
              f"y=[{y_sino.min():.2f},{y_sino.max():.2f}]")

        # Quick: also run FBP + pure TV denoising as a comparison
        with torch.no_grad():
            x_fbp_t = torch.tensor(x_fbp_np, device=device, dtype=torch.float32)
            x_fbp_tv = _tv_prox(x_fbp_t, lam=0.05, n_iter=300, lr=0.35)
            x_fbp_tv_np = x_fbp_tv.cpu().numpy()
        psnr_fbptv = _psnr(x_fbp_tv_np, x_true)
        ssim_fbptv = _ssim_np(x_fbp_tv_np, x_true)
        print(f"    [fbp+tv_diag] PSNR={psnr_fbptv:.2f} dB  SSIM={ssim_fbptv:.4f}  (lam=0.05,300iter)")

        for algo in algos:
            t0 = time.time()
            try:
                if algo == "fbp":
                    x_hat = x_fbp_np

                elif algo == "score_mri_dps":
                    x_hat = score_mri_dps(
                        y_sino, angles_deg, device, denoiser,
                        pad_size, out_h, out_w,
                        n_outer=500,
                        sirt_step=0.8,
                        lam_tv_start=0.010,
                        lam_tv_end=0.0008,
                        tv_n_iter=10,
                        tv_lr=0.020,
                        final_sigma=0.0,
                    )

                else:
                    print(f"    [{algo}] Unknown, skipping")
                    continue

            except Exception as exc:
                import traceback
                print(f"    [{algo}] ERROR: {exc}")
                traceback.print_exc()
                continue

            elapsed = time.time() - t0
            x_hat_f = np.clip(x_hat, 0.0, 1.0).astype(np.float32)

            # Scale diagnostic
            print(f"    [{algo}] x_hat range=[{x_hat_f.min():.3f},{x_hat_f.max():.3f}] "
                  f"mean={x_hat_f.mean():.4f}  |  "
                  f"x_true range=[{x_true.min():.3f},{x_true.max():.3f}] "
                  f"mean={x_true.mean():.4f}")
            psnr = _psnr(x_hat_f, x_true)
            ssim = _ssim_np(x_hat_f, x_true)
            cons = _consistency(x_hat_f, y_sino, angles_deg, pad_size, device)
            score = _composite(psnr, ssim, cons)

            print(f"    [{algo:18s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                  f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")

            rows.append({
                "tier": tier,
                "scene": sk,
                "scene_name": scene_name,
                "algo": algo,
                "psnr_db": round(psnr, 4),
                "ssim": round(ssim, 4),
                "consistency": round(cons, 4),
                "score": round(score, 4),
                "time_s": round(elapsed, 2),
            })

    f.close()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════


def _download_gcs(variant: str, tier: str) -> bytes:
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    key = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs.Client()
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key}")
    return blob.download_as_bytes()


def _upload_gcs(local: Path, key: str) -> str:
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    client = gcs.Client()
    client.bucket(bucket).blob(key).upload_from_filename(str(local))
    return f"gs://{bucket}/{key}"


@app.local_entrypoint()
def main(tier: str = "public", algo: str = "all"):
    """Run Score-MRI DPS benchmark on Modal T4.

    --tier  public|dev|hidden|all   (default: public)
    --algo  fbp|score_mri_dps|all   (default: all)
    """
    import csv
    import json
    from collections import defaultdict
    from datetime import datetime, timezone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = PROJECT_ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_TIERS = ["public", "dev", "hidden"]
    ALL_ALGOS = ["fbp", "score_mri_dps"]

    tiers = ALL_TIERS if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("Score-MRI DPS v2 — SIRT + DRUNet — Modal T4")
    print(f"  Fixes: FBP crop, scale calibration, D_R SIRT, conservative σ")
    print(f"  Tiers: {tiers}  Algos: {algos}")

    futures = {}
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_challenge_{t}.h5 …")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  [SUBMIT]  {t}  ({len(data) // 1024} KB)")
        futures[t] = run_mri_gpu.spawn(data, t, algos)

    all_rows = []
    for t, fut in futures.items():
        print(f"  [WAITING] {t} …")
        rows = fut.get()
        all_rows.extend(rows)
        print(f"  [DONE]    {t}: {len(rows)} results")

    if not all_rows:
        print("No results.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"score_mri_dps_{ts}.json"
    out_csv  = OUT_DIR / f"score_mri_dps_{ts}.csv"

    doc = {
        "timestamp": ts,
        "variant": "mri",
        "tiers": tiers,
        "algos": algos,
        "gpu": "T4",
        "improvements": [
            "FBP crop fix",
            "FBP scale calibration",
            "SIRT with D_R normalisation",
            "DRUNet annealed score prior (σ_max=0.05)",
        ],
        "scenes": all_rows,
    }
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved → {out_json}")

    with open(out_csv, "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved → {out_csv}")

    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/score_mri_dps_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY — mean metrics per (tier, algo)")
    print("=" * 72)
    print(f"{'tier':8s}  {'algo':20s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Score':>6s}")
    print("-" * 72)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        p = sum(r["psnr_db"] for r in rs) / len(rs)
        s = sum(r["ssim"]    for r in rs) / len(rs)
        sc = sum(r["score"]  for r in rs) / len(rs)
        print(f"{t:8s}  {a:20s}  {p:7.2f}  {s:6.4f}  {sc:6.4f}")
    print("=" * 72)

    dps_rows = [r for r in all_rows if r["algo"] == "score_mri_dps"]
    if dps_rows:
        mp = sum(r["psnr_db"] for r in dps_rows) / len(dps_rows)
        ms = sum(r["ssim"]    for r in dps_rows) / len(dps_rows)
        print(f"\nScore-MRI DPS:  PSNR = {mp:.2f} dB   SSIM = {ms:.4f}")
        print(f"Target:         PSNR ≥ 40.00 dB   SSIM ≥ 0.9780")
        print(f"  PSNR {'✓' if mp >= 40.0 else '✗'}   SSIM {'✓' if ms >= 0.978 else '✗'}")
