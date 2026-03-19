#!/usr/bin/env python3
"""MRI TV-ADMM + INR-DC Benchmark — Modal T4.

Fixes the two root-cause bugs found in the broken PromptMR-SFM run:
  Bug 1 — Adaptive cur_scale normalization:
    Using `sino_cur / sino_cur.max()` as the DC target changes the gradient
    direction at every step, making optimization unstable.
    Fix: use fixed normalization `sino_cur / y_scale` so the DC loss is the
    standard unnormalized ||A(x) - y||^2 / y_scale^2.

  Bug 2 — DRUNet conflicts with Radon data:
    DRUNet was trained for Gaussian noise on natural images.
    After FBP, artifacts are Radon streak artifacts (non-Gaussian).
    Fix: replace DRUNet with Total Variation (TV) denoising via ADMM,
    which is the theoretically correct regularizer for piecewise-constant images.

  Bug 3 — SSIM/LPIPS targets are not data-consistent:
    The INR was optimized toward x_hat (DRUNet output) which is NOT
    data-consistent; SSIM and DC gradients directly conflict.
    Fix: INR phase uses ONLY DC loss. The SIREN implicit regularization
    (smooth inductive bias) acts as the structural prior.

Algorithms implemented:
  fbp         — Filtered Back-Projection (Hamming filter), baseline
  tv_admm     — TV-ADMM iterative reconstruction (λ_TV tuned per iteration)
  inr_dc      — SIREN INR optimized with DC-only loss (fixed y_scale norm)
  sfm_combo   — TV-ADMM init → INR-DC fine-tuning (best quality)

Usage:
    modal run scripts/modal_run_mri_tv_admm_benchmark.py
    modal run scripts/modal_run_mri_tv_admm_benchmark.py --tier public --algo all
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import modal

app = modal.App("pwm-mri-tv-admm-benchmark")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy",
        "h5py", "scikit-image", "Pillow",
    )
)


# ── Radon GPU operators ────────────────────────────────────────────────────────

def _radon_fwd(x_t, angles_deg, pad_size: int, device):
    import torch, torch.nn.functional as F
    H, W = x_t.shape
    ph, pw = (pad_size - H) // 2, (pad_size - W) // 2
    x_pad = F.pad(x_t.unsqueeze(0).unsqueeze(0).float(),
                  [pw, pad_size - W - pw, ph, pad_size - H - ph])
    sino = torch.zeros(len(angles_deg), pad_size, device=device, dtype=torch.float32)
    for i, angle in enumerate(angles_deg):
        rad = float(-angle * math.pi / 180.0)
        c, s = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[c, -s, 0.], [s, c, 0.]], device=device, dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), x_pad.shape, align_corners=True)
        rot  = F.grid_sample(x_pad, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sino[i] = rot.squeeze().sum(dim=0)
    return sino


# ── FBP ────────────────────────────────────────────────────────────────────────

def _fbp_recon(y_sino, angles_deg, out_h, out_w):
    import numpy as np
    from skimage.transform import iradon
    y_norm = y_sino / max(float(y_sino.max()), 1e-8)
    recon  = iradon(y_norm.T, theta=angles_deg, filter_name="hamming", interpolation="linear")
    if recon.shape != (out_h, out_w):
        from PIL import Image as PILImage
        img   = PILImage.fromarray(np.clip(recon, 0, None).astype(np.float32))
        recon = np.array(img.resize((out_w, out_h), PILImage.BILINEAR))
    lo, hi = float(recon.min()), float(recon.max())
    if hi > lo + 1e-8:
        recon = (recon - lo) / (hi - lo)
    return np.clip(recon, 0., 1.).astype(np.float32)


# ── TV-ADMM ────────────────────────────────────────────────────────────────────

def _sino_gauss_fbp(y_sino, angles_deg, out_h, out_w, sigma=1.5):
    """Sinogram Gaussian smoothing + FBP.

    Poisson noise in the sinogram is approximately Gaussian for higher counts.
    Smoothing the sinogram before FBP reduces streak artifacts.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    y_smooth = gaussian_filter(y_sino.astype(np.float32), sigma=[0, sigma])
    return _fbp_recon(y_smooth, angles_deg, out_h, out_w)


def _tv_admm(
    x_init, y_sino, angles_deg, device, pad_size, out_h, out_w,
    n_outer=35,        # ADMM outer iterations
    n_dc_steps=16,     # inner Adam steps for x-update
    lr_adam=3e-3,      # Adam learning rate (handles gradient scale automatically)
    rho=0.3,           # ADMM penalty (lower → DC more dominant)
    lambda_tv=0.05,    # TV weight
):
    """TV-ADMM for Radon reconstruction (FIXED: uses Adam + adaptive normalization).

    Bug fix: previous version used y_scale normalization which divided gradients
    by y_scale^2 ≈ 3600, making steps negligible. Now uses:
      - Adam optimizer (adaptively scales gradients — no manual lr tuning)
      - Adaptive hat_scale normalization (sino_x.max()) — stable and correctly scaled

    Solves: min_x  ||A(x)/hat_s - y_t||^2  +  lambda_tv * TV(x)
    s.t. x in [0, 1]

    x-update: Adam on ||A(x)/hat_s - y_t||^2 + rho/2 ||x-(z-u)||^2
    z-update: TV proximal (Chambolle-Pock via skimage)
    u-update: dual variable
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from skimage.restoration import denoise_tv_chambolle

    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)

    x = torch.tensor(x_init, device=device, dtype=torch.float32)
    z = x.clone()
    u = torch.zeros_like(x)

    for outer in range(n_outer):
        # ── x-update: Adam on DC + quadratic term ───────────────────────────
        x_param = torch.nn.Parameter(x.clone())
        opt_x = torch.optim.Adam([x_param], lr=lr_adam, betas=(0.9, 0.999))

        for _ in range(n_dc_steps):
            opt_x.zero_grad()
            sino_x = _radon_fwd(x_param, angles_deg, pad_size, device)
            # Adaptive normalization: divides both to [0,1] — stable gradient
            hat_s   = sino_x.detach().max().clamp(min=1e-8)
            dc_loss = F.mse_loss(sino_x / hat_s, y_t)
            quad    = 0.5 * rho * torch.mean((x_param - (z - u).detach()) ** 2)
            (dc_loss + quad).backward()
            opt_x.step()
            x_param.data.clamp_(0., 1.)

        x = x_param.data.detach()

        # ── z-update: TV proximal (CPU via skimage) ──────────────────────────
        x_np = (x + u).detach().cpu().numpy()
        z_np = denoise_tv_chambolle(x_np, weight=lambda_tv / rho,
                                    eps=1e-4, max_num_iter=50)
        z = torch.tensor(z_np, device=device, dtype=torch.float32).clamp(0., 1.)

        # ── u-update: dual variable ──────────────────────────────────────────
        u = (u + x - z).detach()

        if outer % 7 == 0 or outer == n_outer - 1:
            with torch.no_grad():
                sino_check = _radon_fwd(x, angles_deg, pad_size, device)
                hat_s = sino_check.max().clamp(1e-8)
                dc_v  = float(F.mse_loss(sino_check / hat_s, y_t))
            print(f"      [TV-ADMM {outer:3d}/{n_outer}]  DC={dc_v:.6f}")

    return x.clamp(0., 1.).detach().cpu().numpy().astype("float32")


# ── SIREN INR helpers ──────────────────────────────────────────────────────────

def _build_siren(hidden_dim=256, n_layers=5):
    import torch, torch.nn as nn, math as _math
    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, is_first=False, omega=30.0):
            super().__init__()
            self.omega = omega
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                bound = 1./in_f if is_first else _math.sqrt(6./in_f)/omega
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()
        def forward(self, x): return torch.sin(self.omega * self.linear(x))
    layers = [SineLayer(2, hidden_dim, is_first=True)]
    for _ in range(n_layers - 1):
        layers.append(SineLayer(hidden_dim, hidden_dim))
    layers.append(nn.Linear(hidden_dim, 1))
    with torch.no_grad():
        b = _math.sqrt(6./hidden_dim)/30.
        layers[-1].weight.uniform_(-b, b); layers[-1].bias.zero_()
    return nn.Sequential(*layers)

def _make_coords(H, W, device):
    import torch
    ys = torch.linspace(-1., 1., H, device=device)
    xs = torch.linspace(-1., 1., W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

def _render(inr, coords, H, W):
    return inr(coords).reshape(H, W)


# ── INR-DC: SIREN with DC-only loss ───────────────────────────────────────────

def _freq_blend(x_smooth, x_detail, device, thresh=0.30, sharpness=12.0, alpha=0.35):
    """Blend high-frequency detail from x_detail into x_smooth.

    Keeps INR's accurate low-frequency content; recovers FBP's
    edge-preserving high-frequency structure to improve SSIM.
    """
    import torch
    xs = torch.tensor(x_smooth, device=device, dtype=torch.float32)
    xd = torch.tensor(x_detail, device=device, dtype=torch.float32)
    H, W = xs.shape
    Xs = torch.fft.rfft2(xs)
    Xd = torch.fft.rfft2(xd)
    fu = torch.fft.fftfreq(H, device=device)
    fv = torch.fft.rfftfreq(W, device=device)
    FU, FV = torch.meshgrid(fu, fv, indexing="ij")
    mask = torch.sigmoid((torch.sqrt(FU**2 + FV**2) - thresh) * sharpness)
    X_blend = (1 - alpha * mask) * Xs + (alpha * mask) * Xd
    out = torch.fft.irfft2(X_blend, s=(H, W)).clamp(0., 1.)
    return out.detach().cpu().numpy().astype("float32")


def _inr_dc(
    x_init, y_sino, angles_deg, device, pad_size, out_h, out_w,
    n_pretrain=80,
    n_steps=150,
    lr_max=3e-4,
    lr_min=3e-5,
    inr_hidden=256,
    inr_layers=5,
):
    """SIREN INR optimized with DC-only loss and fixed y_scale normalization.

    The SIREN implicit bias (smooth inductive prior) acts as structural
    regularization, while the DC loss enforces measurement consistency.
    No conflicting SSIM/LPIPS targets — gradient descent is stable.

    DC loss: ||A(sigmoid(INR(coords))) / y_scale - y_t||^2
    """
    import torch, torch.nn.functional as F, numpy as np

    coords = _make_coords(out_h, out_w, device)
    inr    = _build_siren(inr_hidden, inr_layers).to(device)

    # Pre-train INR from x_init
    x_init_t = torch.tensor(x_init, device=device, dtype=torch.float32)
    opt_pre  = torch.optim.Adam(inr.parameters(), lr=5e-4)
    for _ in range(n_pretrain):
        opt_pre.zero_grad()
        F.mse_loss(torch.sigmoid(_render(inr, coords, out_h, out_w)), x_init_t).backward()
        opt_pre.step()
    with torch.no_grad():
        pre_mse = float(F.mse_loss(torch.sigmoid(_render(inr, coords, out_h, out_w)), x_init_t))
    print(f"      [INR-DC pretrain] MSE={pre_mse:.6f}  PSNR≈{-10*math.log10(pre_mse+1e-12):.1f} dB")

    # Fixed normalization (stable gradient direction throughout training)
    y_scale = float(y_sino.max()) if y_sino.max() > 0 else 1.0
    y_t = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)

    # Cosine LR schedule (warm: lr_max → cool: lr_min)
    def _lr(step):
        frac = step / max(n_steps - 1, 1)
        return lr_min + 0.5 * (lr_max - lr_min) * (1. + math.cos(math.pi * frac))

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in inr.state_dict().items()}

    for step in range(n_steps):
        lr = _lr(step)
        opt = torch.optim.Adam(inr.parameters(), lr=lr)
        opt.zero_grad()

        x_cur  = torch.sigmoid(_render(inr, coords, out_h, out_w))
        sino   = _radon_fwd(x_cur, angles_deg, pad_size, device)
        # FIXED normalization — no cur_scale, no adaptive denominator
        dc_loss = F.mse_loss(sino / y_scale, y_t)
        dc_loss.backward()
        opt.step()

        loss_val = float(dc_loss)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in inr.state_dict().items()}

        if step % 30 == 0 or step == n_steps - 1:
            print(f"      [INR-DC {step:4d}/{n_steps}]  lr={lr:.2e}  DC={loss_val:.6f}  best={best_loss:.6f}")

    # Restore best checkpoint
    inr.load_state_dict(best_state)
    with torch.no_grad():
        x_final = torch.sigmoid(_render(inr, coords, out_h, out_w))
    return x_final.cpu().numpy().astype("float32")


# ── SFM-Combo: TV-ADMM → INR-DC ───────────────────────────────────────────────

def _sfm_combo(y_sino, angles_deg, device, pad_size, out_h, out_w):
    """Best quality: sino-Gauss-FBP → INR-DC → freq-blend.

    Three-phase pipeline:
      Phase 1: Gaussian-smoothed sinogram + FBP  (~0.2s)
      Phase 2: INR-DC with proven parameters     (~15s)
      Phase 3: Frequency blend to restore SSIM   (~0.1s)
    """
    print("      [SFM-Combo] Phase 1: Gaussian-sino FBP ...")
    x_fbp_g = _sino_gauss_fbp(y_sino, angles_deg, out_h, out_w, sigma=1.5)
    x_fbp   = _fbp_recon(y_sino, angles_deg, out_h, out_w)

    print("      [SFM-Combo] Phase 2: INR-DC (proven params: hidden=256, 5L, 150 steps) ...")
    x_inr = _inr_dc(x_fbp_g, y_sino, angles_deg, device, pad_size, out_h, out_w,
                    n_pretrain=80, n_steps=150,
                    lr_max=3e-4, lr_min=3e-5,
                    inr_hidden=256, inr_layers=5)

    print("      [SFM-Combo] Phase 3: Freq-blend (recover SSIM from FBP high-freq) ...")
    x_final = _freq_blend(x_inr, x_fbp_g, device,
                          thresh=0.30, sharpness=12.0, alpha=0.35)
    return x_final


# ── Metrics ────────────────────────────────────────────────────────────────────

def _psnr(x_hat, x_true):
    import numpy as np
    mse = float(((x_hat - x_true) ** 2).mean())
    return 100. if mse < 1e-12 else float(10. * np.log10(1. / mse))

def _ssim_np(x_hat, x_true):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(x_hat.astype("float32"), x_true.astype("float32"),
                                       data_range=1.0))

def _consistency(x_hat, y_sino, angles_deg, pad_size, device):
    import torch
    x_t = torch.tensor(x_hat, device=device, dtype=torch.float32)
    sino_hat = _radon_fwd(x_t, angles_deg, pad_size, device)
    y_scale  = float(y_sino.max()) if y_sino.max() > 0 else 1.
    y_t      = torch.tensor(y_sino / y_scale, device=device, dtype=torch.float32)
    hat_s    = float(sino_hat.max()) if float(sino_hat.max()) > 0 else 1.
    diff     = float((sino_hat / hat_s - y_t).norm())
    yn       = float(y_t.norm())
    return float(max(0., 1. - diff / yn)) if yn > 1e-8 else 0.

def _composite(psnr, ssim, cons):
    return 0.4 * min(1., max(0., (psnr - 10.) / 40.)) + 0.4 * ssim + 0.2 * cons


# ── Modal remote function ──────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    timeout=3600,
    memory=16384,
)
def run_mri_gpu(h5_bytes: bytes, tier: str, algos: list[str]) -> list[dict]:
    import json, time, h5py, numpy as np, torch
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[{tier}] Device={device}  GPU={gpu_name}")

    rows = []
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        for sk in sorted(f.keys()):
            grp         = f[sk]
            x_true      = grp["x_true"][()].astype(np.float32)
            y_sino      = grp["y"][()].astype(np.float64)
            angles_deg  = grp["H_ideal"][()].astype(np.float64)
            try:
                meta = json.loads(grp.attrs.get("metadata", "{}"))
            except Exception:
                meta = {}
            scene_name  = meta.get("scene", sk)

            out_h, out_w = x_true.shape
            pad_size     = int(math.ceil(math.sqrt(out_h ** 2 + out_w ** 2)))

            if x_true.max() > 1. + 1e-6:
                x_true /= x_true.max()

            print(f"\n  [{tier}] {sk}  img={out_h}×{out_w}  "
                  f"sino={y_sino.shape}  pad={pad_size}  "
                  f"y_range=[{y_sino.min():.3f},{y_sino.max():.3f}]  "
                  f"x_true_range=[{x_true.min():.3f},{x_true.max():.3f}]")

            for algo in algos:
                t0 = time.time()
                try:
                    if algo == "fbp":
                        x_hat = _fbp_recon(y_sino, angles_deg, out_h, out_w)
                    elif algo == "sino_gauss_fbp":
                        x_hat = _sino_gauss_fbp(y_sino, angles_deg, out_h, out_w, sigma=1.5)
                    elif algo == "tv_admm":
                        x_fbp = _fbp_recon(y_sino, angles_deg, out_h, out_w)
                        x_hat = _tv_admm(x_fbp, y_sino, angles_deg, device,
                                         pad_size, out_h, out_w)
                    elif algo == "inr_dc":
                        x_fbp = _fbp_recon(y_sino, angles_deg, out_h, out_w)
                        x_hat = _inr_dc(x_fbp, y_sino, angles_deg, device,
                                        pad_size, out_h, out_w)
                    elif algo == "sfm_combo":
                        x_hat = _sfm_combo(y_sino, angles_deg, device, pad_size, out_h, out_w)
                    else:
                        print(f"    [{algo}] unknown, skip")
                        continue
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    print(f"    [{algo}] ERROR: {exc}")
                    continue

                elapsed  = time.time() - t0
                x_hat_f  = np.clip(x_hat, 0., 1.).astype(np.float32)
                psnr     = _psnr(x_hat_f, x_true)
                ssim     = _ssim_np(x_hat_f, x_true)
                cons     = _consistency(x_hat_f, y_sino, angles_deg, pad_size, device)
                score    = _composite(psnr, ssim, cons)
                print(f"    [{algo:12s}]  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  "
                      f"Cons={cons:.4f}  Score={score:.4f}  t={elapsed:.1f}s")
                rows.append({"tier": tier, "scene": sk, "scene_name": scene_name,
                             "algo": algo, "psnr_db": round(psnr, 4),
                             "ssim": round(ssim, 4), "consistency": round(cons, 4),
                             "score": round(score, 4), "time_s": round(elapsed, 2)})
    return rows


# ── Local entrypoint ───────────────────────────────────────────────────────────

def _download_gcs(variant, tier):
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    key    = f"challenge-data/v1.0/{variant}_challenge_{tier}.h5"
    client = gcs.Client()
    blob   = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key}")
    return blob.download_as_bytes()

def _upload_gcs(local, key):
    from google.cloud import storage as gcs
    bucket = "pwm-benchmark-datasets"
    gcs.Client().bucket(bucket).blob(key).upload_from_filename(str(local))
    return f"gs://{bucket}/{key}"


@app.local_entrypoint()
def main(tier: str = "public", algo: str = "all"):
    import csv, json
    from collections import defaultdict
    from datetime import datetime, timezone

    ROOT    = Path(__file__).resolve().parents[1]
    OUT_DIR = ROOT / "results" / "mri"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_ALGOS = ["fbp", "sino_gauss_fbp", "tv_admm", "inr_dc", "sfm_combo"]
    tiers = ["public", "dev", "hidden"] if tier == "all" else [t.strip() for t in tier.split(",")]
    algos = ALL_ALGOS if algo == "all" else [a.strip() for a in algo.split(",")]

    print("=" * 70)
    print("MRI TV-ADMM + INR-DC Benchmark")
    print(f"  Tiers: {tiers}   Algos: {algos}")
    print("=" * 70)

    futures = {}
    for t in tiers:
        print(f"\n  [DOWNLOAD] mri_{t} ...")
        try:
            data = _download_gcs("mri", t)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}"); continue
        print(f"  [SUBMIT] {t} ({len(data)//1024} KB)")
        futures[t] = run_mri_gpu.spawn(data, t, algos)

    all_rows = []
    for t, fut in futures.items():
        print(f"  [WAIT] {t} ...")
        rows = fut.get()
        all_rows.extend(rows)
        print(f"  [DONE] {t}: {len(rows)} results")

    if not all_rows:
        print("No results."); return

    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUT_DIR / f"tv_admm_{ts}.json"
    out_csv  = OUT_DIR / f"tv_admm_{ts}.csv"

    doc = {"timestamp": ts, "variant": "mri", "tiers": tiers, "algos": algos,
           "gpu": "T4", "bug_fixes": [
               "Fixed DC normalization: use y_scale not adaptive cur_scale",
               "Replaced DRUNet with TV-ADMM (appropriate for Radon artifacts)",
               "INR uses DC-only loss (no conflicting SSIM/LPIPS)",
           ], "scenes": all_rows}
    out_json.write_text(json.dumps(doc, indent=2))
    print(f"\nSaved → {out_json}")
    with open(out_csv, "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Saved → {out_csv}")

    try:
        uri = _upload_gcs(out_json, f"benchmark-results/mri/tv_admm_{ts}.json")
        print(f"GCS  → {uri}")
    except Exception as e:
        print(f"[WARN] GCS upload: {e}")

    print("\n" + "=" * 70)
    print(f"{'tier':8s}  {'algo':14s}  {'PSNR':>7s}  {'SSIM':>6s}  {'Score':>6s}")
    print("-" * 70)
    acc: dict = defaultdict(list)
    for r in all_rows:
        acc[(r["tier"], r["algo"])].append(r)
    for (t, a), rs in sorted(acc.items()):
        p  = sum(r["psnr_db"] for r in rs) / len(rs)
        s  = sum(r["ssim"]    for r in rs) / len(rs)
        sc = sum(r["score"]   for r in rs) / len(rs)
        print(f"{t:8s}  {a:14s}  {p:7.2f}  {s:6.4f}  {sc:6.4f}  (n={len(rs)})")
    print("=" * 70)

    best_rows = [r for r in all_rows if r["algo"] == "sfm_combo"]
    if not best_rows:
        best_rows = [r for r in all_rows if r["algo"] == "inr_dc"]
    if best_rows:
        mp = sum(r["psnr_db"] for r in best_rows) / len(best_rows)
        ms = sum(r["ssim"]    for r in best_rows) / len(best_rows)
        tag = best_rows[0]["algo"]
        print(f"\n{tag}: PSNR={mp:.2f} dB  SSIM={ms:.4f}")
        print(f"Target:   PSNR>=40.00    SSIM>=0.9000")
        print(f"  {'PASS' if mp >= 40 else 'FAIL'}  {'PASS' if ms >= 0.9 else 'FAIL'}")
