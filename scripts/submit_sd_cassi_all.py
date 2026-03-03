#!/usr/bin/env python3
"""Multi-algorithm SD-CASSI benchmark submission via Modal GPU.

Runs 5 reconstruction algorithms on all SD-CASSI challenge tiers with
proper mismatch correction:
  - GAP-TV: iterative optimization (resolution-agnostic)
  - MST-L / MST-S: Mask-aware Spectral Transformer (tiled for 500x500)
  - HDNet: dual-domain learning (tiled for 500x500)
  - PnP-CASSI: GAP + HSI-SDeCNN denoiser (resolution-agnostic)

Usage:
  modal run scripts/submit_sd_cassi_all.py
  modal run scripts/submit_sd_cassi_all.py --tier public --algorithm gap_tv
  modal run scripts/submit_sd_cassi_all.py --tier all --algorithm all
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("pwm-sd-cassi-all")
vol = modal.Volume.from_name("pwm-models")

_LOCAL_ROOT = Path(__file__).resolve().parent.parent  # Physics_World_Model
_PWM_CORE = _LOCAL_ROOT / "packages" / "pwm_core"
_REF_CASSI = _LOCAL_ROOT / "reference" / "cassi"
_DATASET_DIR = _LOCAL_ROOT / "datasets" / "benchmark" / "sd_cassi"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy<2",
        "scipy",
        "h5py",
        "scikit-image",
        "einops",
    )
    .add_local_dir(
        str(_PWM_CORE / "pwm_core"),
        "/root/packages/pwm_core",
        ignore=[".git", "__pycache__", ".pytest_cache"],
    )
    .add_local_dir(
        str(_REF_CASSI),
        "/root/reference_cassi",
        ignore=[".git", "__pycache__"],
    )
)

ALL_ALGORITHMS = ["gap_tv", "mst_l", "mst_s", "hdnet", "pnp_cassi"]
ALL_TIERS = ["public", "dev", "hidden"]

CHECKPOINT_PATHS = {
    "mst_l": "/models/checkpoint/MST-HDNet/mst/mst_l.pth",
    "mst_s": "/models/checkpoint/MST-HDNet/mst/mst_s.pth",
    "hdnet": "/models/checkpoint/MST-HDNet/hdnet/hdnet.pth",
    "pnp_cassi": "/models/checkpoint/PnP-CASSI/deep_denoiser.pth",
}


# ── Upload helper ────────────────────────────────────────────────────────────

def _upload_datasets():
    """Upload challenge HDF5 files to Modal volume if not already there."""
    import subprocess

    for tier in ALL_TIERS:
        local = _DATASET_DIR / tier / f"sd_cassi_challenge_{tier}.h5"
        if not local.exists():
            print(f"  {tier}: local file not found, skipping")
            continue
        result = subprocess.run(
            ["modal", "volume", "ls", "pwm-models", "sd_cassi/"],
            capture_output=True, text=True,
        )
        if f"sd_cassi_challenge_{tier}.h5" in result.stdout:
            print(f"  {tier}: already on volume, skipping")
            continue
        print(f"  Uploading {local.name} ...")
        subprocess.run(
            ["modal", "volume", "put", "pwm-models", str(local),
             f"sd_cassi/sd_cassi_challenge_{tier}.h5"],
            check=True,
        )


# ── GPU Function ─────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": vol},
    timeout=7200,
    memory=32768,
)
def run_sd_cassi_all(tier: str = "all", algorithm: str = "all"):
    """Run multi-algorithm SD-CASSI reconstruction on GPU."""
    import json
    import time

    import h5py
    import numpy as np
    import torch

    sys.path.insert(0, "/root/packages")
    sys.path.insert(0, "/root")

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    tiers = ALL_TIERS if tier == "all" else [tier]
    algorithms = ALL_ALGORITHMS if algorithm == "all" else [algorithm]

    # ── Algorithm implementations ────────────────────────────────────────

    def _load_model_mst(variant, H, nC=28, step=2):
        from pwm_core.recon.mst import create_mst
        model = create_mst(
            variant=variant, in_channels=nC, out_channels=nC,
            base_resolution=H, step=step,
        ).to(device)
        ckpt = torch.load(
            CHECKPOINT_PATHS[variant], map_location=device, weights_only=False
        )
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        else:
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    def _load_model_hdnet(nC=28):
        from pwm_core.recon.hdnet import HDNet as HDNetModel
        model = HDNetModel(dim=64, n_blocks=4, nC=nC).to(device)
        ckpt = torch.load(
            CHECKPOINT_PATHS["hdnet"], map_location=device, weights_only=False
        )
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        else:
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    def _load_pnp_denoiser():
        """Load the original PnP-CASSI HSI-SDeCNN (in_nc=7, 7-band sliding window)."""
        sys.path.insert(0, "/root/reference_cassi")
        from hsi import HSI_SDeCNN
        model = HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15)
        ckpt_path = CHECKPOINT_PATHS["pnp_cassi"]
        if os.path.exists(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=False)
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(device)
        return model

    # ── Core reconstruction functions ────────────────────────────────────

    # Nominal dispersion step (all CASSI algorithms use integer step=2)
    NOMINAL_STEP = 2

    def _crop_measurement(y, mask, nC=28, step=None):
        """Crop/pad measurement to (H, W + (nC-1)*step).

        Challenge measurements are allocated with width W+(L-1)*ceil(slope)
        which is wider than needed (ceil(2.02)=3 → 81 extra cols vs 54).
        We crop to the nominal step=2 format that all algorithms expect.
        """
        if step is None:
            step = NOMINAL_STEP
        H, W = mask.shape
        W_target = W + (nC - 1) * step
        y_adj = np.zeros((H, W_target), dtype=np.float32)
        h_copy = min(y.shape[0], H)
        w_copy = min(y.shape[1], W_target)
        y_adj[:h_copy, :w_copy] = y[:h_copy, :w_copy].astype(np.float32)
        return y_adj

    def recon_gap_tv(y, mask, nC=28, iters=100, lam=0.02):
        """GAP-TV reconstruction. Resolution-agnostic."""
        from pwm_core.recon.gap_tv import gap_tv_cassi
        y_adj = _crop_measurement(y, mask, nC)
        result = gap_tv_cassi(
            y_adj, mask.astype(np.float32),
            n_bands=nC, iterations=iters, lam=lam,
            step=NOMINAL_STEP, accelerate=True, device="cuda:0",
        )
        return np.clip(result, 0, 1).astype(np.float32)

    def recon_mst(y, mask, model, nC=28, step=2):
        """MST reconstruction for a single 256x256 tile."""
        from pwm_core.recon.mst import shift_torch, shift_back_meas_torch
        H, W = mask.shape
        y_adj = _crop_measurement(y, mask, nC, step=step)

        mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
        mask_t = (
            torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
            .unsqueeze(0).float().to(device)
        )
        mask_shift = shift_torch(mask_t, step=step)
        meas_t = torch.from_numpy(y_adj.copy()).unsqueeze(0).float().to(device)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC) / nC * 2
        with torch.no_grad():
            recon = model(x_init, mask_shift)
        return np.clip(recon.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)

    def recon_hdnet(y, mask, model, nC=28, step=2):
        """HDNet reconstruction for a single 256x256 tile."""
        from pwm_core.recon.mst import shift_back_meas_torch
        H, W = mask.shape
        y_adj = _crop_measurement(y, mask, nC, step=step)

        mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
        mask_3d_t = (
            torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
            .unsqueeze(0).float().to(device)
        )
        meas_t = torch.from_numpy(y_adj.copy()).unsqueeze(0).float().to(device)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC) / nC * 2
        model_input = torch.cat([x_init, mask_3d_t], dim=1)
        with torch.no_grad():
            recon = model(model_input)
        return np.clip(recon.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)

    def recon_pnp_cassi(y, mask, denoiser_model, nC=28, iters=50):
        """GAP + HSI-SDeCNN (original 7-band sliding window PnP)."""
        H, W = mask.shape
        step = NOMINAL_STEP
        y_f = _crop_measurement(y, mask, nC)
        W_meas = W + (nC - 1) * step
        mask_f = mask.astype(np.float32)

        # Build 3D mask (H, W, nC) for CASSI forward/adjoint
        Phi = np.zeros((H, W, nC), dtype=np.float32)
        for k in range(nC):
            Phi[:, :, k] = mask_f

        # Forward/adjoint operators
        def A_op(x, Phi):
            """CASSI forward: x (H,W,nC) -> y (H, W_meas)"""
            y_out = np.zeros((H, W_meas), dtype=np.float32)
            for k in range(nC):
                y_out[:, k * step:k * step + W] += Phi[:, :, k] * x[:, :, k]
            return y_out

        def At_op(y_in, Phi):
            """CASSI adjoint: y (H, W_meas) -> x (H,W,nC)"""
            x_out = np.zeros((H, W, nC), dtype=np.float32)
            for k in range(nC):
                x_out[:, :, k] = Phi[:, :, k] * y_in[:, k * step:k * step + W]
            return x_out

        # Measurement-space normalization: sum of mask^2 at each pixel
        Phi_sum_meas = np.zeros((H, W_meas), dtype=np.float32)
        for k in range(nC):
            Phi_sum_meas[:, k * step:k * step + W] += mask_f ** 2
        Phi_sum_meas[Phi_sum_meas == 0] = 1

        # Initialize via back-projection
        x = At_op(y_f, Phi)

        # GAP iterations with HSI-SDeCNN denoising
        y1 = np.zeros_like(y_f)
        sigma_val = 10.0  # noise level for denoiser

        for it in range(iters):
            yb = A_op(x, Phi)
            y1 = y1 + (y_f - yb)
            x = x + At_op((y1 - yb) / Phi_sum_meas, Phi)

            # HSI-SDeCNN denoising: 7-band sliding window per band
            denoised = np.zeros_like(x)
            for i in range(nC):
                # Build 7-band context window with boundary padding
                bands = []
                for j in range(i - 3, i + 4):
                    if j < 0:
                        bands.append(x[:, :, 0])
                    elif j >= nC:
                        bands.append(x[:, :, nC - 1])
                    else:
                        bands.append(x[:, :, j])
                net_input = np.stack(bands, axis=0)  # (7, H, W)
                net_input = torch.from_numpy(
                    np.ascontiguousarray(net_input)
                ).float().unsqueeze(0).to(device)
                Nsigma = torch.full(
                    (1, 1, 1, 1), sigma_val / 255.0
                ).type_as(net_input)
                with torch.no_grad():
                    output = denoiser_model(net_input, Nsigma)
                denoised[:, :, i] = output.squeeze().cpu().numpy()

            x = denoised
            x = np.maximum(x, 0)

        return np.clip(x, 0, 1).astype(np.float32)

    # ── Tiling for 500x500 with 256x256-trained models ───────────────────

    def _tiled_reconstruct(y, mask, recon_fn, nC=28, step=2,
                           tile_size=256, overlap=32):
        """Tile-based reconstruction with cosine-weighted blending.

        For models trained on 256x256, applies overlapping tiles on larger
        images and blends with cosine weights to avoid boundary artifacts.
        """
        H, W = mask.shape
        if H <= tile_size and W <= tile_size:
            return recon_fn(y, mask)

        stride = tile_size - 2 * overlap
        n_tiles_h = max(1, int(np.ceil((H - 2 * overlap) / stride)))
        n_tiles_w = max(1, int(np.ceil((W - 2 * overlap) / stride)))

        # Output accumulator
        result = np.zeros((H, W, nC), dtype=np.float32)
        weight_map = np.zeros((H, W, 1), dtype=np.float32)

        # Build 2D cosine blend weights
        blend_1d = np.ones(tile_size, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, np.pi / 2, overlap)
            cos_ramp = np.sin(ramp) ** 2
            blend_1d[:overlap] = cos_ramp
            blend_1d[-overlap:] = cos_ramp[::-1]
        blend_2d = blend_1d[:, None] * blend_1d[None, :]  # (tile, tile)

        for ti in range(n_tiles_h):
            for tj in range(n_tiles_w):
                # Tile start/end in object space
                r0 = min(ti * stride, H - tile_size)
                c0 = min(tj * stride, W - tile_size)
                r0 = max(r0, 0)
                c0 = max(c0, 0)
                r1 = r0 + tile_size
                c1 = c0 + tile_size

                # Extract mask tile
                mask_tile = mask[r0:r1, c0:c1]

                # Extract measurement tile
                # In CASSI with step=2, band k's measurement for object
                # column c is at measurement column c + k*step.
                # So for object tile [c0:c0+tile_size], the full
                # measurement span is [c0, c0 + tile_size + (nC-1)*step).
                meas_w = tile_size + (nC - 1) * step
                y_tile = np.zeros((tile_size, meas_w), dtype=y.dtype)
                src_c0 = c0
                src_c1 = min(c0 + meas_w, y.shape[1])
                copy_w = src_c1 - src_c0
                if copy_w > 0:
                    y_tile[:, :copy_w] = y[r0:r1, src_c0:src_c1]

                # Reconstruct tile
                x_tile = recon_fn(y_tile, mask_tile)

                # Blend into output
                actual_h = min(tile_size, H - r0)
                actual_w = min(tile_size, W - c0)
                bw = blend_2d[:actual_h, :actual_w, np.newaxis]
                result[r0:r0 + actual_h, c0:c0 + actual_w, :] += \
                    x_tile[:actual_h, :actual_w, :] * bw
                weight_map[r0:r0 + actual_h, c0:c0 + actual_w, :] += bw

        weight_map = np.maximum(weight_map, 1e-8)
        return (result / weight_map).astype(np.float32)

    # ── Mismatch correction ──────────────────────────────────────────────

    def estimate_mismatch(y_np, mask_np, spec_ranges, H, W, nC=28):
        """3-stage mismatch correction: grid search + gradient + dispersion.

        For 500x500 inputs, runs correction at 2x downsampled resolution
        and scales parameters back.
        """
        from pwm_core.calibration.cassi_torch_modules import (
            DifferentiableMaskWarpFixed,
            DifferentiableCassiForwardSTE,
            DifferentiableGAPTV,
        )

        # Parse spec_ranges
        bounds = {}
        for sr in spec_ranges:
            bounds[sr["name"]] = (sr["min"], sr["max"])

        # Downsample for large images
        do_downsample = H > 300
        if do_downsample:
            factor = 2
            y_ds = y_np[::factor, ::factor].astype(np.float64)
            mask_ds = mask_np[::factor, ::factor].astype(np.float64)
            H_ds, W_ds = mask_ds.shape
        else:
            y_ds, mask_ds = y_np, mask_np
            H_ds, W_ds = H, W
            factor = 1

        s_nom = np.arange(nC, dtype=np.float64) * 2.0

        # Scale dispersion for downsampled grid
        if do_downsample:
            s_nom_ds = s_nom / factor
        else:
            s_nom_ds = s_nom

        y_t = torch.from_numpy(y_ds.copy()).unsqueeze(0).float().to(device)

        # ── Stage 0: Coarse grid search ──────────────────────────────────
        print("    Stage 0: coarse grid search ...", end=" ", flush=True)
        t0 = time.time()

        dx_range = np.linspace(bounds["mask_dx"][0], bounds["mask_dx"][1], 7)
        dy_range = np.linspace(bounds["mask_dy"][0], bounds["mask_dy"][1], 7)
        theta_range = np.linspace(
            bounds["mask_rotation"][0], bounds["mask_rotation"][1], 5
        )

        # Scale translations for downsampled grid
        if do_downsample:
            dx_range_ds = dx_range / factor
            dy_range_ds = dy_range / factor
        else:
            dx_range_ds = dx_range
            dy_range_ds = dy_range

        gaptv = DifferentiableGAPTV(
            s_nom_ds, H_ds, W_ds, nC, n_iter=8,
            gauss_sigma=0.5, use_checkpointing=False,
        ).to(device)
        fwd_op = DifferentiableCassiForwardSTE(s_nom_ds).to(device)

        best_loss = float("inf")
        dx_init = (bounds["mask_dx"][0] + bounds["mask_dx"][1]) / 2
        dy_init = (bounds["mask_dy"][0] + bounds["mask_dy"][1]) / 2
        theta_init = (bounds["mask_rotation"][0] + bounds["mask_rotation"][1]) / 2
        best_dx_ds = dx_init / factor if do_downsample else dx_init
        best_dy_ds = dy_init / factor if do_downsample else dy_init
        best_theta = theta_init

        with torch.no_grad():
            for dx_c in dx_range_ds:
                for dy_c in dy_range_ds:
                    for th_c in theta_range:
                        warp = DifferentiableMaskWarpFixed(
                            mask_ds, dx_c, dy_c, th_c
                        ).to(device)
                        mask_w = warp()
                        phi_d = torch.tensor(0.0, device=device)
                        x_rec = gaptv(y_t, mask_w, phi_d)
                        y_pred = fwd_op(x_rec, mask_w, phi_d)
                        hh = min(y_t.shape[1], y_pred.shape[1])
                        ww = min(y_t.shape[2], y_pred.shape[2])
                        loss = torch.mean(
                            (y_t[:, :hh, :ww] - y_pred[:, :hh, :ww]) ** 2
                        ).item()
                        if loss < best_loss:
                            best_loss = loss
                            best_dx_ds = dx_c
                            best_dy_ds = dy_c
                            best_theta = th_c

        print(f"{time.time() - t0:.1f}s  loss={best_loss:.6f}")

        # ── Stage 1: Gradient refinement ─────────────────────────────────
        print("    Stage 1: gradient refinement ...", end=" ", flush=True)
        t1 = time.time()

        gaptv_fine = DifferentiableGAPTV(
            s_nom_ds, H_ds, W_ds, nC, n_iter=12,
            gauss_sigma=0.7, use_checkpointing=True,
        ).to(device)
        gaptv_fine.train()

        warp = DifferentiableMaskWarpFixed(
            mask_ds, best_dx_ds, best_dy_ds, best_theta
        ).to(device)

        optimizer = torch.optim.Adam([
            {"params": [warp.dx], "lr": 0.02},
            {"params": [warp.dy], "lr": 0.02},
            {"params": [warp.theta_deg], "lr": 0.005},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=80, eta_min=0.001
        )

        # Bounds in downsampled space
        dx_lo = bounds["mask_dx"][0] / factor if do_downsample else bounds["mask_dx"][0]
        dx_hi = bounds["mask_dx"][1] / factor if do_downsample else bounds["mask_dx"][1]
        dy_lo = bounds["mask_dy"][0] / factor if do_downsample else bounds["mask_dy"][0]
        dy_hi = bounds["mask_dy"][1] / factor if do_downsample else bounds["mask_dy"][1]

        for step_i in range(80):
            optimizer.zero_grad()
            mask_w = warp()
            phi_d = torch.tensor(0.0, device=device, requires_grad=False)
            x_rec = gaptv_fine(y_t, mask_w, phi_d)
            y_pred = fwd_op(x_rec, mask_w, phi_d)
            hh = min(y_t.shape[1], y_pred.shape[1])
            ww = min(y_t.shape[2], y_pred.shape[2])
            loss = torch.mean(
                (y_t[:, :hh, :ww] - y_pred[:, :hh, :ww]) ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [warp.dx, warp.dy, warp.theta_deg], 0.5
            )
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                warp.dx.clamp_(dx_lo, dx_hi)
                warp.dy.clamp_(dy_lo, dy_hi)
                warp.theta_deg.clamp_(
                    bounds["mask_rotation"][0], bounds["mask_rotation"][1]
                )

        # Scale translations back to full resolution
        final_dx = warp.dx.item() * factor if do_downsample else warp.dx.item()
        final_dy = warp.dy.item() * factor if do_downsample else warp.dy.item()
        final_theta = warp.theta_deg.item()
        print(f"{time.time() - t1:.1f}s  dx={final_dx:.4f} dy={final_dy:.4f} "
              f"theta={final_theta:.4f}")

        # ── Stage 2: Dispersion grid search ──────────────────────────────
        print("    Stage 2: dispersion grid search ...", end=" ", flush=True)
        t2 = time.time()

        slope_range = np.linspace(
            bounds["dispersion_slope"][0], bounds["dispersion_slope"][1], 9
        )
        axis_range = np.linspace(
            bounds["dispersion_axis"][0], bounds["dispersion_axis"][1], 7
        )

        best_slope = 2.0
        best_axis = 0.0
        best_disp_loss = float("inf")

        # Use final corrected mask for dispersion search
        warp_final_ds = DifferentiableMaskWarpFixed(
            mask_ds,
            final_dx / factor if do_downsample else final_dx,
            final_dy / factor if do_downsample else final_dy,
            final_theta,
        ).to(device)

        with torch.no_grad():
            mask_w = warp_final_ds()
            for slope_c in slope_range:
                for axis_c in axis_range:
                    s_trial = np.arange(nC, dtype=np.float64) * slope_c
                    if do_downsample:
                        s_trial = s_trial / factor
                    fwd_trial = DifferentiableCassiForwardSTE(s_trial).to(device)
                    phi_trial = torch.tensor(axis_c, device=device)

                    gaptv_d = DifferentiableGAPTV(
                        s_trial, H_ds, W_ds, nC, n_iter=8,
                        gauss_sigma=0.5, use_checkpointing=False,
                    ).to(device)

                    x_rec = gaptv_d(y_t, mask_w, phi_trial)
                    y_pred = fwd_trial(x_rec, mask_w, phi_trial)
                    hh = min(y_t.shape[1], y_pred.shape[1])
                    ww = min(y_t.shape[2], y_pred.shape[2])
                    loss_v = torch.mean(
                        (y_t[:, :hh, :ww] - y_pred[:, :hh, :ww]) ** 2
                    ).item()
                    if loss_v < best_disp_loss:
                        best_disp_loss = loss_v
                        best_slope = slope_c
                        best_axis = axis_c

        print(f"{time.time() - t2:.1f}s  slope={best_slope:.4f} "
              f"axis={best_axis:.4f}")

        # Clean up GPU memory
        del gaptv, gaptv_fine, fwd_op, warp, warp_final_ds
        torch.cuda.empty_cache()

        return {
            "mask_dx": round(final_dx, 4),
            "mask_dy": round(final_dy, 4),
            "mask_rotation": round(final_theta, 4),
            "dispersion_slope": round(best_slope, 4),
            "dispersion_axis": round(best_axis, 4),
        }

    # ── Warp mask using corrected parameters ─────────────────────────────

    def warp_mask_corrected(mask_np, corrected):
        """Warp mask using estimated mismatch parameters."""
        from scipy.ndimage import affine_transform
        dx = corrected["mask_dx"]
        dy = corrected["mask_dy"]
        theta = corrected["mask_rotation"]

        H, W = mask_np.shape
        theta_rad = np.deg2rad(theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

        matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        center = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
        offset = center - matrix @ (center + np.array([dy, dx]))

        return affine_transform(
            mask_np, matrix, offset=offset,
            order=1, mode='constant', cval=0.0,
        ).astype(np.float32)

    # ── Run a single algorithm on a single sample ────────────────────────

    def run_algorithm(algo_name, y_np, mask_np, corrected,
                      models_cache, nC=28, H=256, W=256):
        """Run one algorithm on one sample, handling tiling if needed."""
        needs_tiling = H > 256 and algo_name in ("mst_l", "mst_s", "hdnet")
        step = NOMINAL_STEP

        # For physics-based methods, use corrected mask
        # Try both corrected and ideal masks — corrected for physics-based
        if algo_name in ("gap_tv", "pnp_cassi"):
            mask_corrected = warp_mask_corrected(mask_np, corrected)
            mask_use = mask_corrected
        else:
            mask_use = mask_np.astype(np.float32)

        if algo_name == "gap_tv":
            # Debug: print measurement/mask info
            y_adj = _crop_measurement(y_np, mask_use, nC)
            print(f"    y_raw={y_np.shape} y_crop={y_adj.shape} "
                  f"mask={mask_use.shape} step={NOMINAL_STEP}")
            # Run with more iterations and acceleration for better convergence
            return recon_gap_tv(
                y_np, mask_use, nC=nC,
                iters=100, lam=0.02,
            )

        elif algo_name in ("mst_l", "mst_s"):
            model = models_cache.get(f"{algo_name}_{H}")
            if model is None:
                base_res = 256 if needs_tiling else H
                model = _load_model_mst(algo_name, base_res, nC, step)
                models_cache[f"{algo_name}_{H}"] = model

            def _recon_mst_tile(y_tile, mask_tile):
                return recon_mst(y_tile, mask_tile, model, nC, step)

            if needs_tiling:
                return _tiled_reconstruct(
                    y_np, mask_use, _recon_mst_tile,
                    nC=nC, step=step, tile_size=256, overlap=32,
                )
            else:
                return recon_mst(y_np, mask_use, model, nC, step)

        elif algo_name == "hdnet":
            model = models_cache.get(f"hdnet_{H}")
            if model is None:
                model = _load_model_hdnet(nC)
                models_cache[f"hdnet_{H}"] = model

            def _recon_hdnet_tile(y_tile, mask_tile):
                return recon_hdnet(y_tile, mask_tile, model, nC, step)

            if needs_tiling:
                return _tiled_reconstruct(
                    y_np, mask_use, _recon_hdnet_tile,
                    nC=nC, step=step, tile_size=256, overlap=32,
                )
            else:
                return recon_hdnet(y_np, mask_use, model, nC, step)

        elif algo_name == "pnp_cassi":
            denoiser = models_cache.get("pnp_denoiser")
            if denoiser is None:
                denoiser = _load_pnp_denoiser()
                models_cache["pnp_denoiser"] = denoiser
            return recon_pnp_cassi(
                y_np, mask_use, denoiser, nC=nC, iters=50,
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    # ── Process all tiers and algorithms ─────────────────────────────────

    os.makedirs("/models/sd_cassi/submissions", exist_ok=True)
    all_results = {}
    models_cache = {}

    for tier_name in tiers:
        h5_path = f"/models/sd_cassi/sd_cassi_challenge_{tier_name}.h5"
        if not os.path.exists(h5_path):
            print(f"\nSkipping {tier_name}: {h5_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {tier_name.upper()} tier")
        print(f"{'=' * 60}")

        with h5py.File(h5_path, "r") as fin:
            sample_keys = sorted(
                [k for k in fin.keys() if k.startswith("sample_")]
            )
            n_samples = len(sample_keys)

            s0 = fin[sample_keys[0]]
            mask_0 = s0["H_ideal"][:]
            H, W = mask_0.shape
            nC = 28
            spec_ranges = json.loads(s0.attrs["spec_ranges"])

            print(f"  {n_samples} samples, {H}x{W}, algorithms: {algorithms}")

            # ── Step 1: Mismatch correction for all samples ──────────────
            print(f"\n  --- Mismatch Correction ---")
            corrections = {}
            for idx, key in enumerate(sample_keys):
                grp = fin[key]
                y_np = grp["y"][:].astype(np.float64)
                mask_np = grp["H_ideal"][:].astype(np.float64)

                print(f"\n  [{idx:02d}/{n_samples}] {key} correction")
                t_start = time.time()
                corrected = estimate_mismatch(
                    y_np, mask_np, spec_ranges, H, W, nC
                )
                corrections[key] = corrected
                print(f"    {corrected}  ({time.time() - t_start:.1f}s)")

            # ── Step 2: Reconstruct with each algorithm ──────────────────
            for algo_name in algorithms:
                print(f"\n  --- {algo_name.upper()} ---")
                out_path = (
                    f"/models/sd_cassi/submissions/"
                    f"sd_cassi_submission_{algo_name}_{tier_name}.h5"
                )
                algo_results = []

                with h5py.File(out_path, "w") as fout:
                    fout.attrs["variant"] = "sd_cassi"
                    fout.attrs["tier"] = tier_name
                    fout.attrs["algorithm"] = algo_name
                    fout.attrs["submission_type"] = "reconstruction"

                    for idx, key in enumerate(sample_keys):
                        grp = fin[key]
                        y_np = grp["y"][:].astype(np.float64)
                        mask_np = grp["H_ideal"][:].astype(np.float64)
                        corrected = corrections[key]

                        print(f"  [{idx:02d}/{n_samples}] {key} ...",
                              end=" ", flush=True)
                        t_start = time.time()

                        try:
                            x_hat = run_algorithm(
                                algo_name, y_np, mask_np, corrected,
                                models_cache, nC=nC, H=H, W=W,
                            )
                        except Exception as e:
                            print(f"FAILED: {e}")
                            x_hat = np.zeros((H, W, nC), dtype=np.float32)

                        elapsed = time.time() - t_start

                        # Evaluate PSNR if ground truth available
                        psnr_val = None
                        if "x_true" in grp:
                            x_true = grp["x_true"][:].astype(np.float64)
                            mse = np.mean((x_hat - x_true) ** 2)
                            psnr_val = (
                                10 * np.log10(1.0 / mse) if mse > 0 else 100
                            )

                        psnr_str = f"PSNR={psnr_val:.2f}dB" if psnr_val else ""
                        print(f"{elapsed:.1f}s  "
                              f"[{x_hat.min():.3f},{x_hat.max():.3f}] "
                              f"{psnr_str}")

                        # Write submission
                        grp_out = fout.create_group(key)
                        grp_out.create_dataset(
                            "x_hat",
                            data=x_hat.astype(np.float32),
                            compression="gzip",
                            compression_opts=4,
                        )
                        grp_out.attrs["corrected_spec"] = json.dumps(
                            corrected
                        )

                        algo_results.append({
                            "sample": key,
                            "psnr": psnr_val,
                            "time_s": round(elapsed, 1),
                        })

                all_results[f"{algo_name}_{tier_name}"] = algo_results
                print(f"  Saved: {out_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for key, results in all_results.items():
        psnrs = [r["psnr"] for r in results if r["psnr"] is not None]
        times = [r["time_s"] for r in results]
        print(f"\n{key}:")
        if psnrs:
            print(f"  Mean PSNR: {np.mean(psnrs):.2f} dB")
            print(f"  Min/Max:   {np.min(psnrs):.2f} / {np.max(psnrs):.2f} dB")
        print(f"  Mean time: {np.mean(times):.1f}s/sample")
        print(f"  Total:     {np.sum(times):.1f}s")

    # Save summary JSON
    summary_path = "/models/sd_cassi/submissions/summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    return all_results


# ── Entrypoint ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    tier: str = "all",
    algorithm: str = "all",
):
    import time

    print("Uploading challenge datasets to Modal volume ...")
    _upload_datasets()

    print(f"\nLaunching GPU job: tier={tier}, algorithm={algorithm}")
    t0 = time.time()
    results = run_sd_cassi_all.remote(tier=tier, algorithm=algorithm)
    elapsed = time.time() - t0

    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Download submission files
    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)

    tiers = ALL_TIERS if tier == "all" else [tier]
    algos = ALL_ALGORITHMS if algorithm == "all" else [algorithm]

    for t in tiers:
        for a in algos:
            remote = f"sd_cassi/submissions/sd_cassi_submission_{a}_{t}.h5"
            local = out_dir / f"sd_cassi_submission_{a}_{t}.h5"
            try:
                import subprocess
                subprocess.run(
                    ["modal", "volume", "get", "pwm-models", remote, str(local)],
                    check=True, capture_output=True,
                )
                print(f"  Downloaded: {local}")
            except Exception:
                print(f"  Not found: {remote}")

    # Download summary
    try:
        import subprocess
        subprocess.run(
            ["modal", "volume", "get", "pwm-models",
             "sd_cassi/submissions/summary.json",
             str(out_dir / "summary.json")],
            check=True, capture_output=True,
        )
        print(f"  Downloaded: {out_dir / 'summary.json'}")
    except Exception:
        pass

    print("\nDone! Submission files in ./submissions/")
