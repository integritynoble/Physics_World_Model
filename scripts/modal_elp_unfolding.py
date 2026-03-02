#!/usr/bin/env python3
"""Modal GPU Runner — ELP-Unfolding on all 3 InverseNet Scenarios.

Runs ELP-Unfolding (ECCV 2022, 565M params, ~9 GB VRAM) on a remote A10G GPU
since it cannot fit on a local 6 GB card.

Scenarios:
  I   (Ideal):    clean y from ground truth + nominal mask
  II  (Baseline): corrupted y + nominal mask (no correction)
  III (Oracle):   corrupted y + true_spec correction

Usage:
    modal run scripts/modal_elp_unfolding.py
    modal run scripts/modal_elp_unfolding.py --scenario ideal
    modal run scripts/modal_elp_unfolding.py --samples 5  # quick test
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal
import numpy as np

# ── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("elp-unfolding-benchmark")

# Data volume (has public/dev/hidden datasets)
data_vol = modal.Volume.from_name("cacti-competition-data", create_if_missing=False)

# Model volume (has ELP-Unfolding checkpoint)
model_vol = modal.Volume.from_name("pwm-models", create_if_missing=False)

# Local paths for bundling into container image
_LOCAL_ROOT = Path(__file__).resolve().parent.parent  # Physics_World_Model
_PWM_CORE = _LOCAL_ROOT / "packages" / "pwm_core"
_ELP_REPO = _LOCAL_ROOT / "repos" / "ELP-Unfolding"

# ── Container Image ──────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "scipy",
        "h5py",
        "scikit-image",
        "einops",
        "six",
    )
    .add_local_dir(
        str(_PWM_CORE), "/root/packages/pwm_core",
        ignore=[".git", "__pycache__", ".pytest_cache"],
    )
    .add_local_dir(
        str(_ELP_REPO), "/root/repos/ELP-Unfolding",
        ignore=[".git", "__pycache__", "fig"],
    )
)

# ── Volume mount paths (inside container) ────────────────────────────────────
VOL_DATA = "/vol"           # cacti-competition-data
VOL_MODELS = "/models"      # pwm-models

ELP_CKPT = f"{VOL_MODELS}/checkpoint/ELP-Unfolding/ckptall.pth"
H5_PATH_TPL = f"{VOL_DATA}/data/{{tier}}/cacti_challenge_{{tier}}.h5"


# ── Helpers (run inside the Modal container) ────────────────────────────────

def _setup_paths():
    """Set up Python paths for importing model code inside the container."""
    sys.path.insert(0, "/root/packages/pwm_core")
    sys.path.insert(0, "/root/repos/ELP-Unfolding")


def _load_elp(device_str: str):
    """Load ELP-Unfolding model with pretrained weights."""
    import torch
    from SCI_Modelcollect import SCI_backwardcollect

    dev = torch.device(device_str)
    argdict = {
        "init_channels": 512,
        "pres_channels": 512,
        "init_input": 8,
        "pres_input": 8,
        "priors": 6,
        "iter__number": 8,
    }
    model = SCI_backwardcollect(argdict).to(dev)
    ckpt = torch.load(ELP_CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["color_SCI_backward_dict"], strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ELP-Unfolding loaded: {n_params:,} params on {dev}")
    return model


def elp_reconstruct(y, mask, model, device_str: str):
    """Run ELP-Unfolding inference on a single sample."""
    import torch

    dev = torch.device(device_str)
    nF = mask.shape[2]
    H, W = y.shape[:2]

    # mask: (H,W,T) -> (1,T,H,W)
    mask_t = torch.from_numpy(
        mask.transpose(2, 0, 1).copy()
    ).unsqueeze(0).float().to(dev)
    # meas: (H,W) -> (1,1,H,W)
    meas_t = torch.from_numpy(y.copy()).unsqueeze(0).unsqueeze(0).float().to(dev)
    # initial estimate: ones (matching original training code)
    img_out_ori = torch.ones(1, nF, H, W, device=dev)

    with torch.no_grad():
        x_list, _ = model(mask_t, meas_t, img_out_ori)

    recon = x_list[-1].squeeze(0).clamp(0, 1).cpu().numpy()  # (T, H, W)
    return recon.transpose(1, 2, 0).astype(np.float32)  # -> (H, W, T)


def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply spatial mismatch using the paper's exact affine_transform.

    Matches validate_cacti_inversenet.py:129-146 exactly.
    """
    from scipy.ndimage import affine_transform, gaussian_filter
    H, W, T = mask.shape
    out = np.zeros_like(mask)
    cx, cy = W / 2.0, H / 2.0
    th = np.radians(rot)
    cos_t, sin_t = np.cos(th), np.sin(th)
    for t in range(T):
        mat = np.array([
            [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        frame = affine_transform(mask[:, :, t], inv[:2, :2], offset=inv[:2, 2], cval=0)
        if blur > 0:
            frame = gaussian_filter(frame, sigma=blur)
        out[:, :, t] = frame
    return out.astype(np.float32)


def binarize_mask(mask):
    """Binarize mask for DL methods — they were trained on {0,1} masks."""
    return (mask > 0.5).astype(np.float64)


def compute_psnr(x_true, x_hat, max_val=1.0):
    mse = float(np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 60.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x_true, x_hat):
    from skimage.metrics import structural_similarity as sk_ssim
    T = x_true.shape[2]
    vals = [sk_ssim(x_true[:, :, t], x_hat[:, :, t], data_range=1.0)
            for t in range(T)]
    return float(np.mean(vals))


def compute_consistency(y, x_hat, mask, gain=1.0, offset=0.0):
    y_pred = gain * np.sum(mask * x_hat, axis=2) + offset
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-10:
        return 1.0
    return max(0.0, float(1.0 - np.linalg.norm(y - y_pred) / y_norm))


def compute_composite(psnr, ssim_val, consistency):
    psnr_norm = float(np.clip((psnr - 15) / 30, 0, 1))
    return 0.4 * psnr_norm + 0.4 * ssim_val + 0.2 * consistency


# ── InverseNet paper reference values ────────────────────────────────────────
PAPER_BASELINES = {
    "scenario_ii": {"psnr": 26.50, "ssim": 0.910},
    "scenario_iii": {"psnr": 34.09, "ssim": 0.965},
}


# ============================================================================
# Modal GPU Function — run all 3 InverseNet scenarios
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_DATA: data_vol, VOL_MODELS: model_vol},
    timeout=7200,
    memory=32768,
)
def run_elp_benchmark(
    tier: str = "public",
    max_samples: int = 0,
    scenario: str = "all",
):
    """Run ELP-Unfolding on all 3 InverseNet scenarios.

    Args:
        tier: Dataset tier (public, dev, hidden).
        max_samples: Limit samples (0 = all).
        scenario: Which scenario(s) to run (all, ideal, baseline, oracle).
    """
    import h5py
    import torch

    _setup_paths()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    # Verify checkpoint
    if not os.path.isfile(ELP_CKPT):
        print(f"ERROR: Checkpoint not found at {ELP_CKPT}")
        return {"error": "checkpoint_missing"}
    print(f"Checkpoint: {ELP_CKPT} ({os.path.getsize(ELP_CKPT) / 1e9:.2f} GB)")

    # Load model
    print("\nLoading ELP-Unfolding model...")
    t_load = time.time()
    model = _load_elp(device)
    print(f"  Model loaded in {time.time() - t_load:.1f}s")

    # Load data
    h5_path = H5_PATH_TPL.format(tier=tier)
    print(f"\nLoading data: {h5_path}")

    samples = []
    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(f.keys())
        if max_samples > 0:
            sample_keys = sample_keys[:max_samples]
        for sk in sample_keys:
            grp = f[sk]
            samples.append({
                "key": sk,
                "y": grp["y"][:].astype(np.float64),
                "mask": grp["H_ideal"][:].astype(np.float64),
                "x_true": grp["x_true"][:].astype(np.float64) if "x_true" in grp else None,
                "true_spec": json.loads(grp.attrs["true_spec"]) if "true_spec" in grp.attrs else None,
                "metadata": json.loads(grp.attrs.get("metadata", "{}")),
            })
    print(f"  Loaded {len(samples)} samples (shape: {samples[0]['mask'].shape})")

    # Determine which scenarios to run
    run_scenarios = ["ideal", "baseline", "oracle"] if scenario == "all" else [scenario]
    all_results = {}

    # ── Scenario I (Ideal): clean y, nominal mask ─────────────────────
    if "ideal" in run_scenarios:
        print(f"\n{'='*78}")
        print(f"  Scenario I  (Ideal) — ELP-Unfolding")
        print(f"{'='*78}")

        scores = []
        t_total = time.time()
        for i, s in enumerate(samples):
            y_clean = np.sum(s["mask"] * s["x_true"], axis=2).astype(np.float32)
            mask_bin = binarize_mask(s["mask"]).astype(np.float32)

            t0 = time.time()
            x_hat = elp_reconstruct(y_clean, mask_bin, model, device)
            dt = time.time() - t0

            psnr = compute_psnr(s["x_true"], x_hat)
            ssim_val = compute_ssim(s["x_true"], x_hat)
            cons = compute_consistency(y_clean, x_hat, s["mask"])
            comp = compute_composite(psnr, ssim_val, cons)

            scene = s["metadata"].get("scene", s["key"])
            print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
                  f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
                  f"Score={comp:.4f}  ({dt:.1f}s)")
            scores.append({"psnr": psnr, "ssim": ssim_val,
                           "consistency": cons, "composite": comp})

        dt_total = time.time() - t_total
        avg = {k: float(np.mean([s[k] for s in scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        avg["total_time"] = dt_total
        all_results["ideal"] = avg
        print(f"\n    AVG: PSNR={avg['psnr']:.2f} dB  SSIM={avg['ssim']:.4f}  "
              f"Cons={avg['consistency']:.4f}  Score={avg['composite']:.4f}  "
              f"({dt_total:.1f}s total)")

    # ── Scenario II (Baseline): noisy y + nominal mask ────────────────
    if "baseline" in run_scenarios:
        print(f"\n{'='*78}")
        print(f"  Scenario II (Baseline) — ELP-Unfolding")
        print(f"{'='*78}")

        scores = []
        t_total = time.time()
        for i, s in enumerate(samples):
            mask_bin = binarize_mask(s["mask"]).astype(np.float32)

            t0 = time.time()
            x_hat = elp_reconstruct(
                s["y"].astype(np.float32), mask_bin, model, device,
            )
            dt = time.time() - t0

            psnr = compute_psnr(s["x_true"], x_hat)
            ssim_val = compute_ssim(s["x_true"], x_hat)
            cons = compute_consistency(s["y"], x_hat, s["mask"], 1.0, 0.0)
            comp = compute_composite(psnr, ssim_val, cons)

            scene = s["metadata"].get("scene", s["key"])
            print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
                  f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
                  f"Score={comp:.4f}  ({dt:.1f}s)")
            scores.append({"psnr": psnr, "ssim": ssim_val,
                           "consistency": cons, "composite": comp})

        dt_total = time.time() - t_total
        avg = {k: float(np.mean([s[k] for s in scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        avg["total_time"] = dt_total
        all_results["baseline"] = avg
        print(f"\n    AVG: PSNR={avg['psnr']:.2f} dB  SSIM={avg['ssim']:.4f}  "
              f"Cons={avg['consistency']:.4f}  Score={avg['composite']:.4f}  "
              f"({dt_total:.1f}s total)")

    # ── Scenario III (Oracle): noisy y + true_spec correction ─────────
    if "oracle" in run_scenarios:
        print(f"\n{'='*78}")
        print(f"  Scenario III (Oracle) — ELP-Unfolding")
        print(f"{'='*78}")

        scores = []
        t_total = time.time()
        for i, s in enumerate(samples):
            ts = s["true_spec"]
            if ts is None:
                print(f"    [{i+1:2d}/{len(samples)}] {s['key']}: no true_spec, skipping")
                continue

            mask_corrected = apply_mismatch(
                s["mask"], ts["mask_dx"], ts["mask_dy"],
                ts["mask_rotation"], ts["mask_blur"],
            )
            gain = ts["gain_drift"]
            offset = ts["offset_drift"]
            y_corrected = ((s["y"] - offset) / max(abs(gain), 1e-6)).astype(np.float32)

            mask_bin = binarize_mask(mask_corrected).astype(np.float32)

            t0 = time.time()
            x_hat = elp_reconstruct(y_corrected, mask_bin, model, device)
            dt = time.time() - t0

            psnr = compute_psnr(s["x_true"], x_hat)
            ssim_val = compute_ssim(s["x_true"], x_hat)
            cons = compute_consistency(s["y"], x_hat, mask_corrected, gain, offset)
            comp = compute_composite(psnr, ssim_val, cons)

            scene = s["metadata"].get("scene", s["key"])
            print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
                  f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
                  f"Score={comp:.4f}  ({dt:.1f}s)")
            scores.append({"psnr": psnr, "ssim": ssim_val,
                           "consistency": cons, "composite": comp})

        dt_total = time.time() - t_total
        avg = {k: float(np.mean([s[k] for s in scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        avg["total_time"] = dt_total
        all_results["oracle"] = avg
        print(f"\n    AVG: PSNR={avg['psnr']:.2f} dB  SSIM={avg['ssim']:.4f}  "
              f"Cons={avg['consistency']:.4f}  Score={avg['composite']:.4f}  "
              f"({dt_total:.1f}s total)")

    # ── Final summary table ───────────────────────────────────────────
    print(f"\n\n{'#'*78}")
    print(f"#  ELP-UNFOLDING BENCHMARK — FINAL RESULTS")
    print(f"#  Dataset: {tier} tier ({len(samples)} samples)")
    print(f"#  Device: {device}")
    if torch.cuda.is_available():
        print(f"#  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'#'*78}")

    print(f"\n  {'Scenario':<25s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}  "
          f"{'Consist':>10s}  {'Score':>8s}  {'Time':>8s}")
    print(f"  {'-'*75}")

    scenario_names = {
        "ideal":    "I   (Ideal)",
        "baseline": "II  (Baseline)",
        "oracle":   "III (Oracle)",
    }
    for sc_key in ["ideal", "baseline", "oracle"]:
        if sc_key not in all_results:
            continue
        avg = all_results[sc_key]
        name = scenario_names[sc_key]
        print(f"  {name:<25s}  {avg['psnr']:>10.2f}  {avg['ssim']:>8.4f}  "
              f"{avg['consistency']:>10.4f}  {avg['composite']:>8.4f}  "
              f"{avg['total_time']:>7.1f}s")

    # Paper comparison
    print(f"\n  --- InverseNet Paper Reference (peak_photon=10000 noise) ---")
    print(f"  {'Scenario':<25s}  {'Paper PSNR':>10s}  {'Paper SSIM':>10s}")
    print(f"  {'-'*50}")
    for sc_key, paper_key in [("baseline", "scenario_ii"), ("oracle", "scenario_iii")]:
        if sc_key in all_results and paper_key in PAPER_BASELINES:
            p = PAPER_BASELINES[paper_key]
            name = scenario_names[sc_key]
            print(f"  {name:<25s}  {p['psnr']:>10.2f}  {p['ssim']:>10.3f}")

    print(f"\n  NOTE: Dataset uses peak_photon=10000 noise (~40 dB SNR),")
    print(f"  matching the InverseNet paper. Results should be close to paper values.")

    print(f"\n{'='*78}")
    print(f"  DONE")
    print(f"{'='*78}")

    return all_results


# ============================================================================
# Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(
    tier: str = "public",
    samples: int = 0,
    scenario: str = "all",
):
    """Run ELP-Unfolding benchmark on Modal GPU.

    Args:
        tier: Dataset tier (public, dev, hidden).
        samples: Max samples to process (0 = all).
        scenario: Which scenario(s) (all, ideal, baseline, oracle).
    """
    print(f"Launching ELP-Unfolding benchmark on Modal A10G GPU...")
    print(f"  Tier: {tier}")
    print(f"  Samples: {'all' if samples == 0 else samples}")
    print(f"  Scenario: {scenario}")

    result = run_elp_benchmark.remote(
        tier=tier,
        max_samples=samples,
        scenario=scenario,
    )

    if result and "error" not in result:
        print("\nResults received from Modal:")
        for sc_key, avg in result.items():
            if isinstance(avg, dict) and "psnr" in avg:
                print(f"  {sc_key}: PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
                      f"Score={avg['composite']:.4f}")
    else:
        print(f"\nError: {result}")
