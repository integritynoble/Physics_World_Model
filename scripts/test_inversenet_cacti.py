#!/usr/bin/env python3
"""InverseNet CACTI Benchmark — All Solvers × All Scenarios.

Reproduces the InverseNet paper Table results across three scenarios:
  Scenario I   (Ideal):    clean y from ground truth + nominal mask
  Scenario II  (Baseline): corrupted y + nominal mask (no correction)
  Scenario III (Oracle):   corrupted y + true_spec correction

Solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI, HiSViT-9, HiSViT-13

Dataset uses peak_photon=10000 noise (~40 dB measurement SNR), matching the
InverseNet paper. Forward model uses binarized warped mask. Standard accelerated
GAP works for all scenarios.

Usage:
    python scripts/test_inversenet_cacti.py
    python scripts/test_inversenet_cacti.py --samples 5       # quick test
    python scripts/test_inversenet_cacti.py --scenario oracle  # single scenario
    python scripts/test_inversenet_cacti.py --method gap_tv    # single method
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# ── Project paths ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# Force-import cacti_solvers from THIS repo (bypass globally installed pwm_core)
import importlib.util as _ilu
_SOLVERS_FILE = str(_ROOT / "packages" / "pwm_core" / "pwm_core" / "recon" / "cacti_solvers.py")
_spec = _ilu.spec_from_file_location("cacti_solvers", _SOLVERS_FILE)
_cs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cs)

SOLVERS = _cs.SOLVERS
solve_cacti = _cs.solve_cacti
_resolve_device = _cs._resolve_device
_cacti_solvers = _cs

# ── Dataset path ─────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "datasets" / "benchmark" / "cacti"

# ── InverseNet paper reference values ────────────────────────────────────────
# Dataset now uses peak_photon=10000 noise (~40 dB SNR), matching the paper.
# Results should be close to these reference values.
PAPER_BASELINES = {
    "scenario_ii": {
        "EfficientSCI":  {"psnr": 27.38, "ssim": 0.927},
        "ELP-Unfolding": {"psnr": 26.50, "ssim": 0.910},
        "PnP-FFDNet":    {"psnr": 20.15, "ssim": 0.650},
        "GAP-TV":        {"psnr": 14.81, "ssim": 0.303},
    },
    "scenario_iii": {
        "EfficientSCI":  {"psnr": 35.39, "ssim": 0.973},
        "ELP-Unfolding": {"psnr": 34.09, "ssim": 0.965},
        "PnP-FFDNet":    {"psnr": 29.28, "ssim": 0.910},
        "GAP-TV":        {"psnr": 26.75, "ssim": 0.870},
    },
}

# Map solver keys to display names
SOLVER_DISPLAY = {
    "gap_tv": "GAP-TV",
    "pnp_ffdnet": "PnP-FFDNet",
    "elp_unfolding": "ELP-Unfolding",
    "efficient_sci": "EfficientSCI",
    "hisvit9": "HiSViT-9",
    "hisvit13": "HiSViT-13",
}

# DL methods that need binary masks
DL_METHODS = {"efficient_sci", "elp_unfolding", "hisvit9", "hisvit13"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply spatial mismatch to mask using the paper's exact affine_transform.

    Matches validate_cacti_inversenet.py:129-146 exactly: single affine
    combining shift + rotation around image center.
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


def compute_psnr(x_true, x_hat, max_val=1.0):
    mse = float(np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 60.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x_true, x_hat):
    """Per-frame SSIM averaged over temporal dimension."""
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


def binarize_mask(mask):
    return (mask > 0.5).astype(np.float64)


# ── Reconstruction dispatcher ────────────────────────────────────────────────

def reconstruct(y, mask, method, device, iters):
    """Dispatch reconstruction via solve_cacti().

    With peak_photon=10000 noise (~40 dB SNR), the standard accelerated GAP
    works well for all scenarios. No special noisy-data handling needed.
    """
    y_in = y.astype(np.float32)
    mask_in = mask.astype(np.float32)
    if method in DL_METHODS:
        mask_in = binarize_mask(mask).astype(np.float32)
    return solve_cacti(y_in, mask_in, method=method, device=device,
                       iterations=iters)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_public_tier(max_samples=None):
    """Load public tier dataset from HDF5."""
    import h5py
    h5_path = DATA_DIR / "public" / "cacti_challenge_public.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        sys.exit(1)

    samples = []
    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(f.keys())
        if max_samples:
            sample_keys = sample_keys[:max_samples]
        for sk in sample_keys:
            grp = f[sk]
            samples.append({
                "key": sk,
                "y": grp["y"][:].astype(np.float64),
                "mask": grp["H_ideal"][:].astype(np.float64),
                "x_true": grp["x_true"][:].astype(np.float64),
                "spec_ranges": json.loads(grp.attrs["spec_ranges"]),
                "true_spec": json.loads(grp.attrs["true_spec"]),
                "metadata": json.loads(grp.attrs.get("metadata", "{}")),
            })
    return samples


# ── Scenario runners ─────────────────────────────────────────────────────────

def run_scenario_i(samples, method, device, iters=100, **_kw):
    """Scenario I (Ideal): clean y from ground truth + nominal mask."""
    scores = []
    for i, s in enumerate(samples):
        y_clean = np.sum(s["mask"] * s["x_true"], axis=2)

        t0 = time.time()
        x_hat = reconstruct(y_clean, s["mask"], method, device, iters)
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
                        "consistency": cons, "composite": comp, "time": dt})
    return scores


def run_scenario_ii(samples, method, device, iters=100, **_kw):
    """Scenario II (Baseline): corrupted y + nominal mask (no correction)."""
    scores = []
    for i, s in enumerate(samples):
        t0 = time.time()
        x_hat = reconstruct(s["y"], s["mask"], method, device, iters)
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
                        "consistency": cons, "composite": comp, "time": dt})
    return scores


def run_scenario_iii(samples, method, device, iters=100, **_kw):
    """Scenario III (Oracle): corrupted y + true_spec correction.

    Uses binarized warped mask to match the forward model used during
    data generation (InverseNet paper binarizes after warping).
    """
    scores = []
    for i, s in enumerate(samples):
        ts = s["true_spec"]
        mask_corrected = apply_mismatch(
            s["mask"], ts["mask_dx"], ts["mask_dy"],
            ts["mask_rotation"], ts["mask_blur"],
        )
        # Binarize to match forward model (data generated with binarized mask)
        mask_corrected = binarize_mask(mask_corrected)
        gain = ts["gain_drift"]
        offset = ts["offset_drift"]
        y_corrected = (s["y"] - offset) / max(abs(gain), 1e-6)

        t0 = time.time()
        x_hat = reconstruct(y_corrected, mask_corrected, method, device, iters)
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
                        "consistency": cons, "composite": comp, "time": dt})
    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InverseNet CACTI Benchmark")
    parser.add_argument("--samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "ideal", "baseline", "oracle"],
                        help="Which InverseNet scenario(s) to run")
    parser.add_argument("--method", default="all",
                        help="Solver name or 'all'")
    parser.add_argument("--iters", type=int, default=100,
                        help="GAP-TV iterations")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda:0)")
    args = parser.parse_args()

    if args.device == "auto":
        device = _resolve_device("auto")
    else:
        device = args.device
    print(f"Device: {device}")

    # Check solver availability
    cs = _cacti_solvers
    print("\n--- Solver availability ---")
    available_real = []
    available_fallback = []
    for name in SOLVERS:
        display = SOLVER_DISPLAY.get(name, name)
        if name == "gap_tv":
            print(f"  {display:<16s}  [NATIVE]  Classical TV denoiser")
            available_real.append(name)
        elif name == "pnp_ffdnet":
            if os.path.isfile(cs._DNCNN_WEIGHTS):
                print(f"  {display:<16s}  [DnCNN]   Pretrained denoiser")
                available_real.append(name)
            elif os.path.isfile(cs._FFDNET_WEIGHTS):
                print(f"  {display:<16s}  [FFDNet]  Pretrained denoiser")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> GAP-TV (no weights)")
                available_fallback.append(name)
        elif name == "elp_unfolding":
            if os.path.isfile(cs._ELP_CKPT):
                print(f"  {display:<16s}  [NATIVE]  ECCV 2022 (565M params)")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 2-pass GAP-TV")
                available_fallback.append(name)
        elif name == "efficient_sci":
            if os.path.isfile(cs._ESCI_CKPT):
                print(f"  {display:<16s}  [NATIVE]  CVPR 2023")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 3-pass GAP-TV")
                available_fallback.append(name)
        elif name in ("hisvit9", "hisvit13"):
            ckpt = cs._HISVIT9_CKPT if name == "hisvit9" else cs._HISVIT13_CKPT
            if os.path.isfile(ckpt):
                print(f"  {display:<16s}  [NATIVE]  ECCV 2024")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 3-pass GAP-TV")
                available_fallback.append(name)

    # Noise model info
    print("\n--- Dataset noise model ---")
    print("  Model: poisson_gaussian (peak_photon=10000, sigma=1.0)")
    print("  This matches the InverseNet paper (~40 dB measurement SNR).")
    print("  Standard accelerated GAP used for all scenarios.")

    # Select methods
    if args.method == "all":
        methods = list(SOLVERS.keys())
    else:
        methods = [args.method]

    # Load data
    max_samples = args.samples if args.samples > 0 else None
    print(f"\n--- Loading public tier data ---")
    samples = load_public_tier(max_samples)
    print(f"  Loaded {len(samples)} samples ({samples[0]['mask'].shape})")
    print(f"  True spec: {samples[0]['true_spec']}")

    # Select scenarios
    scenarios = {
        "ideal": ("Scenario I  (Ideal)", run_scenario_i),
        "baseline": ("Scenario II (Baseline)", run_scenario_ii),
        "oracle": ("Scenario III (Oracle)", run_scenario_iii),
    }
    if args.scenario == "all":
        run_scenarios = list(scenarios.keys())
    else:
        run_scenarios = [args.scenario]

    # ── Run all combinations ─────────────────────────────────────────────
    all_results = {}

    for sc_key in run_scenarios:
        sc_name, sc_func = scenarios[sc_key]
        all_results[sc_key] = {}

        print(f"\n{'='*78}")
        print(f"  {sc_name}")
        print(f"{'='*78}")

        for method in methods:
            display = SOLVER_DISPLAY.get(method, method)
            print(f"\n  --- {display} ---")

            t_total = time.time()
            scores = sc_func(samples, method, device, iters=args.iters)
            dt_total = time.time() - t_total

            avg = {k: float(np.mean([s[k] for s in scores]))
                   for k in ("psnr", "ssim", "consistency", "composite", "time")}
            avg["total_time"] = dt_total
            all_results[sc_key][method] = avg

            print(f"\n    AVG: PSNR={avg['psnr']:.2f} dB  SSIM={avg['ssim']:.4f}  "
                  f"Cons={avg['consistency']:.4f}  Score={avg['composite']:.4f}  "
                  f"({dt_total:.1f}s total)")

    # ── Final comparison tables ──────────────────────────────────────────
    print(f"\n\n{'#'*78}")
    print(f"#  INVERSENET CACTI BENCHMARK - FINAL RESULTS")
    print(f"#  Dataset: public tier ({len(samples)} samples, 256x256x8)")
    print(f"#  Noise: peak_photon=10000 (~40 dB SNR, matching InverseNet paper)")
    print(f"#  Device: {device}")
    print(f"{'#'*78}")

    for sc_key in run_scenarios:
        sc_name = scenarios[sc_key][0]
        results = all_results[sc_key]

        print(f"\n{'='*78}")
        print(f"  {sc_name}")
        print(f"{'='*78}")
        print(f"  {'Method':<20s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}  "
              f"{'Consist':>10s}  {'Score':>8s}  {'Time':>8s}")
        print(f"  {'-'*72}")

        for method in methods:
            if method not in results:
                continue
            avg = results[method]
            display = SOLVER_DISPLAY.get(method, method)
            fallback = " *" if method in available_fallback else ""
            print(f"  {display + fallback:<20s}  {avg['psnr']:>10.2f}  {avg['ssim']:>8.4f}  "
                  f"{avg['consistency']:>10.4f}  {avg['composite']:>8.4f}  "
                  f"{avg['total_time']:>7.1f}s")

        # Compare with paper baselines
        paper_key = f"scenario_{'ii' if sc_key == 'baseline' else 'iii' if sc_key == 'oracle' else ''}"
        if paper_key in PAPER_BASELINES:
            print(f"\n  --- InverseNet Paper Reference ---")
            print(f"  {'Method':<20s}  {'Paper PSNR':>10s}  {'Paper SSIM':>10s}")
            print(f"  {'-'*44}")
            for method in methods:
                if method not in results:
                    continue
                display = SOLVER_DISPLAY.get(method, method)
                if display in PAPER_BASELINES[paper_key]:
                    paper = PAPER_BASELINES[paper_key][display]
                    print(f"  {display:<20s}  {paper['psnr']:>10.2f}  {paper['ssim']:>10.3f}")

    if available_fallback:
        print(f"\n  * Methods marked with '*' used GAP-TV fallback (no pretrained weights).")
        print(f"    To get paper-grade results, download checkpoints to:")
        print(f"    {_ROOT / 'checkpoint'}/")

    print(f"\n  NOTE: Dataset uses peak_photon=10000 noise (~40 dB SNR),")
    print(f"  matching the InverseNet paper. Results should be close to paper values.")

    print(f"\n{'='*78}")
    print(f"  DONE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
