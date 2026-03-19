#!/usr/bin/env python3
"""Quick test: verify CASSI reconstruction improvement with 3D TV."""
import sys
import time
import numpy as np
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(PROJECT_ROOT / "papers" / "system_design" / "expert_study"))

from data_loader import load_data
from expert_reconstructors import (
    _cassi_gap_tv, _cassi_adjoint_simple,
    expert_e1_cassi, expert_e2_cassi, expert_e3_cassi,
    expert_e4_cassi, expert_e5_cassi,
)

def psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0
    peak = x_true.max()
    return 10 * np.log10(peak ** 2 / mse)

def ssim_3d(x_true, x_recon):
    from skimage.metrics import structural_similarity
    vals = []
    for b in range(x_true.shape[2]):
        dr = x_true[:, :, b].max() - x_true[:, :, b].min()
        if dr < 1e-10:
            dr = 1.0
        vals.append(structural_similarity(
            x_true[:, :, b], x_recon[:, :, b], data_range=dr
        ))
    return np.mean(vals)

def main():
    print("Loading CASSI data from GCS...", flush=True)
    samples = load_data("sd_cassi", n_samples=3)
    print(f"Loaded {len(samples)} samples", flush=True)

    if not samples:
        print("ERROR: No samples loaded")
        return

    s = samples[0]
    y = s["measurements"]
    mask = s["calibration"]["coded_aperture_mask"]
    x_true = s["x_true"]
    n_bands = x_true.shape[2]

    print(f"  y shape: {y.shape}, mask shape: {mask.shape}, "
          f"x_true shape: {x_true.shape}", flush=True)
    print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]", flush=True)

    # Test each expert on sample 0
    experts = {
        "E1 (GAP-TV 3D, iter=50, lam=0.01)": expert_e1_cassi,
        "E2 (Landweber+3DTV, iter=60)": expert_e2_cassi,
        "E3 (GAP-TV 3D, iter=60, lam=0.015)": expert_e3_cassi,
        "E4 (Acc GAP-TV 3D, iter=80, lam=0.008)": expert_e4_cassi,
        "E5 (GAP-TV+NLM, iter=50, lam=0.01)": expert_e5_cassi,
    }

    for name, expert_fn in experts.items():
        t0 = time.time()
        results = expert_fn([s])
        dt = time.time() - t0
        recon = results[0]
        p = psnr(x_true, recon)
        ss = ssim_3d(x_true, recon)
        print(f"  {name}: PSNR={p:.2f} dB, SSIM={ss:.4f}, time={dt:.1f}s", flush=True)

    # Also test parameter sweep on GAP-TV
    print("\n--- Parameter sweep ---", flush=True)
    for iters in [30, 50, 80]:
        for lam in [0.005, 0.01, 0.015, 0.02]:
            t0 = time.time()
            recon = _cassi_gap_tv(y, mask, n_bands, iterations=iters, lam=lam)
            dt = time.time() - t0
            p = psnr(x_true, recon)
            print(f"  iters={iters}, lam={lam:.3f}: PSNR={p:.2f} dB ({dt:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
