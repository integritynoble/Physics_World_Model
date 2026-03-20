#!/usr/bin/env python3
"""Test CT solvers in batches to avoid memory pressure."""
import gc
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithm_base.ct.solvers import SOLVERS, CTOperator, run_solver

SAMPLE = ROOT / "datasets" / "benchmark" / "ct" / "standard" / "standard_ct_00.h5"


def load_data():
    with h5py.File(str(SAMPLE), "r") as f:
        x_true = np.array(f["x_true"], dtype=np.float32)
        y = np.array(f["y_ideal"], dtype=np.float32)
        img_size = f["H_params"].attrs.get("image_size", x_true.shape[0])
    return x_true, y, img_size


def test_batch(keys, x_true, y, img_size, batch_name):
    operator = CTOperator(y.shape[0], y.shape[1], img_size)
    passed = 0
    failed = []
    print(f"\n--- {batch_name} ({len(keys)} solvers) ---")
    for key in keys:
        if key not in SOLVERS:
            continue
        spec = SOLVERS[key]
        gc.collect()
        t0 = time.time()
        try:
            cfg = {"output_size": img_size}
            # Use minimal iterations for iterative solvers
            if key not in {"traditional_cpu", "fbp_shepp_logan", "fbp_cosine",
                           "fbp_hamming", "fbp_hann", "best_quality", "fbp_bm3d",
                           "fbp_bilateral", "fbp_wavelet", "fbp_tv",
                           "famous_dl", "small_gpu", "fbpconvnet", "wgan_vgg",
                           "learn", "learned_pd", "iradonmap", "fbp_unet",
                           "dudonet", "indudonet", "dudotrans", "ctformer",
                           "score_ct", "dps", "diffusion_mbir", "dolce", "ct_fm"}:
                cfg["iters"] = 3
            x_hat = run_solver(key, y, operator, cfg)
            dt = time.time() - t0
            assert x_hat.shape == x_true.shape, f"shape {x_hat.shape} != {x_true.shape}"
            assert np.isfinite(x_hat).all(), "non-finite values"
            p = psnr(x_true, np.clip(x_hat, 0, 1), data_range=1.0)
            passed += 1
            print(f"  PASS {key:25s} {spec['name']:25s} PSNR={p:6.2f} dB  ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            failed.append((key, str(e)))
            print(f"  FAIL {key:25s} {spec['name']:25s} {e}  ({dt:.1f}s)")
        gc.collect()
    return passed, failed


def main():
    if not SAMPLE.exists():
        print(f"SKIP: {SAMPLE} not found")
        return False

    x_true, y, img_size = load_data()
    print(f"x_true: {x_true.shape}, y: {y.shape}, img_size={img_size}")

    # Batch 1: FBP variants (fast, no memory pressure)
    batch1 = ["traditional_cpu", "fbp_shepp_logan", "fbp_cosine", "fbp_hamming", "fbp_hann"]

    # Batch 2: FBP + post-processing (moderate memory)
    batch2 = ["best_quality", "fbp_bm3d", "fbp_bilateral", "fbp_wavelet", "fbp_tv"]

    # Batch 3: DL stubs (fast, use _dl_fallback)
    batch3 = ["famous_dl", "small_gpu", "fbpconvnet", "wgan_vgg", "learn",
              "learned_pd", "iradonmap", "fbp_unet", "dudonet", "indudonet",
              "dudotrans", "ctformer", "score_ct", "dps", "diffusion_mbir",
              "dolce", "ct_fm"]

    # Batch 4: PnP (uses pwm_core.recon.pnp, moderate memory)
    batch4 = ["pnp_admm_nlm", "pnp_hqs_nlm", "pnp_fista_nlm", "pnp_admm_bm3d"]

    # Batch 5: Iterative solvers (heavy memory, test last)
    batch5 = ["landweber", "art", "sirt", "cgls", "mlem", "sart", "osem",
              "tikhonov", "tv_admm", "chambolle_pock"]

    total_passed = 0
    all_failed = []

    for batch, name in [(batch1, "FBP Variants"), (batch2, "FBP + Post-processing"),
                        (batch3, "DL Models"), (batch4, "PnP Methods"),
                        (batch5, "Iterative Methods")]:
        p, f = test_batch(batch, x_true, y, img_size, name)
        total_passed += p
        all_failed.extend(f)
        gc.collect()

    total = len(SOLVERS)
    print(f"\n{'='*70}")
    print(f"Results: {total_passed}/{total} passed ({100*total_passed/total:.1f}%)")
    if all_failed:
        print(f"\nFailed ({len(all_failed)}):")
        for k, e in all_failed:
            print(f"  {k}: {e}")
    else:
        print("ALL SOLVERS PASSED!")
    return len(all_failed) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
