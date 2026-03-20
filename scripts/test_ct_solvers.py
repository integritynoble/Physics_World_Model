#!/usr/bin/env python3
"""Test all CT solvers for 100% pass rate on standard dataset."""
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

# Load sample_00
SAMPLE = ROOT / "datasets" / "benchmark" / "ct" / "standard" / "standard_ct_00.h5"

def main():
    if not SAMPLE.exists():
        print(f"SKIP: {SAMPLE} not found")
        return

    with h5py.File(str(SAMPLE), "r") as f:
        x_true = np.array(f["x_true"], dtype=np.float32)
        y = np.array(f["y_ideal"], dtype=np.float32)
        n_det = f["H_params"].attrs.get("n_detectors", y.shape[1])
        img_size = f["H_params"].attrs.get("image_size", x_true.shape[0])

    print(f"x_true: {x_true.shape}, y: {y.shape}, img_size={img_size}, n_det={n_det}")
    operator = CTOperator(y.shape[0], y.shape[1], img_size)
    cfg = {"output_size": img_size}

    total = len(SOLVERS)
    passed = 0
    failed = []
    results = []

    # Test slow iterative solvers with fewer iterations
    fast_cfg = {
        "output_size": img_size,
        # Override to make tests faster
        "iters": 3,
    }

    # Classify solvers by speed
    fast_keys = {
        "traditional_cpu", "fbp_shepp_logan", "fbp_cosine", "fbp_hamming", "fbp_hann",
        "best_quality", "fbp_bm3d", "fbp_bilateral", "fbp_wavelet", "fbp_tv",
        "famous_dl", "small_gpu",
        "fbpconvnet", "wgan_vgg", "learn", "learned_pd", "iradonmap",
        "fbp_unet", "dudonet", "indudonet", "dudotrans", "ctformer",
        "score_ct", "dps", "diffusion_mbir", "dolce", "ct_fm",
    }

    for key, spec in SOLVERS.items():
        t0 = time.time()
        try:
            # Use fast config for iterative solvers
            if key in fast_keys:
                test_cfg = {"output_size": img_size}
            else:
                test_cfg = {"output_size": img_size, "iters": 3}

            x_hat = run_solver(key, y, operator, test_cfg)
            dt = time.time() - t0

            # Check shape
            assert x_hat.shape == x_true.shape, f"shape {x_hat.shape} != {x_true.shape}"
            assert np.isfinite(x_hat).all(), "non-finite values"

            p = psnr(x_true, np.clip(x_hat, 0, 1), data_range=1.0)
            passed += 1
            status = "PASS"
            results.append((key, spec["name"], p, dt, status))
            print(f"  [{passed}/{total}] {key:25s} {spec['name']:25s} PSNR={p:6.2f} dB  ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            failed.append((key, str(e)))
            results.append((key, spec["name"], 0.0, dt, "FAIL"))
            print(f"  [{passed}/{total}] {key:25s} {spec['name']:25s} FAIL: {e}  ({dt:.1f}s)")
        finally:
            gc.collect()

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for k, e in failed:
            print(f"  {k}: {e}")
    else:
        print("ALL SOLVERS PASSED!")

    return passed == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
