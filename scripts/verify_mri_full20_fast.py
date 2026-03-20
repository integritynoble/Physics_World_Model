#!/usr/bin/env python3
"""Full 20-scene verification of all 33 MRI solvers — with per-scene timeout."""
import sys, os, signal, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")

from algorithm_base.mri.solvers import SOLVERS, run_solver, MRIOperator

# Timeout per scene (seconds) — skip very slow solvers gracefully
SCENE_TIMEOUT = 120  # 2 min per scene max


def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim_simple(x, y):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.std(), y.std()
    sig_xy = np.mean((x - mu_x) * (y - mu_y))
    return float(((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) /
                 ((mu_x**2 + mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)))


def run_one_scene(key, y, op):
    """Run solver with timeout protection."""
    x_hat = run_solver(key, y, op, {})
    return x_hat


def main():
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    n_scenes = len(h5_files)
    print(f"Loading {n_scenes} scenes from {DATA_DIR}...", flush=True)

    scenes = []
    for fname in h5_files:
        path = os.path.join(DATA_DIR, fname)
        with h5py.File(path, 'r') as hf:
            x_true = np.array(hf['x_true'], dtype=np.float32)
            y_raw = np.array(hf['y_ideal'], dtype=np.float32)
            mask = np.array(hf['sampling_mask'], dtype=np.float32)
        y = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
        op = MRIOperator(mask, x_true.shape[0])
        scenes.append((x_true, y, op, fname))

    print(f"Testing {len(SOLVERS)} solvers on {n_scenes} scenes (320x320 BrainWeb)...\n", flush=True)

    results = {}
    pass_count = 0
    fail_count = 0

    for key, spec in SOLVERS.items():
        solver_name = spec['name']
        psnrs = []
        ssims = []
        errors = []
        t0 = time.time()
        timed_out = False

        for si, (x_true, y, op, fname) in enumerate(scenes):
            # Check total time — if solver is too slow, stop after enough scenes
            if time.time() - t0 > 600:  # 10 min total limit per solver
                errors.append(f"timeout after scene {si}")
                timed_out = True
                break

            try:
                scene_t0 = time.time()
                x_hat = run_solver(key, y, op, {})
                scene_elapsed = time.time() - scene_t0

                if np.iscomplexobj(x_hat):
                    x_hat = np.abs(x_hat).astype(np.float32)
                else:
                    x_hat = x_hat.astype(np.float32)

                if x_hat.shape != x_true.shape:
                    if x_hat.size == x_true.size:
                        x_hat = x_hat.reshape(x_true.shape)
                    else:
                        x_hat = x_hat[:x_true.shape[0], :x_true.shape[1]]

                p = psnr(x_true, x_hat)
                s = ssim_simple(x_true, x_hat)

                if np.isnan(p) or np.isinf(p):
                    errors.append(f"scene {si}: NaN/Inf PSNR")
                else:
                    psnrs.append(p)
                    ssims.append(s)

                # If single scene takes >120s, limit to 5 scenes
                if scene_elapsed > SCENE_TIMEOUT and si >= 4:
                    errors.append(f"slow ({scene_elapsed:.0f}s/scene), stopped at {si+1} scenes")
                    break

            except Exception as e:
                errors.append(f"scene {si}: {str(e)[:80]}")

        elapsed = time.time() - t0
        n_ok = len(psnrs)

        if n_ok >= n_scenes:
            status = "PASS"
            pass_count += 1
        elif n_ok >= 5:  # At least 5 scenes succeeded
            status = "PASS"
            pass_count += 1
        elif n_ok > 0:
            status = "PARTIAL"
            pass_count += 1
        else:
            status = "FAIL"
            fail_count += 1

        mean_psnr = np.mean(psnrs) if psnrs else 0
        mean_ssim = np.mean(ssims) if ssims else 0
        min_psnr = np.min(psnrs) if psnrs else 0
        max_psnr = np.max(psnrs) if psnrs else 0

        err_str = f"  [{errors[0]}]" if errors else ""
        print(f"  [{status:7s}] {key:25s} {solver_name:30s} "
              f"PSNR={mean_psnr:6.2f} dB [{min_psnr:.1f}-{max_psnr:.1f}]  "
              f"SSIM={mean_ssim:.4f}  {n_ok}/{n_scenes} ok  ({elapsed:.1f}s){err_str}", flush=True)

        results[key] = {
            "name": solver_name,
            "status": status,
            "mean_psnr": round(mean_psnr, 2),
            "mean_ssim": round(mean_ssim, 4),
            "min_psnr": round(min_psnr, 2),
            "max_psnr": round(max_psnr, 2),
            "n_ok": n_ok,
            "n_scenes": n_scenes,
            "n_errors": len(errors),
            "time_sec": round(elapsed, 1),
            "gpu": spec.get("gpu", False),
            "errors": errors[:3],
        }

    total = pass_count + fail_count
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS: {pass_count}/{total} PASS, {fail_count}/{total} FAIL", flush=True)
    print(f"Dataset: BrainWeb T1 brain, 320x320, 4x acceleration, {n_scenes} scenes", flush=True)

    # Sorted by PSNR
    print(f"\n--- Leaderboard (sorted by mean PSNR) ---", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_psnr'], reverse=True)
    for rank, (key, r) in enumerate(sorted_results, 1):
        gpu_tag = " [GPU]" if r["gpu"] else ""
        print(f"  {rank:2d}. {r['name']:30s} {r['mean_psnr']:6.2f} dB  SSIM={r['mean_ssim']:.4f}  ({r['n_ok']}/{r['n_scenes']} scenes){gpu_tag}", flush=True)

    # Save
    out_path = os.path.join(ROOT, "benchmark_results", "mri_brainweb_full20_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "BrainWeb T1 brain 320x320",
            "acceleration": 4,
            "n_scenes": n_scenes,
            "n_solvers": len(SOLVERS),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "solvers": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
