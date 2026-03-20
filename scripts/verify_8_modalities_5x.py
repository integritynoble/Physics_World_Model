"""Verification script: run all algorithms 5 times for 8 flagship modalities.

Loads standard datasets, runs each solver, computes PSNR/SSIM, repeats 5 times.
Results saved to benchmark_results/verification_8mod_5x.json
"""

import json
import os
import sys
import time
import traceback
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "packages", "pwm_core"))


def compute_psnr(x_true, x_hat, data_range=1.0):
    mse = np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 60.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(x_true, x_hat):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            x_true.astype(np.float64),
            x_hat.astype(np.float64),
            data_range=float(x_true.max() - x_true.min()) + 1e-10
        ))
    except Exception:
        return 0.0


def load_standard_sample(modality, sample_idx=0):
    """Load x_true and y_ideal from standard dataset."""
    import h5py
    path = os.path.join(
        ROOT, "datasets", "benchmark", modality, "standard",
        f"standard_{modality}_{sample_idx:02d}.h5"
    )
    if not os.path.exists(path):
        return None, None
    with h5py.File(path, "r") as f:
        grp = f["sample_00"] if "sample_00" in f else f
        x_true = np.array(grp["x_true"], dtype=np.float32)
        y_ideal = np.array(grp["y_ideal"], dtype=np.float32)
    return x_true, y_ideal


MODALITIES = [
    "spc", "lensless", "holography", "ptychography",
    "cbct", "ultrasound", "cryo_em", "widefield"
]


def run_verification():
    results = {}
    total_algorithms = 0
    total_done = 0

    for modality in MODALITIES:
        print(f"\n{'='*60}")
        print(f"  MODALITY: {modality}")
        print(f"{'='*60}")

        # Load test data
        x_true, y = load_standard_sample(modality, 0)
        if x_true is None:
            print(f"  [SKIP] No standard dataset for {modality}")
            results[modality] = {"error": "no_data"}
            continue

        # Normalize to [0, 1]
        x_range = float(x_true.max() - x_true.min())
        if x_range > 1e-8:
            x_true_n = (x_true - x_true.min()) / x_range
            y_n = (y - y.min()) / (float(y.max() - y.min()) + 1e-10)
        else:
            x_true_n = x_true
            y_n = y

        # Import modality solver
        try:
            mod = __import__(f"algorithm_base.{modality}.solvers", fromlist=["SOLVERS", "run_solver"])
            solvers = mod.SOLVERS
        except Exception as e:
            print(f"  [ERROR] Cannot import {modality} solvers: {e}")
            results[modality] = {"error": str(e)}
            continue

        mod_results = {}
        solver_keys = list(solvers.keys())
        total_algorithms += len(solver_keys)

        for sk in solver_keys:
            info = solvers[sk]
            name = info["name"]
            runs = []
            all_passed = True

            for run_idx in range(5):
                t0 = time.time()
                try:
                    x_hat = mod.run_solver(sk, y_n.copy(), operator=None, cfg={})
                    elapsed = time.time() - t0

                    # Ensure correct shape
                    if x_hat.shape != x_true_n.shape:
                        x_hat = x_hat.reshape(x_true_n.shape) if x_hat.size == x_true_n.size else x_true_n * 0

                    psnr = compute_psnr(x_true_n, x_hat, data_range=1.0)
                    ssim = compute_ssim(x_true_n, x_hat)

                    runs.append({
                        "run": run_idx + 1,
                        "psnr": round(psnr, 2),
                        "ssim": round(ssim, 4),
                        "time_s": round(elapsed, 3),
                        "status": "ok",
                    })
                    marker = "ok"
                except Exception as e:
                    elapsed = time.time() - t0
                    runs.append({
                        "run": run_idx + 1,
                        "psnr": 0.0,
                        "ssim": 0.0,
                        "time_s": round(elapsed, 3),
                        "status": "error",
                        "error": str(e)[:200],
                    })
                    marker = "ERR"
                    all_passed = False

            # Compute mean PSNR/SSIM across successful runs
            ok_runs = [r for r in runs if r["status"] == "ok"]
            mean_psnr = round(np.mean([r["psnr"] for r in ok_runs]), 2) if ok_runs else 0.0
            mean_ssim = round(np.mean([r["ssim"] for r in ok_runs]), 4) if ok_runs else 0.0
            status = "done" if len(ok_runs) == 5 else "partial" if ok_runs else "fail"

            if status == "done":
                total_done += 1

            mod_results[sk] = {
                "name": name,
                "runs": runs,
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "status": status,
                "reference": info.get("reference", ""),
            }

            psnr_str = " ".join(
                f"{r['psnr']:5.1f}" if r["status"] == "ok" else "  ERR" for r in runs
            )
            print(f"  {sk:25s} | {name:25s} | PSNR: {psnr_str} | mean={mean_psnr:5.1f} | {status}")

        results[modality] = {
            "n_solvers": len(solver_keys),
            "n_done": sum(1 for v in mod_results.values() if v["status"] == "done"),
            "solvers": mod_results,
        }

    # Save results
    out_path = os.path.join(ROOT, "benchmark_results", "verification_8mod_5x.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for mod in MODALITIES:
        if mod in results and "n_solvers" in results[mod]:
            r = results[mod]
            print(f"  {mod:20s}: {r['n_done']}/{r['n_solvers']} algorithms done (5/5 runs passed)")
    print(f"\n  Total: {total_done}/{total_algorithms} algorithms fully verified")
    print(f"  Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    run_verification()
