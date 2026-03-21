"""Verification script: run all algorithms 5 times for 8 flagship modalities.

For each algorithm:
  1. Run once to establish reference PSNR/SSIM.
  2. Run 5 more times, comparing each run's PSNR/SSIM to the reference.
  3. A run is "done" only if PSNR and SSIM match the reference exactly.
  4. An algorithm is "done" only if all 5 runs are "done".

Results saved to:
  - benchmark_results/verification_8mod_5x.json  (detailed results)
  - benchmark_results/verified_5x_ref.json       (merged into existing ref file)
"""

import inspect
import json
import math
import os
import sys
import time
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


def run_single(mod, sk, y_n, x_true_n, extra_kw):
    """Run a single solver and return (psnr, ssim) rounded to 2/4 decimals."""
    x_hat = mod.run_solver(sk, y_n.copy(), **extra_kw)
    if x_hat.shape != x_true_n.shape:
        x_hat = (x_hat.reshape(x_true_n.shape)
                 if x_hat.size == x_true_n.size else x_true_n * 0)
    psnr = round(compute_psnr(x_true_n, x_hat, data_range=1.0), 2)
    ssim = round(compute_ssim(x_true_n, x_hat), 4)
    return psnr, ssim


def run_verification():
    # Load existing verified_5x_ref.json to merge into
    ref_path = os.path.join(ROOT, "benchmark_results", "verified_5x_ref.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            all_ref = json.load(f)
    else:
        all_ref = {}

    results = {}
    total_algorithms = 0
    total_done = 0

    for modality in MODALITIES:
        print(f"\n{'='*70}")
        print(f"  MODALITY: {modality}")
        print(f"{'='*70}")

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
            mod = __import__(
                f"algorithm_base.{modality}.solvers",
                fromlist=["SOLVERS", "run_solver"])
            solvers = mod.SOLVERS
        except Exception as e:
            print(f"  [ERROR] Cannot import {modality} solvers: {e}")
            results[modality] = {"error": str(e)}
            continue

        # Detect parameter name: some use 'operator', others use 'physics'
        sig = inspect.signature(mod.run_solver)
        param_names = list(sig.parameters.keys())
        if "physics" in param_names:
            extra_kw = {"physics": None, "cfg": {}}
        else:
            extra_kw = {"operator": None, "cfg": {}}

        solver_keys = list(solvers.keys())
        total_algorithms += len(solver_keys)

        mod_ref = {}  # for verified_5x_ref.json

        for sk in solver_keys:
            info = solvers[sk]
            name = info["name"]

            # --- Step 1: Establish reference PSNR/SSIM ---
            try:
                ref_psnr, ref_ssim = run_single(
                    mod, sk, y_n, x_true_n, extra_kw)
            except Exception as e:
                print(f"  {sk:25s} | {name:30s} | REF FAILED: {str(e)[:60]}")
                mod_ref[sk] = {
                    "name": name,
                    "ref_psnr": None,
                    "ref_ssim": None,
                    "runs": [],
                    "avg_psnr": 0.0,
                    "avg_ssim": 0.0,
                    "status": "fail",
                }
                continue

            # Handle NaN reference
            if math.isnan(ref_psnr) or math.isnan(ref_ssim):
                ref_psnr_display = "NaN"
                ref_ssim_display = "NaN"
            else:
                ref_psnr_display = f"{ref_psnr:.2f}"
                ref_ssim_display = f"{ref_ssim:.4f}"

            # --- Step 2: Run 5 times, compare each to reference ---
            runs = []
            for run_idx in range(5):
                t0 = time.time()
                try:
                    psnr, ssim = run_single(
                        mod, sk, y_n, x_true_n, extra_kw)
                    elapsed = time.time() - t0

                    # Compare to reference
                    if math.isnan(ref_psnr) or math.isnan(psnr):
                        match_psnr = math.isnan(ref_psnr) and math.isnan(psnr)
                    else:
                        match_psnr = (psnr == ref_psnr)

                    if math.isnan(ref_ssim) or math.isnan(ssim):
                        match_ssim = math.isnan(ref_ssim) and math.isnan(ssim)
                    else:
                        match_ssim = (ssim == ref_ssim)

                    run_done = match_psnr and match_ssim

                    runs.append({
                        "psnr": psnr,
                        "ssim": ssim,
                        "match_psnr": match_psnr,
                        "match_ssim": match_ssim,
                        "done": run_done,
                    })
                except Exception as e:
                    elapsed = time.time() - t0
                    runs.append({
                        "psnr": 0.0,
                        "ssim": 0.0,
                        "match_psnr": False,
                        "match_ssim": False,
                        "done": False,
                        "error": str(e)[:200],
                    })

            # --- Step 3: Determine status ---
            n_done = sum(1 for r in runs if r.get("done", False))
            ok_psnrs = [r["psnr"] for r in runs
                        if "error" not in r and not math.isnan(r["psnr"])]
            ok_ssims = [r["ssim"] for r in runs
                        if "error" not in r and not math.isnan(r["ssim"])]
            avg_psnr = round(float(np.mean(ok_psnrs)), 2) if ok_psnrs else 0.0
            avg_ssim = round(float(np.mean(ok_ssims)), 4) if ok_ssims else 0.0

            if n_done == 5:
                status = "done"
                total_done += 1
            elif n_done > 0:
                status = "partial"
            else:
                status = "fail"

            mod_ref[sk] = {
                "name": name,
                "ref_psnr": ref_psnr,
                "ref_ssim": ref_ssim,
                "runs": runs,
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "status": status,
            }

            # Print result
            match_str = " ".join(
                "Y" if r.get("done") else "N" for r in runs)
            psnr_str = " ".join(
                f"{r['psnr']:5.1f}" if not math.isnan(r['psnr'])
                else "  NaN" for r in runs)
            print(
                f"  {sk:25s} | {name:30s} | "
                f"ref={ref_psnr_display:>6s}/{ref_ssim_display:>6s} | "
                f"PSNR: {psnr_str} | "
                f"match: [{match_str}] | {n_done}/5 | {status}")

        # Store in results
        results[modality] = {
            "n_solvers": len(solver_keys),
            "n_done": sum(1 for v in mod_ref.values()
                         if v["status"] == "done"),
            "solvers": mod_ref,
        }

        # Merge into all_ref
        all_ref[modality] = mod_ref

    # --- Save results ---
    os.makedirs(os.path.join(ROOT, "benchmark_results"), exist_ok=True)

    # Save detailed results
    out_path = os.path.join(
        ROOT, "benchmark_results", "verification_8mod_5x.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Merge into verified_5x_ref.json
    with open(ref_path, "w") as f:
        json.dump(all_ref, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY (reference-compared verification)")
    print(f"{'='*70}")
    for mod_name in MODALITIES:
        if mod_name in results and "n_solvers" in results[mod_name]:
            r = results[mod_name]
            print(
                f"  {mod_name:20s}: "
                f"{r['n_done']}/{r['n_solvers']} algorithms done "
                f"(all 5 runs match reference)")
    print(f"\n  Total: {total_done}/{total_algorithms} "
          f"algorithms fully verified (5/5 match reference)")
    print(f"  Results: {out_path}")
    print(f"  Ref DB:  {ref_path}")

    return results


if __name__ == "__main__":
    run_verification()
