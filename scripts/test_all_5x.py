#!/usr/bin/env python3
"""Run every solver 5 times for all non-flagship modalities on standard data.

Outputs JSON results and generates markdown table for index.md.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLAGSHIP = {
    'cassi', 'cacti', 'mri', 'ct', 'spc', 'lensless', 'holography',
    'ptychography', 'cbct', 'ultrasound', 'cryo_em', 'widefield',
}

RESULTS_FILE = ROOT / "benchmark_results" / "template_5x_results.json"
N_RUNS = 5


def get_modalities():
    """Get all non-flagship modalities with standard data."""
    algo_dir = ROOT / "algorithm_base"
    all_mods = sorted([
        d for d in os.listdir(algo_dir)
        if (algo_dir / d).is_dir()
        and not d.startswith('_')
        and not d.startswith('.')
        and d != 'shared'
        and d != '__pycache__'
    ])
    return [m for m in all_mods if m not in FLAGSHIP]


def load_standard_sample(mod_id):
    """Load first standard sample for a modality."""
    import h5py
    import glob
    pattern = str(ROOT / "datasets" / "benchmark" / mod_id / "standard" / f"standard_{mod_id}_00.h5")
    if os.path.exists(pattern):
        fpath = pattern
    else:
        files = sorted(glob.glob(str(ROOT / "datasets" / "benchmark" / mod_id / "standard" / "*.h5")))
        if not files:
            return None, None
        fpath = files[0]
    with h5py.File(fpath, 'r') as f:
        x_true = np.array(f['x_true'], dtype=np.float32)
        y_key = 'y_ideal' if 'y_ideal' in f else 'y'
        y = np.array(f[y_key], dtype=np.float32)
    return x_true, y


def compute_psnr(x_true, x_hat):
    """Compute PSNR."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    # Handle shape mismatch
    if x_hat.shape != x_true.shape:
        if x_hat.ndim == 2 and x_true.ndim == 3:
            # Use first channel
            x_true = x_true[:, :, 0] if x_true.shape[2] <= 4 else x_true
        elif x_hat.ndim == 3 and x_true.ndim == 2:
            x_hat = x_hat[:, :, 0] if x_hat.shape[2] <= 4 else x_hat
        if x_hat.shape != x_true.shape:
            # Resize
            from skimage.transform import resize
            x_hat = resize(x_hat, x_true.shape, anti_aliasing=False).astype(np.float32)
    # data range
    dr = max(float(x_true.max()) - float(x_true.min()), 1e-8)
    return float(psnr(x_true, np.clip(x_hat, x_true.min(), x_true.max()), data_range=dr))


def test_modality(mod_id, x_true, y):
    """Test all solvers for a modality, 5 runs each."""
    import importlib.util
    fpath = str(ROOT / "algorithm_base" / mod_id / "solvers.py")
    try:
        spec = importlib.util.spec_from_file_location(f"solvers_{mod_id}", fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return {"error": str(e), "solvers": {}}

    solvers = mod.SOLVERS
    results = {}

    for solver_key in solvers:
        runs = []
        for run_i in range(N_RUNS):
            try:
                x_hat = mod.run_solver(solver_key, y.copy())
                p = compute_psnr(x_true, x_hat)
                runs.append(round(p, 1))
            except Exception as e:
                runs.append(None)
            gc.collect()

        mean_psnr = None
        if all(r is not None for r in runs):
            mean_psnr = round(sum(runs) / len(runs), 1)

        results[solver_key] = {
            "name": solvers[solver_key]["name"],
            "runs": runs,
            "mean_psnr": mean_psnr,
            "status": "done" if all(r is not None for r in runs) else "fail",
        }

    return {"error": None, "solvers": results}


def main():
    mods = get_modalities()
    print(f"Testing {len(mods)} non-flagship modalities, {N_RUNS} runs each...")

    # Load existing results if resuming
    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        print(f"  Resuming from {len(all_results)} completed modalities")

    done = 0
    failed = 0
    total_solvers = 0
    total_done = 0

    for i, mod_id in enumerate(mods):
        if mod_id in all_results and all_results[mod_id].get("error") is None:
            # Already completed
            n_s = len(all_results[mod_id].get("solvers", {}))
            n_d = sum(1 for v in all_results[mod_id].get("solvers", {}).values()
                      if v.get("status") == "done")
            total_solvers += n_s
            total_done += n_d
            done += 1
            continue

        # Load data
        x_true, y = load_standard_sample(mod_id)
        if x_true is None or y is None:
            all_results[mod_id] = {"error": "no_data", "solvers": {}}
            failed += 1
            print(f"  [{i+1:3d}/{len(mods)}] SKIP {mod_id:30s} no data")
            continue

        # Handle 3D x_true with 2D y
        if x_true.ndim == 3 and y.ndim == 2:
            x_true_2d = x_true[:, :, 0].astype(np.float32)
        else:
            x_true_2d = x_true

        t0 = time.time()
        result = test_modality(mod_id, x_true_2d, y)
        dt = time.time() - t0

        all_results[mod_id] = result

        n_s = len(result.get("solvers", {}))
        n_d = sum(1 for v in result.get("solvers", {}).values()
                  if v.get("status") == "done")
        total_solvers += n_s
        total_done += n_d

        if result.get("error"):
            failed += 1
            print(f"  [{i+1:3d}/{len(mods)}] ERR  {mod_id:30s} {result['error'][:60]}")
        else:
            done += 1
            print(f"  [{i+1:3d}/{len(mods)}] OK   {mod_id:30s} {n_d}/{n_s} done  {dt:.1f}s")

        # Save progress every 5 modalities
        if (i + 1) % 5 == 0 or i == len(mods) - 1:
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

        gc.collect()

    # Final save
    os.makedirs(RESULTS_FILE.parent, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Modalities: {done} done, {failed} failed, {len(mods)} total")
    print(f"Solvers: {total_done}/{total_solvers} done ({100*total_done/max(total_solvers,1):.1f}%)")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
