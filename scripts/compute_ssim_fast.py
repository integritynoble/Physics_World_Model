#!/usr/bin/env python3
"""Fast SSIM computation — one run per solver, skip external modules entirely."""
import json, gc, sys, os, time
import numpy as np
import h5py
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "benchmark_results" / "ssim_results.json"

def get_all_modalities():
    algo_dir = ROOT / "algorithm_base"
    return sorted([
        d for d in os.listdir(algo_dir)
        if (algo_dir / d).is_dir()
        and not d.startswith('_') and not d.startswith('.')
        and d not in ('shared', '__pycache__')
    ])

def load_data(mod_id):
    import glob
    std_dir = ROOT / "datasets" / "benchmark" / mod_id / "standard"
    if not std_dir.exists():
        return None, None
    pattern = str(std_dir / f"standard_{mod_id}_00.h5")
    fpath = pattern if os.path.exists(pattern) else None
    if not fpath:
        files = sorted(glob.glob(str(std_dir / "*.h5")))
        if not files:
            return None, None
        fpath = files[0]
    with h5py.File(fpath, 'r') as f:
        x_true = np.array(f['x_true'], dtype=np.float32)
        if 'y_ideal' in f:
            y = np.array(f['y_ideal'], dtype=np.float32)
        elif 'y' in f:
            y = np.array(f['y'], dtype=np.float32)
        elif 'luminance' in f:
            y = np.array(f['luminance'], dtype=np.float32)
        else:
            return None, None
    if y.ndim == 3 and y.shape[0] < y.shape[1]:
        y = y[0].astype(np.float32)
    if x_true.ndim == 4:
        x_true = np.mean(x_true[0], axis=-1).astype(np.float32)
    elif x_true.ndim == 3 and y.ndim == 2:
        x_true = x_true[:, :, 0].astype(np.float32)
    return x_true, y

def main():
    mods = get_all_modalities()
    print(f"Computing SSIM for {len(mods)} modalities (1 run, inline only)...")

    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        done = sum(1 for v in all_results.values() if v)
        print(f"  Resuming from {done} completed")

    for i, mod_id in enumerate(mods):
        if mod_id in all_results and all_results[mod_id]:
            continue

        x_true, y = load_data(mod_id)
        if x_true is None:
            all_results[mod_id] = {}
            continue

        fpath = str(ROOT / "algorithm_base" / mod_id / "solvers.py")
        if not os.path.exists(fpath):
            all_results[mod_id] = {}
            continue

        spec = importlib.util.spec_from_file_location(f"solvers_{mod_id}", fpath)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            all_results[mod_id] = {}
            continue

        dr = max(float(x_true.max()) - float(x_true.min()), 1e-8)
        results = {}

        for sk, ss in mod.SOLVERS.items():
            solver_mod = ss.get("module", "")
            if not solver_mod.startswith(f"algorithm_base.{mod_id}."):
                continue  # Skip external
            try:
                x_hat = mod.run_solver(sk, y.copy())
                if x_hat.shape != x_true.shape:
                    from skimage.transform import resize
                    x_hat = resize(x_hat, x_true.shape, anti_aliasing=False).astype(np.float32)
                x_hat_clip = np.clip(x_hat, x_true.min(), x_true.max())
                s = float(ssim(x_true, x_hat_clip, data_range=dr))
                results[sk] = round(s, 4)
            except Exception:
                results[sk] = None
            gc.collect()

        all_results[mod_id] = results
        n_done = sum(1 for v in results.values() if v is not None)
        print(f"  [{i+1:3d}/{len(mods)}] {mod_id:30s} {n_done}/{len(results)} ssim computed")

        if (i + 1) % 5 == 0 or i == len(mods) - 1:
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
