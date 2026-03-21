#!/usr/bin/env python3
"""Run every solver 5 times for all non-flagship modalities — fast version.

Uses subprocess isolation and processes modalities sequentially but efficiently.
Saves results incrementally to JSON.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FLAGSHIP = {
    'cassi', 'cacti', 'mri', 'ct', 'spc', 'lensless', 'compressive_holography',
    'ptychography', 'cbct', 'ultrasound', 'cryo_em', 'widefield',
}

RESULTS_FILE = ROOT / "benchmark_results" / "template_5x_results.json"
N_RUNS = 5

TEST_CODE = '''
import sys, gc, json, os, traceback
sys.path.insert(0, "{root}")
import numpy as np, h5py, importlib.util
from skimage.metrics import peak_signal_noise_ratio as psnr

mod_id = "{mod_id}"
fpath = os.path.join("{root}", "algorithm_base", mod_id, "solvers.py")
spec = importlib.util.spec_from_file_location("solvers_" + mod_id, fpath)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Load data
h5_path = os.path.join("{root}", "datasets", "benchmark", mod_id, "standard", f"standard_{{mod_id}}_00.h5")
if not os.path.exists(h5_path):
    import glob
    files = sorted(glob.glob(os.path.join("{root}", "datasets", "benchmark", mod_id, "standard", "*.h5")))
    if not files:
        print(json.dumps({{"error": "no_data", "solvers": {{}}}}))
        sys.exit(0)
    h5_path = files[0]

with h5py.File(h5_path, "r") as f:
    x_true = np.array(f["x_true"], dtype=np.float32)
    if "y_ideal" in f:
        y = np.array(f["y_ideal"], dtype=np.float32)
    elif "y" in f:
        y = np.array(f["y"], dtype=np.float32)
    elif "luminance" in f:
        y = np.array(f["luminance"], dtype=np.float32)
    else:
        print(json.dumps({{"error": "no_y_key", "solvers": {{}}}}))
        sys.exit(0)

# Handle multi-view data (e.g. nerf, gaussian_splatting): take single view
if y.ndim == 3 and y.shape[0] < y.shape[1]:
    y = y[0].astype(np.float32)
if x_true.ndim == 4:
    x_true = np.mean(x_true[0], axis=-1).astype(np.float32)
elif x_true.ndim == 3 and y.ndim == 2:
    x_true = x_true[:, :, 0].astype(np.float32)

dr = max(float(x_true.max()) - float(x_true.min()), 1e-8)
results = {{}}

for solver_key in mod.SOLVERS:
    # Skip external solvers (pwm_core) — only test inline solvers
    solver_mod = mod.SOLVERS[solver_key].get("module", "")
    if solver_mod and not solver_mod.startswith("algorithm_base."):
        continue
    runs = []
    for run_i in range({n_runs}):
        try:
            x_hat = mod.run_solver(solver_key, y.copy())
            if x_hat.shape != x_true.shape:
                from skimage.transform import resize
                x_hat = resize(x_hat, x_true.shape, anti_aliasing=False).astype(np.float32)
            p = float(psnr(x_true, np.clip(x_hat, x_true.min(), x_true.max()), data_range=dr))
            runs.append(round(p, 1))
        except Exception as e:
            runs.append(None)
        gc.collect()
    mean_psnr = round(sum(r for r in runs if r is not None) / max(sum(1 for r in runs if r is not None), 1), 1)
    results[solver_key] = {{
        "name": mod.SOLVERS[solver_key]["name"],
        "runs": runs,
        "mean_psnr": mean_psnr if all(r is not None for r in runs) else None,
        "status": "done" if all(r is not None for r in runs) else "fail",
    }}

print(json.dumps({{"error": None, "solvers": results}}))
'''


def get_modalities():
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


def main():
    mods = get_modalities()
    print(f"Testing {len(mods)} non-flagship modalities, {N_RUNS} runs each...")

    # Load existing results
    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        done_count = sum(1 for v in all_results.values() if v.get("error") is None)
        print(f"  Resuming from {done_count} completed modalities")

    total_done = 0
    total_fail = 0
    total_solvers_done = 0
    total_solvers = 0

    for i, mod_id in enumerate(mods):
        if mod_id in all_results and all_results[mod_id].get("error") is None:
            r = all_results[mod_id]
            n_s = len(r.get("solvers", {}))
            n_d = sum(1 for v in r.get("solvers", {}).values() if v.get("status") == "done")
            total_solvers += n_s
            total_solvers_done += n_d
            total_done += 1
            continue

        # Check if solvers.py exists
        if not (ROOT / "algorithm_base" / mod_id / "solvers.py").exists():
            all_results[mod_id] = {"error": "no_solvers_py", "solvers": {}}
            total_fail += 1
            print(f"  [{i+1:3d}/{len(mods)}] SKIP {mod_id:30s} no solvers.py")
            continue

        code = TEST_CODE.format(
            root=str(ROOT).replace("\\", "/"),
            mod_id=mod_id,
            n_runs=N_RUNS,
        )

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=1800,
            )
            dt = time.time() - t0

            if result.returncode == 0:
                # Find JSON in output (last line)
                output_lines = result.stdout.strip().split('\n')
                json_line = output_lines[-1] if output_lines else '{}'
                try:
                    data = json.loads(json_line)
                except json.JSONDecodeError:
                    data = {"error": f"bad_json: {json_line[:100]}", "solvers": {}}
            else:
                err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown"
                data = {"error": err[:200], "solvers": {}}

        except subprocess.TimeoutExpired:
            dt = 600
            data = {"error": "TIMEOUT", "solvers": {}}

        all_results[mod_id] = data

        n_s = len(data.get("solvers", {}))
        n_d = sum(1 for v in data.get("solvers", {}).values() if v.get("status") == "done")
        total_solvers += n_s
        total_solvers_done += n_d

        if data.get("error"):
            total_fail += 1
            err_msg = str(data["error"])[:60]
            print(f"  [{i+1:3d}/{len(mods)}] ERR  {mod_id:30s} {err_msg}")
        else:
            total_done += 1
            print(f"  [{i+1:3d}/{len(mods)}] OK   {mod_id:30s} {n_d}/{n_s} done  {dt:.0f}s")

        # Save progress every 3 modalities
        if (i + 1) % 3 == 0 or i == len(mods) - 1:
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Final save
    os.makedirs(RESULTS_FILE.parent, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Modalities: {total_done} done, {total_fail} failed, {len(mods)} total")
    print(f"Solvers: {total_solvers_done}/{total_solvers} ({100*total_solvers_done/max(total_solvers,1):.1f}%)")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
