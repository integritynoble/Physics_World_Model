"""Test ALL solvers for all 8 modalities, 1 run each, to see PSNR landscape."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import h5py
import importlib
import inspect
import time

MODALITIES = ["spc", "lensless", "holography", "ptychography",
              "cbct", "ultrasound", "cryo_em", "widefield"]


def compute_psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 60.0
    return float(10 * np.log10(1.0 / mse))


for modality in MODALITIES:
    print(f"\n{'='*60}")
    print(f"  {modality}")
    print(f"{'='*60}")

    path = os.path.join(ROOT, "datasets", "benchmark", modality, "standard",
                        f"standard_{modality}_00.h5")
    try:
        with h5py.File(path, "r") as f:
            x_true = f["x_true"][:].astype(np.float32)
            y = f["y_ideal"][:].astype(np.float32)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue

    x_true_n = x_true / max(x_true.max(), 1e-8)
    y_n = y / max(y.max(), 1e-8)

    # Baseline: just using y as x_hat
    psnr_baseline = compute_psnr(x_true_n, np.clip(y_n, 0, 1))
    print(f"  Baseline (y=x_hat): PSNR={psnr_baseline:.2f} dB")

    try:
        mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
        solver_dict = mod.SOLVERS
    except Exception as e:
        print(f"  ERROR importing: {e}")
        continue

    sig = inspect.signature(mod.run_solver)
    params = list(sig.parameters.keys())

    for sk in solver_dict.keys():
        try:
            t0 = time.time()
            if 'physics' in params:
                x_hat = mod.run_solver(sk, y_n.copy(), physics=None, cfg={})
            else:
                x_hat = mod.run_solver(sk, y_n.copy(), operator=None, cfg={})
            elapsed = time.time() - t0
            x_hat = np.clip(np.asarray(x_hat, dtype=np.float32), 0, 1)
            if x_hat.shape != x_true_n.shape:
                x_hat = x_hat.reshape(x_true_n.shape) if x_hat.size == x_true_n.size else np.zeros_like(x_true_n)
            psnr = compute_psnr(x_true_n, x_hat)
            gpu = "GPU" if solver_dict[sk].get("gpu", False) else "CPU"
            print(f"  [{sk:25s}] PSNR={psnr:6.2f} dB  t={elapsed:5.1f}s  {gpu}  ({solver_dict[sk]['name']})")
        except Exception as e:
            print(f"  [{sk:25s}] FAIL: {str(e)[:80]}")

print("\nDone.")
