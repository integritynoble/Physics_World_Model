#!/usr/bin/env python3
"""Test CT solvers with their default iterations (from SOLVERS cfg_override)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from algorithm_base.ct.solvers import SOLVERS

SAMPLE = ROOT / "datasets" / "benchmark" / "ct" / "standard" / "standard_ct_00.h5"

# Test these solvers with their default cfg_override iterations
TEST_KEYS = [
    "pnp_admm_nlm", "pnp_hqs_nlm", "pnp_fista_nlm", "pnp_admm_bm3d",
]

TEST_CODE = '''
import sys, gc
sys.path.insert(0, "{root}")
import numpy as np, h5py
from skimage.metrics import peak_signal_noise_ratio as psnr
from algorithm_base.ct.solvers import CTOperator, run_solver
with h5py.File("{sample}", "r") as f:
    x_true = np.array(f["x_true"], dtype=np.float32)
    y = np.array(f["y_ideal"], dtype=np.float32)
op = CTOperator(y.shape[0], y.shape[1], 362)
cfg = {{"output_size": 362}}
x_hat = run_solver("{key}", y, op, cfg)
assert x_hat.shape == x_true.shape
assert np.isfinite(x_hat).all()
p = psnr(x_true, np.clip(x_hat, 0, 1), data_range=1.0)
print(f"PASS PSNR={{p:.2f}}")
'''

for key in TEST_KEYS:
    if key not in SOLVERS:
        continue
    spec = SOLVERS[key]
    code = TEST_CODE.format(root=str(ROOT).replace("\\", "/"),
                           sample=str(SAMPLE).replace("\\", "/"),
                           key=key)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            psnr_line = [l for l in result.stdout.strip().split("\n") if "PASS" in l]
            psnr_val = psnr_line[0] if psnr_line else "?"
            print(f"  PASS {key:25s} {spec['name']:25s} {psnr_val}")
        else:
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
            print(f"  FAIL {key:25s} {spec['name']:25s} {err[:80]}")
    except subprocess.TimeoutExpired:
        print(f"  FAIL {key:25s} {spec['name']:25s} TIMEOUT")
