# 05 — Hands-On Tutorial: Solar EUV/X-ray Imaging

This tutorial walks through running the PWM benchmark for Solar EUV/X-ray Imaging,
from loading data to computing metrics.

## Setup

```python
import sys
from pathlib import Path

ROOT = Path("/home/spiritai/abraham/pwm/production/Physics_World_Model")
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

import numpy as np
```

---

## 1. Loading the Benchmark Config

```python
import yaml

config_path = ROOT / "benchmarks" / "configs" / "solar_imaging.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print(f"Modality: {cfg['display_name']}")
print(f"Category: {cfg['category']}")
print(f"Forward model: {cfg['forward_model_type']}")
print(f"Default solver: {cfg['default_solver']}")
print(f"Image shape: {cfg['x_shape']}")
```

---

## 2. Understanding the Forward Model

```python
# The forward model type determines how measurements relate to the object
fwd_type = cfg["forward_model_type"]
cat_module = cfg["category_module"]

print(f"Forward model type: {fwd_type}")
print(f"Category module: {cat_module}")

# Mismatch parameters define the physics errors
for p in cfg.get("mismatch_params", []):
    print(f"  {p['name']}: nominal={p['nominal']}, "
          f"range={p['range']}, unit={p['unit']}")
```

---

## 3. Running the Default Solver

```python
# Import the traditional CPU solver
try:
    from pwm_core.recon.adjoint import run_adjoint
    print("Solver loaded: Adjoint")
except ImportError:
    print("Solver not available — install required dependencies")
```

---

## 4. Running the Benchmark

```python
# Use the expanded benchmark runner
# This handles data loading, solver execution, and metric computation

# Command-line usage:
# python benchmarks/runners/run_expanded.py --modality solar_imaging

# Or programmatically:
from benchmarks.framework.metrics import compute_psnr, compute_ssim

# After obtaining x_true and x_hat:
# psnr = compute_psnr(x_true, x_hat, max_val=1.0)
# ssim = compute_ssim(x_true, x_hat, data_range=1.0)
# print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
```

---

## 5. Comparing Solvers

```python
# Compare traditional vs best quality solver
solvers_to_test = {
    "Adjoint": ("pwm_core.recon.adjoint", "run_adjoint"),
    "PnP-ADMM": ("pwm_core.recon.pnp_admm", "pnp_admm_recon"),
}

# results = {}
# for name, (module, func) in solvers_to_test.items():
#     x_hat = run_solver(module, func, y, physics)
#     psnr = compute_psnr(x_true, x_hat, max_val=1.0)
#     ssim = compute_ssim(x_true, x_hat, data_range=1.0)
#     results[name] = {"psnr": psnr, "ssim": ssim}
#     print(f"{name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
```

---

## 6. Visualising Results

```python
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(x_true, cmap="gray")
# axes[0].set_title("Ground Truth")
# axes[1].imshow(x_hat_trad, cmap="gray")
# axes[1].set_title("Adjoint")
# axes[2].imshow(x_hat_best, cmap="gray")
# axes[2].set_title("PnP-ADMM")
# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.savefig("solar_imaging_comparison.png", dpi=150)
# plt.show()
```

---

## 7. Next Steps

- Read the full config: `benchmarks/configs/solar_imaging.yaml`
- Explore the expanded config: `benchmarks/expanded_configs/solar_imaging_expanded.yaml`
- Compare across tiers to see mismatch impact
- Add your own solver to the benchmark

---

*Previous: [04 — PWM Benchmark](04_pwm_benchmark.md)*
*Back to: [README](README.md)*
