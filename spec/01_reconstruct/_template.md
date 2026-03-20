# {MODALITY_NAME} — Reconstruction Spec (Template)

> **Use Case 1: Reconstruct with specific algorithms**
> Replace `{MODALITY_ID}` with the actual modality ID from `algorithm_base/`.

---

## System Overview

*(Describe the imaging system and the forward model. Explain what measurement `y` is,
what image `x` we want to recover, and what physics governs the mapping y = A(x) + noise.)*

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Modality ID | `{MODALITY_ID}` | — |
| Forward model | `y = A(x) + η` | — |
| Noise model | Gaussian / Poisson / Speckle | — |
| Measurement shape | Input `y` shape | — |
| Output shape | Reconstructed `x` shape | Usually (256, 256) |
| Dataset | Built-in dataset or user-provided | PWM benchmark |

---

## Algorithm Catalog

*(List the solvers from `algorithm_base/{modality_id}/solvers.py`)*

```python
# List all available solvers dynamically:
import sys
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public')
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public/packages/pwm_core')
from algorithm_base.{MODALITY_ID}.solvers import list_solvers
for key, info in list_solvers():
    gpu = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {key:<25} {info['name']:<35} {gpu}  {info.get('reference','')}")
```

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | *(first solver)* | — | No | Baseline |
| `best_quality` | *(best solver)* | — | — | Best quality |

---

## Measurement Data

### Option A: PWM Benchmark Data (recommended for evaluation)
*(Replace with actual GCS path if available)*
```python
import gcsfs, h5py, numpy as np
fs = gcsfs.GCSFileSystem(token='anon')
with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY_ID}/public/{MODALITY_ID}_public.h5') as f:
    with h5py.File(f, 'r') as hf:
        y      = hf['y'][0]       # measurement
        x_true = hf['x_true'][0]  # ground truth (if available)
```

### Option B: Your Own Data
```python
y = np.load('your_measurement.npy').astype(np.float32)
# Shape should match what the modality expects
# x_true = np.load('your_gt.npy') if you have ground truth
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iters` | 50 | Number of iterations |
| `lam` | 0.01 | Regularization weight |

---

## Run Button

```python
# ============================================================
# {MODALITY_NAME} Reconstruction — PWM Run Button
# ============================================================
import sys, os

# --- Setup ---
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Colab: BASE = '/content/Physics_World_Model/pwm/public'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.{MODALITY_ID}.solvers import run_solver, list_solvers

# --- Load data ---
y = np.random.rand(256, 256).astype(np.float32)  # Replace with real data
x_true = None; has_gt = False

# --- List solvers ---
print("Available solvers:")
for key, info in list_solvers():
    gpu = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {key:<25} {info['name']:<30} {gpu}  {info.get('reference','')}")

# --- Run ---
SOLVER = 'traditional_cpu'   # Change to any solver key above
cfg    = {}                   # Add parameters: {'iters': 50, 'lam': 0.01}

print(f"\nRunning {SOLVER} ...")
x_hat = run_solver(SOLVER, y, operator=None, cfg=cfg)
print(f"Output shape: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")

# --- Evaluate ---
if has_gt and x_true is not None:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    ref_max = max(x_true.max(), x_hat.max())
    psnr = peak_signal_noise_ratio(x_true, x_hat, data_range=ref_max)
    ssim = structural_similarity(x_true, x_hat, data_range=ref_max)
    print(f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")

# --- Visualize ---
n_panels = 3 if has_gt else 2
fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
axes[0].imshow(y if y.ndim == 2 else y[0], cmap='gray')
axes[0].set_title('Measurement (y)')
axes[1].imshow(x_hat if x_hat.ndim == 2 else x_hat[0], cmap='gray')
axes[1].set_title(f'Reconstruction ({SOLVER})')
if has_gt and x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('{MODALITY_ID}_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Expected Output

After running, you will see:
1. **Reconstructed image** — shape depends on modality
2. **PSNR** (if ground truth provided)
3. **SSIM** (if ground truth provided)
4. **Saved**: `{MODALITY_ID}_reconstruction.png`

---

## References

*(Add relevant citations for the algorithms and dataset)*
