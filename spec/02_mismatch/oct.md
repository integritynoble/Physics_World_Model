# Optical Coherence Tomography (OCT) — Mismatch Correction + Reconstruct

**Input**: spectrum (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`
**Algorithms**: 12 CPU, 4 GPU — reconstruct with `run_solver(key, y)`

## Mismatch

| Parameter | Range | Typical Error | Correction Method |
|-----------|-------|---------------|-------------------|
| dispersion coefficients (β₂, β₃) | `β₂ ∈ [-1e-27, 1e-27] s²/m` | ±2e-28 s²/m | numerical dispersion compensation (NDC) |

## User Parameters

```python
# Set known values or leave None to auto-calibrate
mismatch_cfg = {
    'disp_coeff': None,    # float if known; None = auto-estimate
    'search_range': 'β₂ ∈ [-1e-27, 1e-27] s²/m',
    'search_steps': 20,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('oct_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # spectrum (wavenumbers × A-scans, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.oct.solvers import run_solver, list_solvers
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Reconstruct WITHOUT mismatch correction (baseline)
x_wrong = run_solver('traditional_cpu', y)

# 2. Reconstruct WITH mismatch correction
x_corrected = run_solver('traditional_cpu', y, cfg=mismatch_cfg)

# 3. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {compute_psnr(x_true, x_wrong):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_wrong):.4f}")
    print(f"Corrected      PSNR {compute_psnr(x_true, x_corrected):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_corrected):.4f}")

# 4. Use any other solver
# list_solvers()
# x = run_solver('your_key', y, cfg=mismatch_cfg)

x = x_corrected  # for visualization below
# Visualize
import matplotlib.pyplot as plt
ncols = 3 if x_true is not None else 2
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
y_show = y if y.ndim == 2 else y[0] if y.ndim == 3 else y
axes[0].imshow(y_show, cmap='gray'); axes[0].set_title('Measurement')
axes[1].imshow(x, cmap='gray'); axes[1].set_title('Reconstruction')
if x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('oct_corrected.png'); plt.show()
```

## Reference

- `packages/pwm_core/contrib/mismatch_db.yaml` — mismatch parameter ranges
- `papers/system_design/` — system design with mismatch analysis
