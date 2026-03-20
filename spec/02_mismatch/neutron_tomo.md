# Neutron Radiography / Tomography — Mismatch Correction + Reconstruct

**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/`
**Algorithms**: 11 CPU, 4 GPU — reconstruct with `run_solver(key, y)`

## Mismatch

| Parameter | Range | Typical Error | Correction Method |
|-----------|-------|---------------|-------------------|
| operator model error | `modality-dependent` | calibration dependent | grid search on reconstruction quality |

## User Parameters

```python
# Set known values or leave None to auto-calibrate
mismatch_cfg = {
    'mismatch_param': None,    # float if known; None = auto-estimate
    'search_range': 'modality-dependent',
    'search_steps': 20,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('neutron_tomo_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # projections (angles × H × W, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.neutron_tomo.solvers import run_solver, list_solvers
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
# Visualize (3D: orthogonal slices)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
mid = [s // 2 for s in x.shape]
axes[0, 0].imshow(x[mid[0]], cmap='gray'); axes[0, 0].set_title('Recon axial')
axes[0, 1].imshow(x[:, mid[1], :], cmap='gray'); axes[0, 1].set_title('Recon coronal')
axes[0, 2].imshow(x[:, :, mid[2]], cmap='gray'); axes[0, 2].set_title('Recon sagittal')
if x_true is not None:
    axes[1, 0].imshow(x_true[mid[0]], cmap='gray'); axes[1, 0].set_title('GT axial')
    axes[1, 1].imshow(x_true[:, mid[1], :], cmap='gray'); axes[1, 1].set_title('GT coronal')
    axes[1, 2].imshow(x_true[:, :, mid[2]], cmap='gray'); axes[1, 2].set_title('GT sagittal')
plt.tight_layout(); plt.savefig('neutron_tomo_corrected.png'); plt.show()
```

## Reference

- `packages/pwm_core/contrib/mismatch_db.yaml` — mismatch parameter ranges
- `papers/system_design/` — system design with mismatch analysis
