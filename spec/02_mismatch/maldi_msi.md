# MALDI Mass Spectrometry Imaging — Mismatch Correction + Reconstruct

**Input**: mass image (H × W × m/z, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/public/`
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
with h5py.File('maldi_msi_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # mass image (H × W × m/z, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.maldi_msi.solvers import run_solver, list_solvers
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
# Visualize (hyperspectral: spatial slice + spectral profile)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
n_bands = x.shape[-1] if x.ndim == 3 else x.shape[0]
band = n_bands // 2
xb = x[..., band] if x.ndim == 3 else x[band]
axes[0].imshow(xb, cmap='gray'); axes[0].set_title(f'Recon band {band}')
axes[1].plot(x[x.shape[0]//2, x.shape[1]//2] if x.ndim==3 else x[:,x.shape[1]//2,x.shape[2]//2])
axes[1].set_title('Spectral profile (center pixel)')
if x_true is not None:
    xtb = x_true[..., band] if x_true.ndim == 3 else x_true[band]
    axes[2].imshow(xtb, cmap='gray'); axes[2].set_title('GT band')
plt.tight_layout(); plt.savefig('maldi_msi_corrected.png'); plt.show()
```

## Reference

- `packages/pwm_core/contrib/mismatch_db.yaml` — mismatch parameter ranges
- `papers/system_design/` — system design with mismatch analysis
