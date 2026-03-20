# Cone-Beam Computed Tomography (CBCT) — ML-EM + Gradient Mismatch Correction

**Device**: CPU  **Input**: projections (angles × H × W, float32)
**Reference**: Shepp, L.A. & Vardi, Y. (1982) Maximum likelihood reconstruction for emission tomography, IEEE TMI


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| Source-detector geometry offset (mm) | `[-10, +10] mm` | ±2 mm per session | Phantom-based calibration; gradient descent on geometric projection error |

## Correction Parameters

```python
mismatch_cfg = {
    'source_detector_geometry_offset': None,  # None = auto-calibrate
    'search_range': '[-10, +10] mm',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('cbct_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # projections (angles × H × W, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cbct.solvers import run_solver
from pwm_core.mismatch.operators import cbct_calibrate_offset
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('mlem', y, cfg={})

# 2. Calibrate mismatch parameter
geo_offset = cbct_calibrate_offset(y)

# 3. Reconstruct WITH correction
cfg = {"geo_offset": float(geo_offset)}
cfg.update({})
x_corrected = run_solver('mlem', y, cfg=cfg)

# 4. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {compute_psnr(x_true, x_wrong):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_wrong):.4f}")
    print(f"Corrected      PSNR {compute_psnr(x_true, x_corrected):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_corrected):.4f}")

x = x_corrected
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
plt.tight_layout(); plt.savefig('cbct_mlem_corrected.png'); plt.show()
```

## Benchmark

`Cone-Beam Computed Tomography (CBCT) — ML-EM + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/cbct` (mismatch correction tier).
