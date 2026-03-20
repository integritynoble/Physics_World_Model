# Cone-Beam Computed Tomography (CBCT) — System Design

## System DAG

```
[X-ray Tube] → [Object] → [Cone Geometry] → [Flat Panel] → [ADC] → y
      ↓                         ↓                  ↓
 [Scatter]              [Geometry error]      [Lag artifact]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | Cone-Beam Computed Tomography (CBCT) physics | source-detector distance (SAD/SDD) |
| Detector | measurement | noise |

**Mismatch**: source-detector distance (SAD/SDD) in range `SAD ±5 mm, SDD ±10 mm`
**Correction**: phantom-based geometric calibration

## Reconstruction

**Dataset**: Cone-Beam Computed Tomography (CBCT)
**Input**: projections (angles × H × W, float32)
**Algorithms**: 17 CPU, 5 GPU — see `spec/cbct.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cbct.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('cbct_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('tv_admm', y, cfg={'geometry_error': None})  # None = auto-calibrate

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

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
plt.tight_layout(); plt.savefig('cbct_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality cbct --period forward --prompt "your system description"
python3 main.py --modality cbct --period reconstruction --prompt "your algorithm"
```
