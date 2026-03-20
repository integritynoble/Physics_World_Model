# Single Photon Emission CT (SPECT) — System Design

## System DAG

```
[Radiotracer] → [Collimator] → [NaI Crystal] → [PMT Array] → y
                     ↓                ↓
              [Septal penetration] [Scatter 25%]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | Single Photon Emission CT (SPECT) physics | scatter fraction |
| Detector | measurement | noise |

**Mismatch**: scatter fraction in range `[0.1, 0.4]`
**Correction**: triple-energy window (TEW) subtraction

## Reconstruction

**Dataset**: Single Photon Emission CT (SPECT)
**Input**: projections (angles × detectors, float32)
**Algorithms**: 9 CPU, 6 GPU — see `spec/spect.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.spect.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('spect_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('os_em', y, cfg={'scatter_frac': None})  # None = auto-calibrate

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

# Visualize
import matplotlib.pyplot as plt
ncols = 3 if x_true is not None else 2
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
y_show = y if y.ndim == 2 else y[0] if y.ndim == 3 else y
axes[0].imshow(y_show, cmap='gray'); axes[0].set_title('Measurement')
axes[1].imshow(x, cmap='gray'); axes[1].set_title('Reconstruction')
if x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('spect_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality spect --period forward --prompt "your system description"
python3 main.py --modality spect --period reconstruction --prompt "your algorithm"
```
