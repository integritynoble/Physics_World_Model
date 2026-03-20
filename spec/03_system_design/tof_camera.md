# Time-of-Flight Depth Camera — System Design

## System DAG

```
[Source] → [Forward Model (Time-of-Flight Depth Camera)] → [Detector] → y
      ↓                ↓
  [Noise]         [Mismatch]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | Time-of-Flight Depth Camera physics | operator model error |
| Detector | measurement | noise |

**Mismatch**: operator model error in range `modality-dependent`
**Correction**: grid search

## Reconstruction

**Dataset**: Time-of-Flight Depth Camera
**Input**: depth + amplitude (H × W × 2, float32)
**Algorithms**: 11 CPU, 4 GPU — see `spec/tof_camera.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.tof_camera.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('tof_camera_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/tof_camera/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('traditional_cpu', y, cfg={'mismatch_param': None})  # None = auto-calibrate

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
plt.tight_layout(); plt.savefig('tof_camera_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality tof_camera --period forward --prompt "your system description"
python3 main.py --modality tof_camera --period reconstruction --prompt "your algorithm"
```
