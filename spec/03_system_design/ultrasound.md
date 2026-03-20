# Ultrasound B-mode Imaging — System Design

## System DAG

```
[Probe Array] → [Pulse-Echo] → [Tissue] → [RF Sampling] → y
                                   ↓              ↓
                           [Speed of sound]  [Reverberation]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | Ultrasound B-mode Imaging physics | sound speed (m/s) |
| Detector | measurement | noise |

**Mismatch**: sound speed (m/s) in range `[1400, 1600] m/s`
**Correction**: coherence-based speed estimation

## Reconstruction

**Dataset**: Ultrasound B-mode Imaging
**Input**: RF data (elements × samples, float32)
**Algorithms**: 11 CPU, 6 GPU — see `spec/ultrasound.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ultrasound.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('ultrasound_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('tv_admm', y, cfg={'c0': None})  # None = auto-calibrate

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
plt.tight_layout(); plt.savefig('ultrasound_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality ultrasound --period forward --prompt "your system description"
python3 main.py --modality ultrasound --period reconstruction --prompt "your algorithm"
```
