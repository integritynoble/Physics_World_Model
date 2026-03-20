# Lattice Light-Sheet Microscopy — System Design

## System DAG

```
[Source] → [Forward Model (Lattice Light-Sheet Microscopy)] → [Detector] → y
      ↓                ↓
  [Noise]         [Mismatch]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | Lattice Light-Sheet Microscopy physics | operator model error |
| Detector | measurement | noise |

**Mismatch**: operator model error in range `modality-dependent`
**Correction**: grid search

## Reconstruction

**Dataset**: Lattice Light-Sheet Microscopy
**Input**: Z-stack (Z × H × W, float32)
**Algorithms**: 9 CPU, 6 GPU — see `spec/lattice_lightsheet.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.lattice_lightsheet.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('lattice_lightsheet_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('traditional_cpu', y, cfg={'mismatch_param': None})  # None = auto-calibrate

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
plt.tight_layout(); plt.savefig('lattice_lightsheet_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality lattice_lightsheet --period forward --prompt "your system description"
python3 main.py --modality lattice_lightsheet --period reconstruction --prompt "your algorithm"
```
