# MALDI Mass Spectrometry Imaging — System Design

## System DAG

```
[Source] → [Forward Model (MALDI Mass Spectrometry Imaging)] → [Detector] → y
      ↓                ↓
  [Noise]         [Mismatch]
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | MALDI Mass Spectrometry Imaging physics | operator model error |
| Detector | measurement | noise |

**Mismatch**: operator model error in range `modality-dependent`
**Correction**: grid search

## Reconstruction

**Dataset**: MALDI Mass Spectrometry Imaging
**Input**: mass image (H × W × m/z, float32)
**Algorithms**: 11 CPU, 4 GPU — see `spec/maldi_msi.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.maldi_msi.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('maldi_msi_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('traditional_cpu', y, cfg={'mismatch_param': None})  # None = auto-calibrate

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

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
plt.tight_layout(); plt.savefig('maldi_msi_recon.png'); plt.show()
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality maldi_msi --period forward --prompt "your system description"
python3 main.py --modality maldi_msi --period reconstruction --prompt "your algorithm"
```
