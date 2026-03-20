# X-ray Computed Tomography (CT) — System Design

> Source: `papers/system_design/outputs/ct_forward_v1_iter1.md` + `ct_reconstruction_v1_iter1.md`

## System DAG

```
[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]
       ↓                      ↓                        ↓
  [Polychromatic         [Beer-Lambert           [CoR offset
   beam hardening]        attenuation]            mismatch]
                                                       ↓
                              → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y
                                        ↓
                                  [Poisson I0=1e4]
                                  [Gaussian σ=3 e⁻]
                                  [Dark current 0.05 e⁻/s]
```

## Key Mismatch Sources

- center-of-rotation (CoR) offset: `[-5, +5] px`

## Reconstruction

**Algorithms**: 26 CPU, 15 GPU
**Best CPU**: `pnp_admm_nlm`
See `spec/ct.md` for full algorithm table.

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ct.solvers import run_solver, list_solvers
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('ct_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Simulate mismatch then correct
x = run_solver('pnp_admm_nlm', y, cfg={'cor_offset': None})

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
plt.tight_layout(); plt.savefig('ct_recon.png'); plt.show()
```

## Papers

- Forward model: `papers/system_design/outputs/ct_forward_v1_iter1.md`
- Reconstruction: `papers/system_design/outputs/ct_reconstruction_v1_iter1.md`
- Multi-agent: `python3 papers/system_design/main.py --modality ct --period forward`
