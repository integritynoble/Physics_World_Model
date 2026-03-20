# Structured Illumination Microscopy (SIM) — System Design

> Source: `papers/system_design/outputs/sim_forward_v1_iter1.md` + `sim_reconstruction_v1_iter1.md`

## System DAG

```
[488nm Laser] → [SLM: 3 orient. × 3 phases] → [Fluorescent Sample]
                                                        ↓
                                              [1.4 NA Objective (PSF)]
                                                        ↓
                                               [sCMOS Detector] → [16-bit ADC] → y
                                                        ↓
                                                [Poisson + readout σ=1.5 e⁻]
```

## Key Mismatch Sources

- illumination pattern phase offset: `[0, 2π/3] rad`

## Reconstruction

**Algorithms**: 12 CPU, 4 GPU
**Best CPU**: `wienersim`
See `spec/sim.md` for full algorithm table.

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.sim.solvers import run_solver, list_solvers
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('sim_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Simulate mismatch then correct
x = run_solver('wienersim', y, cfg={'phase_offset': None})

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
plt.tight_layout(); plt.savefig('sim_recon.png'); plt.show()
```

## Papers

- Forward model: `papers/system_design/outputs/sim_forward_v1_iter1.md`
- Reconstruction: `papers/system_design/outputs/sim_reconstruction_v1_iter1.md`
- Multi-agent: `python3 papers/system_design/main.py --modality sim --period forward`
