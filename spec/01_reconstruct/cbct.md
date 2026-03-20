# CBCT — Cone-Beam CT Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

Cone-Beam CT uses a divergent X-ray beam (cone geometry) rotating around the object to acquire a 3D volume in one rotation. Unlike parallel-beam CT, the reconstruction must account for 3D cone geometry.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Geometry | Cone-beam (circular orbit) | 360 projections × 512 × 512 detector |
| 3D volume | Reconstructed | 256 × 256 × 256 voxels |
| Forward model | 3D cone-beam projector | y = A_cone · x + η |
| Noise model | Poisson + detector scatter | I₀ = 10⁴–10⁵ |
| Applications | Dental, IGRT, head & neck, extremities |

**Key differences from CT:**
- 3D reconstruction (volume, not single slice)
- Cone-beam artifacts (Feldkamp artifacts) at large cone angles
- Scatter is more severe due to larger illuminated volume
- Center-of-rotation alignment is 3D

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | FDK (Feldkamp) | ~32 dB | No | Standard CBCT reconstruction |
| `best_quality` | TV-ADMM 3D | ~37 dB | No | Iterative + TV; corrects cone artifacts |
| `sart_3d` | SART (3D) | 34 dB | No | Row-action iterative |
| `cnn_cbct` | CBCT-CNN | ~38 dB | Yes | 3D CNN post-processing |
| `fdk_dl` | FDK + DL | ~38 dB | Yes | FDK initialization + deep learning |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.cbct.solvers import run_solver, list_solvers

# Synthetic demo: random cone-beam projections
n_proj, det_h, det_w = 60, 256, 256
y = np.random.rand(n_proj, det_h, det_w).astype(np.float32)

print("CBCT solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'
x_hat = run_solver(SOLVER, y, operator=None, cfg={})
print(f"3D volume shape: {x_hat.shape}")  # Expected: (256, 256, 256) or similar

# Visualize 3 orthogonal slices
if x_hat.ndim == 3:
    d, h, w = x_hat.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x_hat[d//2], cmap='gray'); axes[0].set_title('Axial')
    axes[1].imshow(x_hat[:, h//2, :], cmap='gray'); axes[1].set_title('Coronal')
    axes[2].imshow(x_hat[:, :, w//2], cmap='gray'); axes[2].set_title('Sagittal')
    plt.savefig('cbct_reconstruction.png', dpi=150, bbox_inches='tight'); plt.show()
```

---

## References

- **FDK**: Feldkamp, Davis & Kress, JOSA A 1984
- **TV-ADMM 3D**: Sidky & Pan, Phys Med Biol 2008 (extended to 3D)
