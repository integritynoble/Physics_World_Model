# Lensless Imaging — Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

Lensless cameras replace the lens with a diffuser or coded mask. The sensor directly records a
diffraction pattern (speckle). Reconstruction deconvolves the system PSF to recover the scene.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Forward model | Convolution with PSF | y = PSF ⊛ x + η |
| PSF | Diffuser point-spread function | Calibrated once off-line |
| Noise model | Poisson + Gaussian readout | Low-light capable |
| Image size | 270 × 270 (monochrome) or 270 × 270 × 3 (RGB) |
| Compression | Object area / sensor area | ~1:1 (no lens needed) |

**Key considerations:**
- PSF must be calibrated before reconstruction (see `02_mismatch_reconstruct/lensless_mismatch.md`)
- Near-field (Fresnel) vs. far-field (Fraunhofer) diffusion regime changes forward model
- Color (RGB) requires per-channel PSF or joint spectral model

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | ADMM-TV | ~32 dB | No | Good general baseline |
| `best_quality` | Wiener + ADMM | ~34 dB | No | Fast Wiener init + ADMM refinement |
| `admm_tv` | ADMM-TV | 32 dB | No | TV regularization |
| `rl_proxy` | Richardson-Lucy | 28 dB | No | Classic deconvolution |
| `fista_tv` | FISTA-TV | 31 dB | No | Fast proximal gradient |
| `physen_net` | PhysenNet | ~38 dB | Yes | Physics-informed neural net |
| `unrolled_admm` | Unrolled ADMM | ~36 dB | Yes | Learned proximal |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.lensless.solvers import run_solver, list_solvers

# Load calibrated PSF and measurement
# psf = np.load('psf.npy').astype(np.float32)      # shape (H, W)
# y   = np.load('measurement.npy').astype(np.float32)  # shape (H, W)

# Synthetic demo
H, W = 256, 256
psf = np.zeros((H, W), np.float32); psf[H//2-2:H//2+2, W//2-2:W//2+2] = 1.0
from scipy.ndimage import gaussian_filter
psf = gaussian_filter(psf, sigma=8); psf /= psf.sum()
y = np.random.rand(H, W).astype(np.float32)

print("Lensless solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'  # ADMM-TV
x_hat = run_solver(SOLVER, y, operator=None, cfg={'iters': 50, 'lam': 0.01})

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(y, cmap='gray'); axes[0].set_title('Lensless Measurement')
axes[1].imshow(x_hat, cmap='gray'); axes[1].set_title(f'Reconstruction ({SOLVER})')
plt.savefig('lensless_reconstruction.png', dpi=150, bbox_inches='tight'); plt.show()
```

---

## References

- **ADMM-TV**: Antipa et al., "DiffuserCam", Optica 2018
- **PhysenNet**: Wang et al., Nature Electronics 2020
- **Richardson-Lucy**: Richardson, JOSA 1972; Lucy, Astron J 1974
