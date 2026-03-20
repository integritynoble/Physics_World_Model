# SPECT — Single Photon Emission CT Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

SPECT detects single gamma photons using a rotating gamma camera with a collimator. The measurement is a 2D sinogram (or set of 2D projections for 3D SPECT). The reconstruction recovers the radiotracer distribution.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Geometry | Rotating gamma camera | 64 angles × 64 detectors |
| Forward model | Radon (parallel-hole collimator) | y = A · x + η |
| Noise | Poisson (lower count rates than PET) | ~100–1000 counts/pixel |
| Resolution | Limited by collimator | ~10–15 mm FWHM |
| Applications | Myocardial perfusion, bone scan, brain SPECT |

**Key differences from PET:**
- Single photon detection (no coincidence requirement) → simpler hardware
- Collimator provides directionality but reduces sensitivity (~100× lower than PET)
- No time-of-flight information
- Attenuation correction still needed (especially for cardiac SPECT)

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | FBP (ramp filter) | ~22 dB | No | Fast but noisy |
| `best_quality` | OSEM (16 subsets) | ~28 dB | No | Clinical standard |
| `mlem` | MLEM | 25 dB | No | Slow but unbiased |
| `osem` | OSEM | **28 dB** | No | Industry standard |
| `map_spect` | MAP-EM (TV) | 30 dB | No | Regularized OSEM |
| `dl_spect` | DL SPECT | ~33 dB | Yes | Post-processing CNN |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from algorithm_base.spect.solvers import run_solver, list_solvers

# Sinogram: (n_angles, n_detectors)
y = np.random.poisson(100, (64, 64)).astype(np.float32)

print("SPECT solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'best_quality'  # OSEM
x_hat = run_solver(SOLVER, y, operator=None, cfg={'iters': 5, 'n_subsets': 8})
print(f"SPECT image: {x_hat.shape}")

plt.imshow(x_hat, cmap='hot'); plt.colorbar(); plt.title(f'SPECT ({SOLVER})')
plt.savefig('spect_reconstruction.png', dpi=150); plt.show()
```

---

## References

- **OSEM**: Hudson & Larkin, IEEE TMI 1994
- **MAP-EM**: Green, IEEE TMI 1990; De Pierro, IEEE TMI 1995
