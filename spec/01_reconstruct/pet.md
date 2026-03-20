# PET — Positron Emission Tomography Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

PET detects gamma-ray pairs from positron annihilation events. The measurement is a **sinogram** of coincidence counts. Reconstruction recovers the radiotracer distribution (functional/metabolic image).

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Geometry | Ring detector (parallel-beam sinogram) | 180 angles × 256 detectors |
| Forward model | Radon transform + Poisson noise | y ~ Poisson(A·x + scatter + randoms) |
| Noise model | **Poisson** (photon-limited) | Low counts → high noise |
| Applications | Oncology (FDG), neurology (amyloid PET), cardiology |
| Image size | 256 × 256 (2D) |

**Key considerations:**
- Always Poisson noise → requires maximum-likelihood estimation (OSEM/MLEM), not least-squares
- Attenuation correction needed (often from co-registered CT)
- Scatter and random coincidences must be corrected
- Resolution is limited by positron range (~1–2 mm) and detector size

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | FBP (proxy) | ~22 dB | No | Not recommended for PET (Poisson) |
| `best_quality` | OSEM | ~30 dB | No | Standard clinical PET reconstruction |
| `mlem` | MLEM | 28 dB | No | Maximum-likelihood EM; slow |
| `osem` | OSEM | **30 dB** | No | Ordered-subset EM; standard of care |
| `map_em` | MAP-EM (TV prior) | 32 dB | No | Regularized OSEM |
| `pet_net` | PET-Net | ~35 dB | Yes | DL post-processing |
| `pet_unrolled` | PET Unrolled | ~36 dB | Yes | DL unrolled EM |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.pet.solvers import run_solver, list_solvers

# Load or simulate sinogram (n_angles, n_det)
y = np.random.poisson(20, (180, 256)).astype(np.float32)   # Simulated PET counts

print("PET solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'best_quality'  # OSEM — recommended for PET
x_hat = run_solver(SOLVER, y, operator=None, cfg={'iters': 10, 'n_subsets': 10})
print(f"PET image shape: {x_hat.shape}")

plt.imshow(x_hat, cmap='hot'); plt.colorbar(); plt.title(f'PET ({SOLVER})')
plt.savefig('pet_reconstruction.png', dpi=150, bbox_inches='tight'); plt.show()
```

---

## References

- **MLEM**: Shepp & Vardi, IEEE TMI 1982
- **OSEM**: Hudson & Larkin, IEEE TMI 1994
- **MAP-EM**: Green, IEEE TMI 1990
