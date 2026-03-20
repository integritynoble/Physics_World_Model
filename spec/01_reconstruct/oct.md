# OCT — Optical Coherence Tomography Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

OCT measures depth-resolved tissue structure via low-coherence interferometry. Spectral-domain (SD-OCT) acquires interference spectra; Fourier transform of the spectrum gives A-scan depth profile.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Forward model | STFT / Fourier of interference spectrum | A-scan = FT(interference(k)) |
| Axial resolution | ~5–15 µm | Determined by source bandwidth |
| Lateral resolution | ~20 µm | Determined by objective NA |
| Noise | Shot noise + RIN | SNR ~100 dB dynamic range |
| Output | 2D B-scan or 3D volume | 1024 A-scans × 2048 pixels |

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | Standard FT (no resampling) | ~30 dB | No | Baseline |
| `best_quality` | Iterative Phase Unwrapping + TV | ~35 dB | No | Speckle reduction |
| `cs_oct` | Compressed Sensing OCT | 33 dB | No | Fewer A-scans |
| `dl_oct` | DL OCT Denoising | ~40 dB | Yes | CNN denoising |
| `sparse_oct` | Sparse Bayesian OCT | 34 dB | No | Bayesian deconvolution |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.oct.solvers import run_solver, list_solvers

# Interference spectrum: (n_wavenumbers, n_ascans)
y = np.random.rand(2048, 512).astype(np.float32)

print("OCT solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'
x_hat = run_solver(SOLVER, y, operator=None, cfg={})
print(f"B-scan shape: {x_hat.shape}")

plt.imshow(20*np.log10(np.abs(x_hat)+1e-6), cmap='gray', aspect='auto')
plt.title(f'OCT B-scan ({SOLVER}) [dB]')
plt.savefig('oct_reconstruction.png', dpi=150); plt.show()
```

---

## References

- **SD-OCT**: Wojtkowski et al., Optics Express 2004
- **Speckle reduction**: Schmitt, Opt. Lett. 1997; Pircher et al., JBO 2003
