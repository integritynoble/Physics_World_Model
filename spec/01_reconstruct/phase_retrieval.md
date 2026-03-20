# Phase Retrieval — Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

Phase retrieval reconstructs a complex wavefield x from intensity measurements |Ax|² (only magnitude is measured). This is the core problem in X-ray crystallography, coherent X-ray imaging, and optical phase retrieval.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Measurement | Intensity (squared magnitude) | y = |Ax|² + η |
| Forward model | Fourier magnitude (oversampled) | A = FFT (with oversampling ratio σ ≥ 2) |
| Oversampling | Required for unique recovery | σ ≥ 2 in each dimension |
| Input | n_diffraction_patterns × (H × W) | e.g., 4 × 64 × 64 |
| Output | Complex image | (H, W) complex64 |

**Key considerations:**
- **Phase problem**: only |FFT(x)| is measured; phase is unknown
- **Oversampling**: provides unique recovery if σ ≥ 2 (for real objects)
- **Ptychography**: solves phase retrieval with scanning overlapping probes
- **Diversity**: multiple diverse measurements improve convergence

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | HIO + ER | ~36 dB | No | Classic iterative phase retrieval |
| `best_quality` | rPIE | ~38 dB | No | Relaxed PIE; robust |
| `hio_er` | HIO+ER | 36 dB | No | Hybrid input-output + error reduction |
| `raar` | RAAR | 37 dB | No | Relaxed averaged alternating reflections |
| `rpie` | rPIE | 38 dB | No | Extended and regularized |
| `dm` | Difference Map | 36 dB | No | Primal-dual projections |
| `prox_phase` | Proximal Gradient | 37 dB | No | Smooth + non-smooth split |
| `phasepack_dl` | PhaseNet | ~41 dB | Yes | DL phase retrieval |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from algorithm_base.phase_retrieval.solvers import run_solver, list_solvers

# Diffraction intensities: (n_patterns, H, W) float32
y = np.abs(np.fft.fft2(np.random.rand(64, 64) + 1j*np.random.rand(64, 64)))**2
y = y[np.newaxis].repeat(4, axis=0).astype(np.float32)

print("Phase retrieval solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'  # HIO + ER
x_hat = run_solver(SOLVER, y, operator=None, cfg={'iters': 200, 'oversampling': 2})
print(f"Recovered phase shape: {x_hat.shape}")

# Show amplitude and phase
if np.iscomplexobj(x_hat):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.abs(x_hat), cmap='gray'); axes[0].set_title('Amplitude')
    axes[1].imshow(np.angle(x_hat), cmap='hsv'); axes[1].set_title('Phase')
else:
    plt.imshow(x_hat, cmap='gray'); plt.title(f'Phase Retrieval ({SOLVER})')
plt.savefig('phase_retrieval.png', dpi=150); plt.show()
```

---

## References

- **HIO**: Fienup, Applied Optics 1982
- **RAAR**: Luke, Inverse Problems 2004
- **rPIE**: Maiden & Rodenburg, Ultramicroscopy 2009
- **Difference Map**: Elser, JOSA A 2003
