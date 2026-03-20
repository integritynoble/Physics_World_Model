# Ultrasound — B-mode Imaging Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

Ultrasound imaging transmits acoustic pulses and measures back-scattered echoes. B-mode reconstruction beamforms the received echoes into a 2D image.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Forward model | Linear wave propagation | y = A · x + η |
| Beamforming | Delay-and-Sum (DAS) | Standard baseline |
| Transducer | Linear array | 128 elements, 5 MHz |
| Noise | Speckle (coherent) + thermal | — |
| Image | B-mode image | (256, 256) gray |

**Key considerations:**
- Speckle is signal (coherent) — not independent noise; regularization must preserve it
- Lateral resolution is limited by aperture; axial by bandwidth
- Coherent compounding (multiple angles) improves image quality
- Total variation regularization can over-smooth speckle

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | DAS Beamforming | ~28 dB | No | Standard clinical |
| `best_quality` | MV Beamforming + TV | ~33 dB | No | Minimum-variance + TV |
| `das` | Delay-and-Sum | 28 dB | No | Baseline |
| `mv` | Minimum Variance | 31 dB | No | Adaptive beamformer |
| `coherence_factor` | CF Beamforming | 30 dB | No | Coherence-weighted |
| `cs_us` | Compressed Sensing US | 32 dB | No | Fewer transmissions |
| `dl_us` | DL Beamforming | ~36 dB | Yes | Learned beamformer |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.ultrasound.solvers import run_solver, list_solvers

# Raw RF data (n_elements × n_samples) — replace with real data
y = np.random.rand(128, 2048).astype(np.float32)

print("Ultrasound solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'
x_hat = run_solver(SOLVER, y, operator=None, cfg={})
print(f"B-mode image shape: {x_hat.shape}")

plt.imshow(x_hat, cmap='gray', aspect='auto')
plt.title(f'Ultrasound B-mode ({SOLVER})')
plt.xlabel('Lateral'); plt.ylabel('Depth')
plt.savefig('ultrasound_reconstruction.png', dpi=150); plt.show()
```

---

## References

- **DAS**: Synnevag et al., IEEE TUFFC 2007
- **Minimum Variance**: Capon, Proc. IEEE 1969 (adapted for US)
- **CS Ultrasound**: Quinsac et al., Ultrasonics 2012
