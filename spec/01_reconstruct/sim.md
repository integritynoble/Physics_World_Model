# SIM — Structured Illumination Microscopy Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

SIM achieves ~2× lateral super-resolution by illuminating the sample with sinusoidal patterns at multiple angles and phases, then using the Moiré effect to encode high-frequency information into the passband.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Raw frames | n_angles × n_phases | 9 frames (3 angles × 3 phases) |
| Forward model | Frequency-shifted OTF convolution | y_k = FT⁻¹{OTF(f-f_k) · X(f)} |
| Pixel size | ~65 nm (2× super-resolved) | Input: 130 nm, Output: 65 nm |
| Input size | Raw frame stack | (9, 512, 512) |
| Output | Super-resolved image | (1024, 1024) |
| Applications | Live-cell imaging, cytoskeleton, chromosomes |

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | GPU | Notes |
|-----------|-----------|------|-----|-------|
| `traditional_cpu` | Wiener-SIM | ~37 dB | No | Standard SIM reconstruction |
| `best_quality` | Fairsim-TV | ~39 dB | No | Open-source SIM |
| `wiener_sim` | Wiener-SIM | 37 dB | No | Wiener deconvolution in Fourier |
| `fairsim` | fairSIM | 38 dB | No | Robust to pattern errors |
| `dfcan` | DFCAN | **42 dB** | Yes | Deep frequency attention |
| `rcan_sim` | RCAN-SIM | 41 dB | Yes | Channel attention net |
| `ml_sim` | ML-SIM | 40 dB | Yes | Blind SIM (no PSF needed) |

> **CPU Best**: `wiener_sim` (37 dB) — fast, standard clinical SIM.
> **GPU Best**: `dfcan` (42 dB) — requires CUDA.

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from algorithm_base.sim.solvers import run_solver, list_solvers

# Raw SIM frames: (n_phases*n_angles, H, W) = (9, H, W)
y = np.random.rand(9, 512, 512).astype(np.float32)

print("SIM solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'  # Wiener-SIM
x_hat = run_solver(SOLVER, y, operator=None, cfg={})
print(f"Super-resolved image shape: {x_hat.shape}")  # (1024, 1024) or (512, 512)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(y[0], cmap='gray'); axes[0].set_title('SIM Frame 0 (raw)')
axes[1].imshow(x_hat, cmap='gray'); axes[1].set_title(f'Super-resolved ({SOLVER})')
plt.savefig('sim_reconstruction.png', dpi=150); plt.show()
```

---

## References

- **SIM theory**: Gustafsson, PNAS 2005
- **wienerSIM**: Wicker, Optics Express 2013
- **fairSIM**: Müller & Enderlein, Optica 2016
- **DFCAN**: Qiao et al., Nature Methods 2021
