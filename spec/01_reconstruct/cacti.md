# CACTI — Compressed Video Reconstruction (Compressed Ultrafast Photography)

> **Use Case 1: Reconstruct with specific algorithms**

---

## System Overview

CACTI (Content-Aware Compressed Temporal Imaging) captures a video sequence compressed into a single 2D snapshot by modulating frames with a physical binary mask that shifts each frame (CACTI hardware variant).

| Parameter | Description | Value |
|-----------|-------------|-------|
| Measurement | 2D compressed snapshot | (H, W) = (256, 256) |
| Video frames | Temporal frames to recover | B = 8–16 frames |
| Forward model | Masked temporal integration | y = sum_{t=1}^B M_t ⊙ x_t |
| Mask | Binary random mask (shifts per frame) | (H, W) per frame |
| Applications | Fluorescence microscopy, high-speed events |

---

## Algorithm Catalog

| Solver Key | Algorithm | PSNR | SSIM | GPU |
|-----------|-----------|------|------|-----|
| `traditional_cpu` | GAP-TV | ~35 dB | ~0.93 | No |
| `best_quality` | EfficientSCI | ~40 dB | ~0.96 | Yes |
| `gap_tv` | GAP-TV | 35 dB | 0.93 | No |
| `admm_tv` | ADMM-TV | 34 dB | 0.92 | No |
| `twist` | TwIST | 33 dB | 0.91 | No |
| `efficient_sci` | EfficientSCI | **40 dB** | **0.96** | Yes |
| `rev_sci` | RevSCI | 38 dB | 0.95 | Yes |
| `dense_net` | DenseNet-SCI | 37 dB | 0.95 | Yes |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from algorithm_base.cacti.solvers import run_solver, list_solvers

# Compressed snapshot: (H, W)
y = np.random.rand(256, 256).astype(np.float32)

print("CACTI solvers:"); [print(f"  {k}: {v['name']}") for k, v in list_solvers()]

SOLVER = 'traditional_cpu'  # GAP-TV
x_hat = run_solver(SOLVER, y, operator=None, cfg={'B': 8, 'iters': 50})

if x_hat.ndim == 3:  # (B, H, W) video frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat[:x_hat.shape[0]]):
        ax.imshow(x_hat[i], cmap='gray'); ax.set_title(f'Frame {i+1}')
    plt.suptitle(f'CACTI Reconstruction ({SOLVER})')
else:
    plt.imshow(x_hat, cmap='gray'); plt.title(f'CACTI ({SOLVER})')
plt.savefig('cacti_reconstruction.png', dpi=150); plt.show()
```

---

## References

- **GAP-TV**: Liao et al., ICIP 2014
- **EfficientSCI**: Wang et al., CVPR 2023
- **RevSCI**: Cheng et al., CVPR 2021
