# CASSI — Coded-Aperture Snapshot Spectral Imaging Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**
> Dataset: KAIST TSA Dataset | Platform: [pwm.platformai.org/benchmark/cassi](https://pwm.platformai.org/benchmark/cassi)

---

## System Overview

CASSI captures the entire 3D hyperspectral datacube (x, y, λ) in a **single 2D snapshot** by encoding spectral channels with a random coded aperture and dispersing them with a prism. Reconstruction recovers the (H, W, n_channels) spectral cube from the 2D measurement.

| Parameter | Description | Value (KAIST TSA) |
|-----------|-------------|-------------------|
| Measurement | 2D compressed snapshot | (H, W+shift) = (256, 310) |
| Datacube | Spectral cube to recover | (256, 256, 28 channels) |
| Spectral range | Visible wavelengths | 450–650 nm (28 bands) |
| Dispersion | Pixel shift per wavelength | 1 pixel/band |
| Coded aperture | Binary random mask | (256, 256) |
| SNR | Detector noise | ~40 dB |

**Physical factors affecting reconstruction:**
- **Dispersion step Δ**: must match physical prism dispersion — see `02_mismatch_reconstruct/cassi_mismatch.md`
- **Coded aperture pattern**: determines conditioning of the reconstruction problem
- **Number of spectral bands**: more bands → more underdetermined system

---

## Algorithm Catalog

| Solver Key | Algorithm | Reference PSNR | SSIM | GPU |
|-----------|-----------|----------------|------|-----|
| `traditional_cpu` | GAP-TV | ~38 dB | ~0.95 | No |
| `best_quality` | MST-L (Transformer) | ~42 dB | ~0.97 | **Yes** |
| `gap_tv` | GAP-TV | **38 dB** | 0.95 | No |
| `twist` | TwIST | 35 dB | 0.92 | No |
| `admm_tv` | ADMM-TV | 36 dB | 0.93 | No |
| `dnu` | DNU | 38.5 dB | 0.95 | Yes |
| `hdnet` | HDNet | 40.2 dB | 0.96 | Yes |
| `mst_s` | MST-S | 41 dB | 0.97 | Yes |
| `mst_l` | MST-L | **42 dB** | **0.97** | Yes |
| `cst_s` | CST-S | 40.5 dB | 0.96 | Yes |
| `dauhst_9stg` | DAUHST-9stg | 41.8 dB | 0.97 | Yes |

> **CPU Best**: `gap_tv` (38 dB) — no GPU needed, runs in ~5 minutes on CPU.
> **GPU Best**: `mst_l` (42 dB) — requires CUDA GPU with ≥4 GB VRAM.

---

## Measurement Data

### Option A: PWM Benchmark Data

```python
import gcsfs, h5py, numpy as np
fs = gcsfs.GCSFileSystem(token='anon')
with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/cassi_public.h5') as f:
    with h5py.File(f, 'r') as hf:
        y      = hf['y'][0]       # 2D measurement, shape (64, 92), float32
        mask   = hf['mask'][0]    # coded aperture, shape (64, 64), float32
        x_true = hf['x_true'][0]  # spectral cube, shape (64, 64, 28), float32
```

### Option B: Your Own Measurement

- **y shape**: `(H, W + (n_bands-1) * dispersion_step)` — 2D compressed measurement
- **mask shape**: `(H, W)` — binary coded aperture pattern
- **Values**: raw detector counts (float32)

```python
y    = np.load('cassi_measurement.npy').astype(np.float32)
mask = np.load('cassi_mask.npy').astype(np.float32)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step` | 1 | Dispersion step (pixels per wavelength band) |
| `iters` | 50 | Iterations for iterative solvers |
| `lam_tv` | 0.01 | TV regularization weight |
| `n_bands` | 28 | Number of spectral channels |

---

## Run Button

```python
# ============================================================
# CASSI Reconstruction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Colab: BASE = '/content/Physics_World_Model/pwm/public'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.cassi.solvers import run_solver, list_solvers

# --- Load data (synthetic demo) ---
H, W, n_bands = 64, 64, 28
rng = np.random.default_rng(42)
mask = rng.integers(0, 2, (H, W)).astype(np.float32)
# Simulate: y is (H, W + n_bands - 1) compressed measurement
y = rng.random((H, W + n_bands - 1)).astype(np.float32)

# --- List solvers ---
print("Available CASSI solvers:")
for key, info in list_solvers():
    gpu = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {key:<20} {info['name']:<30} {gpu}")

# --- Run ---
SOLVER = 'traditional_cpu'  # Change to: 'gap_tv', 'mst_l', etc.
cfg = {'step': 1, 'iters': 50}

print(f"\nRunning {SOLVER} ...")
x_hat = run_solver(SOLVER, y, operator=None, cfg=cfg)
print(f"Output shape: {x_hat.shape}")   # Expected: (H, W, n_bands) or (H, W)

# --- Visualize spectral cube ---
if x_hat.ndim == 3:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        band = int(i * x_hat.shape[2] / 8)
        ax.imshow(x_hat[:, :, band], cmap='inferno')
        ax.set_title(f'Band {band}')
    plt.suptitle(f'CASSI Reconstruction ({SOLVER}) — 8 of {x_hat.shape[2]} bands')
else:
    plt.imshow(x_hat, cmap='gray')
    plt.title(f'CASSI Reconstruction ({SOLVER})')
plt.tight_layout()
plt.savefig('cassi_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3D Datacube Visualization

The CASSI output is a (H, W, n_bands) spectral datacube. To visualize:

```python
# Interactive spectral explorer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.15)
im = ax.imshow(x_hat[:, :, 0], cmap='inferno')
ax.set_title('CASSI Spectral Cube')

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 'Band', 0, x_hat.shape[2]-1, valinit=0, valstep=1)

def update(val):
    im.set_data(x_hat[:, :, int(slider.val)])
    fig.canvas.draw_idle()
slider.on_changed(update)
plt.show()
```

---

## References

- **GAP-TV**: Liao et al., "Generalized Alternating Projection for Weighed-TV", ICIP 2014
- **MST**: Cai et al., "Mask-guided Spectral-wise Transformer", CVPR 2022
- **HDNet**: Hu et al., "HDNet: High-resolution Dual-domain Learning", CVPR 2022
- **Dataset**: Meng et al., "TSA-Net", ECCV 2020 (KAIST TSA dataset)
