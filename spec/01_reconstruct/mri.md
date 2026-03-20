# MRI — Magnetic Resonance Imaging Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**
> Dataset: [M4Raw](https://github.com/mylyu/M4Raw) | Platform: [pwm.platformai.org/benchmark/mri](https://pwm.platformai.org/benchmark/mri)

---

## System Overview

MRI measures incomplete Fourier coefficients (k-space) of the object. Undersampling k-space
(acceleration factor R) reduces scan time but creates aliasing artifacts that must be removed by
the reconstruction algorithm.

| Parameter | Description | Typical Value (PWM MRI Benchmark) |
|-----------|-------------|----------------------------------|
| Forward model | Fourier transform (SENSE) | y = F_u · S · x + η |
| k-space sampling | Cartesian undersampling | Acceleration R = 4× (25% of k-space) |
| Coils | Multi-channel receiver | 4–32 coils |
| Noise model | Gaussian (thermal noise) | SNR = 30–40 dB |
| Image size | Reconstructed | 128×128 (PWM) or 320×320 (fastMRI) |
| Dataset | M4Raw (multi-contrast brain MRI) | 308 subjects, 4 contrasts |

**Physical factors affecting reconstruction choice:**
- **Acceleration factor R**: higher R → more aliasing → stronger prior needed
- **Sampling pattern**: Cartesian (regular) vs. radial vs. spiral → different algorithms
- **Coil count**: parallel imaging (SENSE, ESPIRiT) exploits spatial diversity of coil arrays
- **Contrast**: T1/T2/FLAIR/DWI → different SNR and anatomy structure
- **Field strength**: 1.5T / 3T / 7T → different SNR and artifact profiles

---

## Algorithm Catalog

PWM provides **22 MRI solvers**. Run `list_solvers()` for the full list.

### CPU Solvers (no GPU required)

| Solver Key | Algorithm | Reference PSNR | SSIM | Notes |
|-----------|-----------|----------------|------|-------|
| `traditional_cpu` | Zero-Filled IFFT | ~25 dB | ~0.60 | Baseline; severe aliasing |
| `sense` | SENSE | ~30 dB | ~0.80 | Parallel imaging; needs coil maps |
| `espirit` | ESPIRiT | **34.2 dB** | **0.91** | **Best CPU (fastMRI knee 4×)** |
| `best_quality` | CS-MRI (Wavelet) | 33.0 dB | 0.88 | Compressed sensing; no GPU |
| `cs_tv` | CS-MRI (TV) | 31.5 dB | 0.86 | Total variation CS |
| `pocs` | POCS | 30.5 dB | 0.84 | Projection onto convex sets |
| `admm_mri` | ADMM | 32.0 dB | 0.87 | Split variable optimization |
| `conjugate_gradient` | Conjugate Gradient | 30.8 dB | 0.85 | Iterative least squares |
| `pnp_admm` | PnP-ADMM | 33.5 dB | 0.89 | Plug-and-play prior; no GPU |
| `low_rank` | LORAKS (Low-Rank) | 32.5 dB | 0.88 | Structured low-rank |
| `split_bregman` | Split Bregman | 31.8 dB | 0.86 | Equivalent to ADMM |
| `ista_mri` | ISTA | 30.0 dB | 0.83 | Iterative shrinkage |

### GPU Solvers (CUDA required)

| Solver Key | Algorithm | Reference PSNR | SSIM | Notes |
|-----------|-----------|----------------|------|-------|
| `varnet` | VarNet | 36.5 dB | 0.93 | Variational network |
| `reconformer` | ReconFormer | **37.82 dB** | **0.95** | **Best GPU quality** |
| `modl` | MoDL | 36.2 dB | 0.92 | Model-based deep learning |
| `e2e_varnet` | E2E-VarNet | 37.0 dB | 0.94 | End-to-end trained |
| `cascadenet` | Cascade-Net | 35.8 dB | 0.92 | Cascaded data consistency |
| `ista_net` | ISTA-Net+ | 35.0 dB | 0.91 | Learned ISTA |
| `dc_cnn` | DC-CNN | 34.5 dB | 0.90 | Data-consistent CNN |
| `k_interp` | KIKI-Net | 34.8 dB | 0.91 | k-space + image domain |

> **GPU Note**: GPU solvers require CUDA. CPU recommendation: `espirit` (34.2 dB) or
> `pnp_admm` (33.5 dB).

---

## Measurement Data

### Option A: PWM Benchmark Data (with ground truth → PSNR & SSIM)

```python
import gcsfs, h5py, numpy as np
fs = gcsfs.GCSFileSystem(token='anon')
with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/mri_public.h5') as f:
    with h5py.File(f, 'r') as hf:
        y      = hf['y'][0]       # k-space, shape (4, 128, 128) complex64 (coils × H × W)
        x_true = hf['x_true'][0]  # ground truth RSS image, shape (128, 128), float32
        mask   = hf['mask'][0]    # undersampling mask, shape (128, 128), float32
```

### Option B: Your Own k-space Data

- **Format**: NumPy array, shape `(n_coils, H, W)` complex64, OR `(H, W, 2)` real+imag float32
- **Values**: raw k-space measurements (no pre-processing needed)
- **Undersampling mask**: binary array `(H, W)` where 1 = sampled, 0 = not sampled

```python
y    = np.load('kspace.npy').astype(np.complex64)   # (n_coils, H, W) or (H, W, 2)
mask = np.load('mask.npy').astype(np.float32)        # (H, W), values 0 or 1
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acceleration` | 4 | Undersampling factor R (center fraction = 1/R) |
| `center_fraction` | 0.08 | Fraction of center k-space always sampled |
| `num_low_frequencies` | 24 | ACS (auto-calibration signal) lines for ESPIRiT |
| `regularization` | 0.01 | Regularization weight λ (CS, PnP) |
| `iters` | 50 | Number of iterations |
| `tolerance` | 1e-5 | Convergence tolerance |

---

## Run Button

```python
# ============================================================
# MRI Reconstruction — PWM Run Button
# ============================================================
import sys, os

# --- Setup (choose one) ---
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Colab: BASE = '/content/Physics_World_Model/pwm/public'

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.mri.solvers import run_solver, list_solvers

# --- 1. Load data ---
# Option A: PWM benchmark
# import gcsfs, h5py
# fs = gcsfs.GCSFileSystem(token='anon')
# with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/mri_public.h5') as f:
#     with h5py.File(f, 'r') as hf:
#         y = hf['y'][0]; x_true = hf['x_true'][0]; has_gt = True

# Option B: Your data
# y = np.load('kspace.npy').astype(np.complex64)
# x_true = None; has_gt = False

# Option C: Synthetic demo (4-coil, 4× accelerated)
H, W, n_coils = 128, 128, 4
# Simulate k-space: (n_coils, H, W) real/imag packed as (H, W, 2)
rng = np.random.default_rng(42)
x_phantom = rng.random((H, W)).astype(np.float32)
kspace_full = np.fft.fft2(x_phantom)
# 4× Cartesian undersampling: sample every 4th row + center
mask = np.zeros((H, W), dtype=np.float32)
mask[::4, :] = 1.0; mask[H//2 - 8 : H//2 + 8, :] = 1.0
kspace_us = kspace_full * mask
y = np.stack([np.real(kspace_us), np.imag(kspace_us)], axis=-1).astype(np.float32)
x_true = x_phantom; has_gt = True

# --- 2. List solvers ---
print("Available MRI solvers:")
for key, info in list_solvers():
    gpu = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {key:<25} {info['name']:<30} {gpu}")

# --- 3. Run reconstruction ---
SOLVER = 'traditional_cpu'   # Change to: 'espirit', 'best_quality', 'pnp_admm', etc.
cfg    = {}

print(f"\nRunning {SOLVER} ...")
x_hat = run_solver(SOLVER, y, operator=None, cfg=cfg)
# RSS combine if multi-coil output
if x_hat.ndim == 3:
    x_hat = np.sqrt(np.sum(x_hat**2, axis=0))
print(f"Output shape: {x_hat.shape}")

# --- 4. Evaluate ---
if has_gt and x_true is not None:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    ref_max = max(x_true.max(), x_hat.max()) if x_true.max() > 0 else 1.0
    psnr = peak_signal_noise_ratio(x_true, x_hat, data_range=ref_max)
    ssim = structural_similarity(x_true, x_hat, data_range=ref_max)
    print(f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")

# --- 5. Visualize ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Undersampled zero-fill reference
zf = np.abs(np.fft.ifft2(kspace_full * mask if has_gt else np.zeros((H,W))))
axes[0].imshow(zf, cmap='gray'); axes[0].set_title('Zero-Filled')
axes[1].imshow(x_hat, cmap='gray'); axes[1].set_title(f'Reconstruction ({SOLVER})')
if has_gt:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('mri_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mri_reconstruction.png")
```

---

## Expected Output

| Solver | Expected PSNR | Expected SSIM | GPU |
|--------|--------------|---------------|-----|
| `traditional_cpu` (Zero-Fill) | ~25 dB | ~0.60 | No |
| `best_quality` (CS-Wavelet) | ~33 dB | ~0.88 | No |
| `espirit` | ~34 dB | ~0.91 | No |
| `varnet` | ~36 dB | ~0.93 | Yes |
| `reconformer` | ~37.8 dB | ~0.95 | Yes |

---

## References

- **Dataset**: Lyu et al., "M4Raw: A Multi-Contrast / Multi-Repetition MRI Dataset", 2023
- **SENSE**: Pruessmann et al., Magnetic Resonance in Medicine 1999
- **ESPIRiT**: Uecker et al., MRM 2014
- **CS-MRI**: Lustig et al., MRM 2007
- **VarNet**: Sriram et al., MICCAI 2020
- **ReconFormer**: Guo et al., IEEE TMI 2023
- **MoDL**: Aggarwal et al., IEEE TMI 2019
