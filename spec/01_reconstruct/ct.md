# CT — X-ray Computed Tomography Reconstruction

> **Use Case 1: Reconstruct with specific algorithms**
> Dataset: [LoDoPaB-CT](https://lodopab.grand-challenge.org/) | Platform: [pwm.platformai.org/benchmark/ct](https://pwm.platformai.org/benchmark/ct)

---

## System Overview

X-ray CT measures how much tissue attenuates X-rays from many angles. The reconstruction
("inverse problem") recovers the 2D attenuation map from a set of line-integral projections (sinogram).

| Parameter | Description | Typical Value (LoDoPaB-CT) |
|-----------|-------------|---------------------------|
| Geometry | Parallel-beam | 1000 angles × 512 detectors |
| Image size | Reconstructed | 362 × 362 pixels |
| Forward model | Radon transform | y = Radon(x) + η |
| Noise model | Poisson (photon counting) + Gaussian readout | I₀ = 10⁵ photons/pixel |
| Dataset | LoDoPaB-CT (real clinical chest CTs) | 42,895 training / 3,553 test |
| Data range | Attenuation coefficient | 0–0.08 mm⁻¹ (normalized 0–1) |

**Physical factors affecting reconstruction choice:**
- **Dose** (high vs. low-dose): low-dose requires stronger regularization
- **View count** (full vs. sparse-view): sparse-view (≤64 angles) causes streak artifacts with FBP
- **Geometry** (parallel vs. fan/cone-beam): affects which FBP filter applies
- **Object size** (small vs. large FOV): truncation artifacts if object exceeds detector
- **Energy** (mono- vs. polychromatic): polychromatic causes beam hardening → correction needed

---

## Algorithm Catalog

PWM provides **41 CT solvers** spanning 1951–2026. Run `list_solvers()` to see all 41.

### CPU Solvers (no GPU required)

| Solver Key | Algorithm | Reference PSNR (LoDoPaB) | Speed | Notes |
|-----------|-----------|--------------------------|-------|-------|
| `traditional_cpu` | FBP (Ram-Lak) | ~27 dB | ★★★★★ | Classic baseline; fast but noisy at low dose |
| `fbp_shepp_logan` | FBP (Shepp-Logan) | ~27.2 dB | ★★★★★ | Smoother FBP filter |
| `sirt` | SIRT | 29.5 dB | ★★★ | Better than FBP; robust to noise |
| `cgls` | CGLS | 30.2 dB | ★★★ | Conjugate gradient; faster than SIRT |
| `sart` | SART | 29.1 dB | ★★★ | Row-action; good for sparse-view |
| `tv_admm` | TV-ADMM | 27.8 dB | ★★ | Edge-preserving; good for sparse-view |
| `chambolle_pock` | Chambolle-Pock | ~28 dB | ★★ | Primal-dual TV; stable |
| `pnp_admm_nlm` | PnP-ADMM (NLM) | **39.5 dB** | ★★ | **Best CPU quality** |
| `pnp_hqs_nlm` | PnP-HQS (NLM) | 39.1 dB | ★★ | Similar to PnP-ADMM |
| `famous_dl` | RED-CNN (CPU mode) | 33.2 dB | ★★ | DL post-processing; CPU-compatible |

### GPU Solvers (CUDA required)

| Solver Key | Algorithm | Reference PSNR (LoDoPaB) | Speed | Notes |
|-----------|-----------|--------------------------|-------|-------|
| `fbpconvnet` | FBPConvNet | 38.5 dB | ★★★★ | FBP + learned CNN |
| `wgan_vgg` | WGAN-VGG | 34.1 dB | ★★★ | GAN-based; perceptually sharp |
| `learn` | LEARN | 43.1 dB | ★★★ | Unrolled network |
| `learned_pd` | Learned Primal-Dual | 36.2 dB | ★★★ | Bilevel optimization network |
| `dudonet` | DuDoNet | 40.2 dB | ★★★ | Dual-domain (sinogram + image) |
| `indudonet` | InDuDoNet | **43.5 dB** | ★★ | **Best GPU quality** |
| `dudotrans` | DuDoTrans | 42.1 dB | ★★★ | Transformer-based dual-domain |
| `ctformer` | CTformer | 40.8 dB | ★★★ | Transformer in image domain |
| `score_ct` | Score-CT (diffusion) | 43.0 dB | ★ | Diffusion model posterior sampling |

> **GPU Note**: GPU solvers require CUDA. If your machine has no GPU, they raise a `RuntimeError`
> — this does **not** affect CPU solvers. CPU recommendation: `pnp_admm_nlm` (39.5 dB).

---

## Measurement Data

### Option A: PWM Benchmark Data (with ground truth → get PSNR & SSIM)

PWM provides 20 LoDoPaB-CT test sinograms with ground truth images. Use this to benchmark your method.

```python
# Download PWM CT benchmark data
# pip install gcsfs
import gcsfs, h5py, numpy as np
fs = gcsfs.GCSFileSystem(token='anon')
with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/ct_public.h5') as f:
    with h5py.File(f, 'r') as hf:
        y      = hf['y'][0]       # sinogram, shape (1000, 513), float32
        x_true = hf['x_true'][0]  # ground truth, shape (362, 362), float32
```

### Option B: Your Own Sinogram

Upload your sinogram as a NumPy array:
- **Shape**: `(n_angles, n_detectors)` — e.g., `(180, 256)` for 180-angle parallel-beam
- **Values**: log-attenuated projections: `-log(I / I₀)`, float32
- **Units**: dimensionless (integrated linear attenuation)

```python
y = np.load('your_sinogram.npy').astype(np.float32)
# Optional ground truth for PSNR/SSIM:
x_true = np.load('your_gt.npy').astype(np.float32)
```

---

## Parameters

Key reconstruction parameters (modify in `cfg` dict):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter` | `'ramlak'` | FBP filter: `ramlak`, `shepp_logan`, `cosine`, `hamming`, `hann` |
| `iters` | 20 | Number of iterations (for iterative solvers) |
| `lam` | 0.005 | Regularization weight (TV, Tikhonov, PnP) |
| `rho` | 1.0 | ADMM penalty parameter |
| `sigma` | 0.05 | PnP denoiser noise level |
| `circle` | False | Whether to restrict to circular FOV in FBP |

---

## Run Button

```python
# ============================================================
# CT Reconstruction — PWM Run Button
# ============================================================
import sys, os

# --- Setup (choose one) ---
# Local after git clone:
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Google Colab:
# !git clone https://github.com/integritynoble/Physics_World_Model
# BASE = '/content/Physics_World_Model/pwm/public'

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from algorithm_base.ct.solvers import run_solver, list_solvers

# --- 1. Load Data ---
# Option A: PWM benchmark data (requires gcsfs + h5py)
# import gcsfs, h5py
# fs = gcsfs.GCSFileSystem(token='anon')
# with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/ct_public.h5') as f:
#     with h5py.File(f, 'r') as hf:
#         y, x_true = hf['y'][0], hf['x_true'][0]
#         has_gt = True

# Option B: Your own data
# y = np.load('your_sinogram.npy').astype(np.float32)
# x_true = None; has_gt = False

# Option C: Synthetic (demo only)
from skimage.transform import radon
from skimage.data import shepp_logan_phantom
phantom = shepp_logan_phantom()
angles  = np.linspace(0, 180, 180, endpoint=False)
y       = radon(phantom, theta=angles).astype(np.float32)
x_true  = phantom.astype(np.float32)
has_gt  = True

# --- 2. List solvers ---
print("Available CT solvers:")
for key, info in list_solvers():
    gpu = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {key:<25} {info['name']:<30} {gpu}  {info.get('reference','')}")

# --- 3. Run reconstruction ---
SOLVER = 'traditional_cpu'   # Change to: 'pnp_admm_nlm', 'tv_admm', 'cgls', etc.
cfg    = {}                   # Optional: {'iters': 30, 'lam': 0.005}

print(f"\nRunning solver: {SOLVER} ...")
x_hat = run_solver(SOLVER, y, operator=None, cfg=cfg)
print(f"Reconstruction shape: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")

# --- 4. Evaluate (if ground truth available) ---
if has_gt and x_true is not None:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    # Normalize to same range
    ref_max = max(x_true.max(), x_hat.max())
    psnr = peak_signal_noise_ratio(x_true, x_hat, data_range=ref_max)
    ssim = structural_similarity(x_true, x_hat, data_range=ref_max)
    print(f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")

# --- 5. Visualize ---
n_panels = 3 if (has_gt and x_true is not None) else 2
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

axes[0].imshow(y, cmap='gray', aspect='auto')
axes[0].set_title('Sinogram (y)')
axes[0].set_xlabel('Detector'); axes[0].set_ylabel('Angle')

axes[1].imshow(x_hat, cmap='gray')
axes[1].set_title(f'Reconstruction ({SOLVER})')

if has_gt and x_true is not None:
    axes[2].imshow(x_true, cmap='gray')
    axes[2].set_title('Ground Truth')

plt.tight_layout()
plt.savefig('ct_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ct_reconstruction.png")
```

---

## Expected Output

After running, you will see:
1. **Reconstructed image** — 2D attenuation map (same size as phantom or 362×362 for LoDoPaB)
2. **PSNR** (if ground truth): 27–43 dB depending on solver
3. **SSIM** (if ground truth): 0.80–0.99
4. **Plots**: sinogram | reconstruction | ground truth (side by side)
5. **Saved file**: `ct_reconstruction.png`

| Solver | Expected PSNR | Expected SSIM | GPU |
|--------|--------------|---------------|-----|
| `traditional_cpu` (FBP) | ~27 dB | ~0.80 | No |
| `cgls` | ~30 dB | ~0.87 | No |
| `pnp_admm_nlm` | ~39.5 dB | ~0.96 | No |
| `indudonet` | ~43.5 dB | ~0.99 | Yes |

---

## 3D Datacube Visualization

For **Cone-Beam CT** (3D volume), use:
```python
# Show three orthogonal slices of 3D reconstruction
from algorithm_base.cbct.solvers import run_solver as run_cbct
# x_hat_3d shape: (D, H, W)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
d, h, w = x_hat_3d.shape
axes[0].imshow(x_hat_3d[d//2], cmap='gray'); axes[0].set_title('Axial')
axes[1].imshow(x_hat_3d[:, h//2, :], cmap='gray'); axes[1].set_title('Coronal')
axes[2].imshow(x_hat_3d[:, :, w//2], cmap='gray'); axes[2].set_title('Sagittal')
```

---

## References

- **Dataset**: Leuschner et al., "LoDoPaB-CT", Scientific Data 2021
- **FBP**: Ramachandran & Lakshminarayanan, Proc. Nat. Acad. Sci. 1971
- **TV-ADMM**: Sidky & Pan, Phys. Med. Biol. 2008; Boyd et al., Found. Trends ML 2010
- **PnP-ADMM**: Venkatakrishnan et al., GlobalSIP 2013
- **InDuDoNet**: Song et al., MICCAI 2021
- **Score-CT**: Song et al., ICLR 2022
