# Magnetic Resonance Imaging (MRI)

**Input**: k-space (H × W × 2 real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

## Algorithms (41 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Zero-Filled IFFT |  | CPU |
| `best_quality` | CS-MRI (Wavelet) | ~33.0 dB | CPU |
| `sense` | SENSE |  | CPU |
| `espirit` | ESPIRiT | ~34.2 dB | CPU |
| `cs_tv` | CS-MRI (TV) |  | CPU |
| `pocs` | POCS |  | CPU |
| `admm_mri` | ADMM |  | CPU |
| `conjugate_gradient` | Conjugate Gradient |  | CPU |
| `truncated_ifft` | Truncated IFFT |  | CPU |
| `gradient_descent` | Gradient Descent |  | CPU |
| `split_bregman` | Split Bregman |  | CPU |
| `pnp_admm` | PnP-ADMM |  | CPU |
| `low_rank` | Low-Rank |  | CPU |
| `ista_mri` | ISTA |  | CPU |
| `grappa_like` | GRAPPA-like |  | CPU |
| `fista_mri` | FISTA |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `homodyne` | Homodyne Detection |  | CPU |
| `nuclear_norm` | Nuclear Norm (SVT) |  | CPU |
| `proximal_gradient` | Proximal Gradient Descent |  | CPU |
| `bm3d_mri` | BM3D-MRI |  | CPU |
| `spirit_like` | SPIRiT-like |  | CPU |
| `red_mri` | RED (Regularization by Denoising) |  | CPU |
| `dictionary_learning` | Dictionary Learning MRI |  | CPU |
| `aloha` | ALOHA (Hankel Low-Rank) |  | CPU |
| `kt_sparse_sense` | k-t SPARSE-SENSE |  | CPU |
| `smash` | SMASH |  | CPU |
| `famous_dl` | MoDL | ~36.0 dB | GPU |
| `small_gpu` | MoDL (5 unrolls) |  | GPU |
| `varnet` | E2E-VarNet | ~40.5 dB | GPU |
| `unet_mri` | U-Net (fastMRI) |  | GPU |
| `dccnn` | DC-CNN |  | GPU |
| `deep_admm_net` | Deep ADMM-Net |  | GPU |
| `ista_net_plus` | ISTA-Net+ |  | GPU |
| `pnp_dncnn` | PnP-DnCNN |  | GPU |
| `score_mri` | Score-MRI (diffusion) |  | GPU |
| `cascade_net` | CascadeNet |  | GPU |
| `kiki_net` | KIKI-Net |  | GPU |
| `reconformer` | ReconFormer | ~40.1 dB | GPU |
| `mamba_recon` | MambaRecon |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.mri.solvers import run_solver, list_solvers
list_solvers()                    # 41 algorithms
y = ...                           # k-space (H × W × 2 real+imag, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **coil sensitivity maps (B1 field)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/universal_simulation/benchmark/02_electromagnetics/`
