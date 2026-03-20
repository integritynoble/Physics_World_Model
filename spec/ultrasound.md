# Ultrasound B-mode Imaging

**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | DAS (Delay-and-Sum) |  | CPU |
| `wiener` | Wiener Filter |  | CPU |
| `dmas` | Delay-Multiply-and-Sum |  | CPU |
| `mv_capon` | Minimum-Variance Capon Beamformer |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularisation |  | CPU |
| `tv_admm` | Total Variation ADMM |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM denoiser) |  | CPU |
| `best_quality` | DAS + NLM Post-filter |  | CPU |
| `famous_dl` | US-UNet (PnP-PGD DRUNet) |  | GPU |
| `small_gpu` | US-CNN (DnCNN denoise) |  | GPU |
| `able` | ABLE (PnP-HQS DRUNet) |  | GPU |
| `us_diffusion` | US-Diffusion (PnP-PGD DRUNet) |  | GPU |
| `us_vit` | US-ViT (PnP-DRS DRUNet) |  | GPU |
| `us_mamba` | US-Mamba (RED DRUNet) |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ultrasound.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # RF data (elements × samples, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **sound speed (m/s)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/universal_simulation/benchmark/01_classical_mechanics/`
