# Optical Coherence Tomography (OCT)

**Input**: spectral interferogram (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FFT Recon |  | CPU |
| `best_quality` | Spectral Estimation |  | CPU |
| `famous_dl` | OCT Denoising Net |  | CPU |
| `small_gpu` | OCT Denoising Net |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | Med-UNet |  | GPU |
| `dl_swinir` | SwinIR-Med |  | GPU |
| `dl_diffusion` | DiffusionMed |  | GPU |
| `dl_mamba` | MedMamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.oct.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # spectral interferogram (wavenumbers × A-scans, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **dispersion coefficients (β₂, β₃)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/universal_simulation/benchmark/09_optics/`
