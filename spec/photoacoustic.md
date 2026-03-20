# Photoacoustic Imaging

**Input**: PA time-series (elements × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Back Projection |  | CPU |
| `best_quality` | Time Reversal [proxy] |  | CPU |
| `famous_dl` | Deep-PAT [proxy] |  | CPU |
| `small_gpu` | Deep-PAT [proxy] |  | CPU |
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
from algorithm_base.photoacoustic.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # PA time-series (elements × time, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Papers

- `papers/universal_simulation/benchmark/02_electromagnetics/`
