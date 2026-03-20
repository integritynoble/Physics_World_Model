# Confocal 3D Z-Stack

**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_3d/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | 3D Richardson-Lucy |  | CPU |
| `famous_dl` | CARE-3D |  | CPU |
| `small_gpu` | CARE-3D (slice-wise) |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `best_quality` | 3D CARE |  | GPU |
| `dl_care` | CARE |  | GPU |
| `dl_n2v` | Noise2Void |  | GPU |
| `dl_restormer` | Restormer |  | GPU |
| `dl_diffusion` | DiffusionMicro |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.confocal_3d.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # Z-stack (Z × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
