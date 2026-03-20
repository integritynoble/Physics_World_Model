# TIRF Microscopy

**Input**: TIRF frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tirf/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Richardson-Lucy (TIRF) |  | CPU |
| `best_quality` | TIRF-Net (CARE) [proxy] |  | CPU |
| `famous_dl` | TIRF-SRRF [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_care` | CARE |  | GPU |
| `dl_n2v` | Noise2Void |  | GPU |
| `dl_restormer` | Restormer |  | GPU |
| `dl_diffusion` | DiffusionMicro |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.tirf.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # TIRF frames (T × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
