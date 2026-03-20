# Image Scanning Microscopy (ISM)

**Input**: ISM raw stack (H_scan × W_scan × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ism/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Richardson-Lucy |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `best_quality` | CARE |  | GPU |
| `ism_dl` | ISM-Reassignment-Net |  | GPU |
| `dl_care` | CARE |  | GPU |
| `dl_n2v` | Noise2Void |  | GPU |
| `dl_restormer` | Restormer |  | GPU |
| `dl_diffusion` | DiffusionMicro |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ism.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # ISM raw stack (H_scan × W_scan × px × py, float32)
x = run_solver('best_quality', y) # swap key to compare
```
