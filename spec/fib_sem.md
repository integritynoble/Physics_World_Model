# Focused Ion Beam SEM (FIB-SEM)

**Input**: cross-section stack (Z × H × W, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

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
| `fibsem_dl` | FIB-SEM-Net |  | GPU |
| `dl_3dcnn` | 3D-CNN |  | GPU |
| `dl_nerf_dl` | NeRF-DL |  | GPU |
| `dl_transformer` | 3D-Transformer |  | GPU |
| `dl_diffusion` | 3D-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.fib_sem.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # cross-section stack (Z × H × W, uint8)
x = run_solver('best_quality', y) # swap key to compare
```
