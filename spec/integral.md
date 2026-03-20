# Integral Photography

**Input**: integral image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Depth Estimation |  | CPU |
| `best_quality` | DIBR [proxy] |  | CPU |
| `famous_dl` | EPINet [proxy] |  | CPU |
| `small_gpu` | EPINet [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_reconnet` | ReconNet |  | GPU |
| `dl_unrolled` | Unrolled-Net |  | GPU |
| `dl_transformer` | CS-Transformer |  | GPU |
| `dl_diffusion` | CS-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.integral.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # integral image (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
