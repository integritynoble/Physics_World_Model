# Generic Matrix Sensing

**Input**: matrix completion (M × N, float32, with NaNs)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L1 |  | CPU |
| `best_quality` | FISTA-L1 (high quality) |  | CPU |
| `famous_dl` | LISTA |  | CPU |
| `small_gpu` | LISTA |  | CPU |
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
from algorithm_base.matrix.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # matrix completion (M × N, float32, with NaNs)
x = run_solver('best_quality', y) # swap key to compare
```
