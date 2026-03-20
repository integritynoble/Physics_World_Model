# Structured-Light Depth Camera

**Input**: pattern images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (phase unwrap) [proxy] |  | CPU |
| `best_quality` | SL-Net [proxy] |  | CPU |
| `famous_dl` | FTPD [proxy] |  | CPU |
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
from algorithm_base.structured_light.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # pattern images (N × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
