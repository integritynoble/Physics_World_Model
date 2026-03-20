# Coherent Anti-Stokes Raman (CARS) Microscopy

**Input**: CARS spectrum (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `cars_dl` | CARS-DeepSpec [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_cnn` | Spec-CNN |  | GPU |
| `dl_autoencoder` | Spec-AE |  | GPU |
| `dl_transformer` | Spec-Transformer |  | GPU |
| `dl_diffusion` | Spec-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cars.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # CARS spectrum (H × W × λ, float32)
x = run_solver('best_quality', y) # swap key to compare
```
