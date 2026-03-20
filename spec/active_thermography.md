# Active Thermography (IR)

**Input**: thermal sequence (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `pulsed_phase_tv` | Pulsed-Phase TV [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_cnn` | Probe-CNN |  | GPU |
| `dl_gan` | Probe-GAN |  | GPU |
| `dl_transformer` | Probe-Transformer |  | GPU |
| `dl_diffusion` | Probe-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.active_thermography.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # thermal sequence (T × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
