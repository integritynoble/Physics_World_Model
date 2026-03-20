# Near-field Scanning Optical Microscopy (NSOM)

**Input**: near-field signal (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

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
| `nsom_dl` | NSOM-Net |  | GPU |
| `dl_cnn` | Probe-CNN |  | GPU |
| `dl_gan` | Probe-GAN |  | GPU |
| `dl_transformer` | Probe-Transformer |  | GPU |
| `dl_diffusion` | Probe-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.nsom.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # near-field signal (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
