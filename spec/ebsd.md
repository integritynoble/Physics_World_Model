# Electron Backscatter Diffraction (EBSD)

**Input**: Kikuchi pattern (H × W × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ebsd/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (Hough baseline) |  | CPU |
| `best_quality` | EBSD-DL (DictIndex) [proxy] |  | CPU |
| `famous_dl` | EMsoft-EBSD [proxy] |  | CPU |
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
from algorithm_base.ebsd.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # Kikuchi pattern (H × W × px × py, float32)
x = run_solver('best_quality', y) # swap key to compare
```
