# 3D Gaussian Splatting (3DGS)

**Input**: multi-view images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | EWA Splatting |  | CPU |
| `famous_dl` | NeRF (baseline comparison) |  | CPU |
| `small_gpu` | 3DGS (compact) |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `best_quality` | 3DGS (full) |  | GPU |
| `dl_3dcnn` | 3D-CNN |  | GPU |
| `dl_nerf_dl` | NeRF-DL |  | GPU |
| `dl_transformer` | 3D-Transformer |  | GPU |
| `dl_diffusion` | 3D-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.gaussian_splatting.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # multi-view images (N × H × W × 3, float32)
x = run_solver('best_quality', y) # swap key to compare
```
