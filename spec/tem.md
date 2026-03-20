# Transmission Electron Microscopy (TEM)

**Input**: TEM image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (CTF correction) |  | CPU |
| `best_quality` | TEM-DL (ePIE-Net) [proxy] |  | CPU |
| `famous_dl` | TEM-UNet [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_3dcnn` | 3D-CNN |  | GPU |
| `dl_nerf_dl` | NeRF-DL |  | GPU |
| `dl_transformer` | 3D-Transformer |  | GPU |
| `dl_diffusion` | 3D-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.tem.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # TEM image (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
