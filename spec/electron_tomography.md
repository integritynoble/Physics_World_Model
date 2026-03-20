# Electron Tomography

**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP (SIRT baseline) |  | CPU |
| `best_quality` | IMOD-SIRT-DL [proxy] |  | CPU |
| `famous_dl` | SIRT-3D [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | U-Net Recon |  | GPU |
| `dl_transformer` | TransCT |  | GPU |
| `dl_diffusion` | DiffusionRecon |  | GPU |
| `dl_mamba` | MambaRecon |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.electron_tomography.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # tilt series (angles × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
