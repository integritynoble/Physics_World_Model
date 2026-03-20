# Dual-Energy X-ray Absorptiometry (DEXA)

**Input**: dual-energy projections (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (dual-energy) |  | CPU |
| `best_quality` | DXA-Net [proxy] |  | CPU |
| `famous_dl` | DEXA-UNet [proxy] |  | CPU |
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
from algorithm_base.dexa.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # dual-energy projections (2 × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
