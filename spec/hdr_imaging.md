# High Dynamic Range (HDR) Imaging

**Input**: multi-exposure stack (K × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `hdr_dl` | HDR-Net [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | DL-UNet |  | GPU |
| `dl_transformer` | DL-Transformer |  | GPU |
| `dl_diffusion` | DL-Diffusion |  | GPU |
| `dl_mamba` | DL-Mamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.hdr_imaging.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # multi-exposure stack (K × H × W × 3, uint8)
x = run_solver('best_quality', y) # swap key to compare
```
