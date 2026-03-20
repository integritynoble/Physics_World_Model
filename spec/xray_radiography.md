# X-ray Radiography

**Input**: attenuation image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP (X-ray radiography) |  | CPU |
| `best_quality` | CheXNet [proxy] |  | CPU |
| `famous_dl` | X-ray UNet [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | XR-UNet |  | GPU |
| `dl_swinir` | XR-SwinIR |  | GPU |
| `dl_diffusion` | XR-Diffusion |  | GPU |
| `dl_mamba` | XR-Mamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.xray_radiography.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # attenuation image (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
