# Doppler Ultrasound

**Input**: IQ data (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Back-Projection (Doppler) |  | CPU |
| `best_quality` | UDoppler-Net [proxy] |  | CPU |
| `famous_dl` | Doppler CFAR [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | Med-UNet |  | GPU |
| `dl_swinir` | SwinIR-Med |  | GPU |
| `dl_diffusion` | DiffusionMed |  | GPU |
| `dl_mamba` | MedMamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.doppler_ultrasound.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # IQ data (H × W × T, float32)
x = run_solver('best_quality', y) # swap key to compare
```
