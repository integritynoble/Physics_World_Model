# Functional Near-Infrared Spectroscopy (fNIRS)

**Input**: time-resolved optical (channels × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP [proxy] |  | CPU |
| `best_quality` | DL-Recon [proxy] |  | CPU |
| `nirs_dl` | fNIRS-Net [proxy] |  | CPU |
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
from algorithm_base.nirs_brain.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # time-resolved optical (channels × T, float32)
x = run_solver('best_quality', y) # swap key to compare
```
