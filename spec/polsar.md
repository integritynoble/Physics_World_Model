# Polarimetric SAR (PolSAR)

**Input**: scattering matrix (H × W × 4, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | RDA [proxy] |  | CPU |
| `best_quality` | SAR-DL [proxy] |  | CPU |
| `polsar_dl` | PolSAR-Net [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_cnn` | RS-CNN |  | GPU |
| `dl_transformer` | RS-Transformer |  | GPU |
| `dl_diffusion` | RS-Diffusion |  | GPU |
| `dl_mamba` | RS-Mamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.polsar.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # scattering matrix (H × W × 4, complex64)
x = run_solver('best_quality', y) # swap key to compare
```
