# PALM/STORM Single-Molecule Localization

**Input**: localisation list (N × 4: x,y,σ,intensity)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Richardson-Lucy (STORM/PALM) |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `best_quality` | DECODE-SMLM |  | GPU |
| `famous_dl` | DeepSTORM |  | GPU |
| `dl_care` | CARE |  | GPU |
| `dl_n2v` | Noise2Void |  | GPU |
| `dl_restormer` | Restormer |  | GPU |
| `dl_diffusion` | DiffusionMicro |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.palm_storm.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # localisation list (N × 4: x,y,σ,intensity)
x = run_solver('best_quality', y) # swap key to compare
```
