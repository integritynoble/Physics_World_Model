# 4D-STEM Electron Diffraction

**Input**: diffraction pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | ePIE (electron ptychography) |  | CPU |
| `best_quality` | ED-Net [proxy] |  | CPU |
| `famous_dl` | CRISP-ED [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_phasenet` | PhaseNet |  | GPU |
| `dl_prdeep` | prDeep |  | GPU |
| `dl_transformer` | Phase-Transformer |  | GPU |
| `dl_diffusion` | Phase-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.electron_diffraction.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # diffraction pattern (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
