# Talbot-Lau X-ray Grating Interferometry

**Input**: grating stepping images (N_steps × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `talbot_dl` | Talbot-Net [proxy] |  | CPU |
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
from algorithm_base.talbot_lau.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # grating stepping images (N_steps × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
