# Fourier Ptychographic Microscopy (FPM)

**Input**: LED array images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Sequential Phase Retrieval |  | CPU |
| `best_quality` | Gradient Descent FPM [proxy] |  | CPU |
| `famous_dl` | Fourier Ptychnet |  | CPU |
| `small_gpu` | Fourier Ptychnet |  | CPU |
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
from algorithm_base.fpm.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # LED array images (N_leds × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
