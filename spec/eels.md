# Electron Energy Loss Spectroscopy (EELS)

**Input**: energy-loss spectrum (H × W × E, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (Fourier ratio) [proxy] |  | CPU |
| `best_quality` | EELS-Net [proxy] |  | CPU |
| `famous_dl` | MLLS-EELS [proxy] |  | CPU |
| `eels_dl` | EELS-Net [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_cnn` | Spec-CNN |  | GPU |
| `dl_autoencoder` | Spec-AE |  | GPU |
| `dl_transformer` | Spec-Transformer |  | GPU |
| `dl_diffusion` | Spec-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.eels.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # energy-loss spectrum (H × W × E, float32)
x = run_solver('best_quality', y) # swap key to compare
```
