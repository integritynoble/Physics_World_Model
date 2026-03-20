# Sonar Imaging

**Input**: echo data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sonar/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FISTA-L2 (DAS) [proxy] |  | CPU |
| `best_quality` | SonarSR-Net [proxy] |  | CPU |
| `famous_dl` | Sonar-CNN [proxy] |  | CPU |
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
from algorithm_base.sonar.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # echo data (elements × samples, float32)
x = run_solver('best_quality', y) # swap key to compare
```
