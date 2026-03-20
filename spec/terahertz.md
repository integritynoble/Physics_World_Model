# Terahertz Imaging (THz)

**Input**: THz waveform (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/terahertz/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `thz_dl` | THz-Net [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | DL-UNet |  | GPU |
| `dl_transformer` | DL-Transformer |  | GPU |
| `dl_diffusion` | DL-Diffusion |  | GPU |
| `dl_mamba` | DL-Mamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.terahertz.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # THz waveform (T × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Papers

- `papers/universal_simulation/benchmark/02_electromagnetics/`
