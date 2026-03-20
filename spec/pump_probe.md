# Pump-Probe Microscopy

**Input**: transient spectra (T × wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `pp_dl` | PumpProbe-Net [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_reconnet` | ReconNet |  | GPU |
| `dl_unrolled` | Unrolled-Net |  | GPU |
| `dl_transformer` | CS-Transformer |  | GPU |
| `dl_diffusion` | CS-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.pump_probe.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # transient spectra (T × wavelengths, float32)
x = run_solver('best_quality', y) # swap key to compare
```
