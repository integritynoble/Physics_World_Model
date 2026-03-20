# Structured Illumination Microscopy (SIM)

**Input**: raw SIM frames (9 × H × W: 3 angles × 3 phases)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Wiener-SIM |  | CPU |
| `best_quality` | HiFi-SIM |  | CPU |
| `famous_dl` | fairSIM (open-source) |  | CPU |
| `small_gpu` | Wiener-SIM (fast) |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_care` | CARE |  | GPU |
| `dl_n2v` | Noise2Void |  | GPU |
| `dl_restormer` | Restormer |  | GPU |
| `dl_diffusion` | DiffusionMicro |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.sim.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # raw SIM frames (9 × H × W: 3 angles × 3 phases)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **illumination pattern phase**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/system_design/outputs/sim_forward_v1_iter1.md`
- `papers/system_design/outputs/sim_reconstruction_v1_iter1.md`
