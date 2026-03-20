# Muon Tomography

**Input**: muon tracks (N × 6, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/public/`

## Algorithms (15 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP (muon tomography) |  | CPU |
| `best_quality` | POCA-DL [proxy] |  | CPU |
| `famous_dl` | EM-POCA [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_unet` | U-Net Recon |  | GPU |
| `dl_transformer` | TransCT |  | GPU |
| `dl_diffusion` | DiffusionRecon |  | GPU |
| `dl_mamba` | MambaRecon |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.muon_tomo.solvers import run_solver, list_solvers
list_solvers()                    # 15 algorithms
y = ...                           # muon tracks (N × 6, float32)
x = run_solver('best_quality', y) # swap key to compare
```
