# Panorama Multi-Focus Fusion

**Input**: overlapping images (N × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/panorama/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Laplacian Pyramid Fusion |  | CPU |
| `best_quality` | Guided Filter Fusion |  | CPU |
| `famous_dl` | IFCNN |  | CPU |
| `small_gpu` | IFCNN |  | CPU |
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
from algorithm_base.panorama.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # overlapping images (N × H × W × 3, uint8)
x = run_solver('best_quality', y) # swap key to compare
```
