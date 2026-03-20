# Widefield Fluorescence Microscopy

**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Richardson-Lucy Deconvolution |  | CPU |
| `wiener` | Wiener Filter |  | CPU |
| `gold` | Gold Deconvolution |  | CPU |
| `jansson` | Jansson-van Cittert Iteration |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `tikhonov` | Tikhonov Regularisation |  | CPU |
| `tv_deconv` | Total Variation Deconvolution |  | CPU |
| `rl_tv` | Richardson-Lucy with TV Regularisation |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM denoiser) |  | CPU |
| `best_quality` | CARE (PnP-PGD DRUNet) |  | GPU |
| `famous_dl` | Noise2Void (PnP-PGD DRUNet) |  | GPU |
| `small_gpu` | CSBDeep (DnCNN denoise) |  | GPU |
| `restormer` | Restormer (PnP-HQS DRUNet) |  | GPU |
| `wf_diffusion` | WF-Diffusion (PnP-PGD DRUNet) |  | GPU |
| `deepcad_rt` | DeepCAD-RT (PnP-DRS DRUNet) |  | GPU |
| `wf_mamba` | WF-Mamba (RED DRUNet) |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.widefield.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # fluorescence image (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
