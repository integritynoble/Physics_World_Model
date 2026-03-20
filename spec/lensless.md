# Lensless (Diffuser Camera) Imaging

**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `wiener` | Wiener Deconvolution |  | CPU |
| `tikhonov` | Tikhonov Regularisation |  | CPU |
| `traditional_cpu` | Richardson-Lucy Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `fista_deconv` | FISTA Deconvolution |  | CPU |
| `tv_admm` | TV-ADMM Deconvolution |  | CPU |
| `admm_tv` | ADMM-TV (Lensless) |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_hqs_nlm` | PnP-HQS (NLM) |  | CPU |
| `best_quality` | FlatNet |  | GPU |
| `famous_dl` | Le-ADMM-U |  | GPU |
| `small_gpu` | FlatNet-Lite |  | GPU |
| `phlatcam` | PhlatCam |  | GPU |
| `lensless_former` | LenslessFormer |  | GPU |
| `diffuser_dm` | DiffuserDM |  | GPU |
| `l3fnet` | L3Fnet |  | GPU |
| `lens_mamba` | LensMamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.lensless.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # diffuser measurement (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **PSF shift (px)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/system_design/outputs/lensless_forward_v1_iter1.md`
- `papers/system_design/outputs/lensless_reconstruction_v1_iter1.md`
- `papers/system_design/outputs/lensless_3d_forward_v1_iter1.md`
- `papers/universal_simulation/benchmark/09_optics/`
