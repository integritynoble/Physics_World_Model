# Cone-Beam Computed Tomography (CBCT)

**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

## Algorithms (22 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FDK Ram-Lak |  | CPU |
| `fdk_shepp_logan` | FDK Shepp-Logan |  | CPU |
| `fdk_hamming` | FDK Hamming |  | CPU |
| `fdk_hann` | FDK Hann |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `art` | Algebraic Reconstruction Technique (ART) |  | CPU |
| `sirt` | Simultaneous Iterative Reconstruction (SIRT) |  | CPU |
| `cgls` | Conjugate Gradient Least Squares (CGLS) |  | CPU |
| `sart` | Simultaneous ART (SART) |  | CPU |
| `mlem` | ML-EM |  | CPU |
| `osem` | Ordered Subsets EM (OS-EM) |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock Primal-Dual |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM with NLM |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA with NLM |  | CPU |
| `best_quality` | FDK + NLM Post-Processing |  | CPU |
| `famous_dl` | FDK-DL (DL-PGD) |  | GPU |
| `small_gpu` | CBCT-UNet (DnCNN) |  | GPU |
| `cbct_diffusion` | CBCT Diffusion (DL-PGD) |  | GPU |
| `cbct_naf` | CBCT Neural Attenuation Fields (DL-DRS) |  | GPU |
| `cbct_mamba` | CBCT-Mamba (RED-DRUNet) |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cbct.solvers import run_solver, list_solvers
list_solvers()                    # 22 algorithms
y = ...                           # projections (angles × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **geometric calibration (source-detector distance)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`
