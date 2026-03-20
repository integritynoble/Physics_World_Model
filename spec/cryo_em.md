# Cryo-EM Single Particle Analysis

**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Wiener-CTF Correction |  | CPU |
| `phase_flip` | Phase-Flip CTF Correction |  | CPU |
| `back_projection` | Back-Projection |  | CPU |
| `sirt_3d` | SIRT (Simultaneous Iterative) |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `tikhonov` | Tikhonov Regularisation |  | CPU |
| `tv_admm` | Total Variation ADMM |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) |  | CPU |
| `best_quality` | RELION (PnP-PGD DRUNet) |  | GPU |
| `cryosparc` | CryoSPARC (PnP-PGD DRUNet) |  | GPU |
| `famous_dl` | CryoDRGN (PnP-PGD DRUNet) |  | GPU |
| `cryodrgn2` | CryoDRGN2 (PnP-HQS DRUNet) |  | GPU |
| `small_gpu` | CryoAI (DnCNN denoise) |  | GPU |
| `deep_em_enhancer` | DeepEMenhancer (DRUNet denoise) |  | GPU |
| `topaz_denoise` | Topaz-Denoise (DRUNet denoise) |  | GPU |
| `cryostar` | CryoSTAR (PnP-DRS DRUNet) |  | GPU |
| `cryo_mamba` | CryoMamba (RED DRUNet) |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cryo_em.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # particle images (N × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
