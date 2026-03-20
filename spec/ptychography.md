# Ptychographic Imaging

**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `error_reduction` | Error Reduction (Fienup) |  | CPU |
| `wdd` | Wigner Distribution Deconvolution (WDD) |  | CPU |
| `difference_map` | Difference Map |  | CPU |
| `pie` | Ptychographic Iterative Engine (PIE) |  | CPU |
| `raar` | Relaxed Averaged Alternating Reflections (RAAR) |  | CPU |
| `traditional_cpu` | Extended PIE (ePIE) |  | CPU |
| `mpie` | Momentum PIE (mPIE) |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM with NLM |  | CPU |
| `best_quality` | PtychoNN (DL-PGD) |  | GPU |
| `famous_dl` | AutoPhase (DL-PGD) |  | GPU |
| `small_gpu` | PtychoNN 2.0 (DnCNN) |  | GPU |
| `ptycho_diffusion` | Ptychography Diffusion (DL-PGD) |  | GPU |
| `ptycho_former` | PtychoFormer (DL-DRS) |  | GPU |
| `ptycho_mamba` | PtychoMamba (RED-DRUNet) |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ptychography.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # diffraction patterns (N_pos × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
