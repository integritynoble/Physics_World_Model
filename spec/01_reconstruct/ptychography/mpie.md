# Ptychographic Imaging — Momentum PIE (mPIE)

**CPU**  *Maiden, A.M. et al. (2012) Further improvements to the ptychographical iterative engine, Optica*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('mpie', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
