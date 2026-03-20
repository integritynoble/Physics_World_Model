# Ptychographic Imaging — Landweber Iteration

**CPU**  *Landweber, L. (1951) An iteration formula for Fredholm integral equations, American Journal of Mathematics*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('landweber', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
