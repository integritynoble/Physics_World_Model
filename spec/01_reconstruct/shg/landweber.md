# Second Harmonic Generation (SHG) Microscopy — Landweber Iteration

**CPU**  *Landweber, Am J Math 1951*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/public/`

```python
from algorithm_base.shg.solvers import run_solver
cfg = {'iters': 50, 'step': 0.5}
x = run_solver('landweber', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
