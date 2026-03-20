# US/MRI Fusion — Landweber Iteration

**CPU**  *Landweber, Am J Math 1951*
**Input**: US + MRI combined data
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/us_mri/public/`

```python
from algorithm_base.us_mri.solvers import run_solver
cfg = {'iters': 50, 'step': 0.5}
x = run_solver('landweber', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
