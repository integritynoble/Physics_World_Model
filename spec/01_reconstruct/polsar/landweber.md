# Polarimetric SAR (PolSAR) — Landweber Iteration

**CPU**  *Landweber, Am J Math 1951*
**Input**: scattering matrix (H × W × 4, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

```python
from algorithm_base.polsar.solvers import run_solver
cfg = {'iters': 50, 'step': 0.5}
x = run_solver('landweber', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
