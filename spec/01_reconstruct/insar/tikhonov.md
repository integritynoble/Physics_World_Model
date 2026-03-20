# Interferometric SAR (InSAR) — Tikhonov Regularization

**CPU**  *Tikhonov, Soviet Math Doklady 1963*
**Input**: interferometric phase (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver
cfg = {'iters': 50, 'lam': 0.01, 'step': 0.5}
x = run_solver('tikhonov', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
