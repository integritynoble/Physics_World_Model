# Focused Ion Beam SEM (FIB-SEM) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: cross-sections (Z × H × W, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

```python
from algorithm_base.fib_sem.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
