# Dual-Energy X-ray Absorptiometry (DEXA) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: dual-energy projections (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/public/`

```python
from algorithm_base.dexa.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
