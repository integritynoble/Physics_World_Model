# Integral Photography — Unrolled-Net

**GPU**  *Deep unrolling for CS, 2020*
**Input**: integral image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/public/`

```python
from algorithm_base.integral.solvers import run_solver
x = run_solver('dl_unrolled', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
