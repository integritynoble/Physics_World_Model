# Structured-Light Depth Camera — Unrolled-Net

**GPU**  *Deep unrolling for CS, 2020*
**Input**: pattern images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/public/`

```python
from algorithm_base.structured_light.solvers import run_solver
x = run_solver('dl_unrolled', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
