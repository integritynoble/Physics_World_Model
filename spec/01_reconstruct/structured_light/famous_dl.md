# Structured-Light Depth Camera — FTPD [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: pattern images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/public/`

```python
from algorithm_base.structured_light.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
