# Structured-Light Depth Camera — FISTA-L2 (phase unwrap) [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: pattern images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/public/`

```python
from algorithm_base.structured_light.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
