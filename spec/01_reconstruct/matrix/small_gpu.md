# Generic Matrix Sensing — LISTA

**CPU**
**Input**: partial matrix (M × N, float32, NaN=missing)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/`

```python
from algorithm_base.matrix.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
