# Lensless (Diffuser Camera) Imaging — Landweber Iteration

**CPU**  *Landweber L., An iteration formula for Fredholm integral equations of the first kind, American Journal of Mathematics, 1951*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('landweber', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
