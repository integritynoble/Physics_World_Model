# Lensless (Diffuser Camera) Imaging — Tikhonov Regularisation

**CPU**  *Tikhonov A.N., Solution of incorrectly formulated problems and the regularization method, Soviet Mathematics Doklady, 1963*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('tikhonov', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
