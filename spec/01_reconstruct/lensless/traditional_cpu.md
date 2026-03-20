# Lensless (Diffuser Camera) Imaging — Richardson-Lucy Deconvolution

**CPU**  *Richardson W.H., JOSA 1972; Lucy L.B., AJ 1974*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
