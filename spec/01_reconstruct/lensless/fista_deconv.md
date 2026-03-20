# Lensless (Diffuser Camera) Imaging — FISTA Deconvolution

**CPU**  *Beck A. & Teboulle M., A Fast Iterative Shrinkage-Thresholding Algorithm, SIAM J. Imaging Sciences, 2009*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('fista_deconv', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
