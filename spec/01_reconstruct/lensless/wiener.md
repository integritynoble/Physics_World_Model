# Lensless (Diffuser Camera) Imaging — Wiener Deconvolution

**CPU**  *Wiener N., Extrapolation, Interpolation, and Smoothing of Stationary Time Series, MIT Press, 1949*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('wiener', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
