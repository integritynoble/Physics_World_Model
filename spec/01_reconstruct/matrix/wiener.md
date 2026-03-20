# Generic Matrix Sensing — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: partial matrix (M × N, float32, NaN=missing)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/`

```python
from algorithm_base.matrix.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
