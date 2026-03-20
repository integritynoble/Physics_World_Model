# Interferometric SAR (InSAR) — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: interferometric phase (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
