# Multispectral Satellite Imaging — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: radiance (H × W × bands, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/public/`

```python
from algorithm_base.multispectral_sat.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
