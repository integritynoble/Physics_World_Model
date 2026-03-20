# Flash LiDAR — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: range + intensity (H × W × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/public/`

```python
from algorithm_base.flash_lidar.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
