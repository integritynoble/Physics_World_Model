# Flash LiDAR — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: range + intensity (H × W × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/public/`

```python
from algorithm_base.flash_lidar.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
