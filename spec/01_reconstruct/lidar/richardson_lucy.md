# LiDAR Scanner — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: point cloud (N × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lidar/public/`

```python
from algorithm_base.lidar.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
