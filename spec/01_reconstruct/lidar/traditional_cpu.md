# LiDAR Scanner — FISTA-L2 (depth)

**CPU**
**Input**: point cloud (N × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lidar/public/`

```python
from algorithm_base.lidar.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
