# Time-of-Flight Depth Camera — ReconNet

**GPU**  *DL for CS reconstruction, 2016*
**Input**: depth + amplitude (H × W × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tof_camera/public/`

```python
from algorithm_base.tof_camera.solvers import run_solver
x = run_solver('dl_reconnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
