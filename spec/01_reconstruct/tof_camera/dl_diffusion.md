# Time-of-Flight Depth Camera — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: depth + amplitude (H × W × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tof_camera/public/`

```python
from algorithm_base.tof_camera.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
