# Streak Camera Imaging — StreakNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: streak image (time × space, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/streak_camera/public/`

```python
from algorithm_base.streak_camera.solvers import run_solver
x = run_solver('streak_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
