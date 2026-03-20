# Streak Camera Imaging — StreakNet [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: streak image (time × space, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/streak_camera/public/`

```python
from algorithm_base.streak_camera.solvers import run_solver


x_wrong = run_solver('streak_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('streak_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
