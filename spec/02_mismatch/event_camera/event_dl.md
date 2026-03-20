# Event Camera / Dynamic Vision Sensor (DVS) — E2VID+ [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: event stream (N × 4: t,x,y,p)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/public/`

```python
from algorithm_base.event_camera.solvers import run_solver


x_wrong = run_solver('event_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('event_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
