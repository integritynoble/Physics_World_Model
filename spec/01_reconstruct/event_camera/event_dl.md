# Event Camera / Dynamic Vision Sensor (DVS) — E2VID+ [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: event stream (N × 4: t,x,y,p)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/public/`

```python
from algorithm_base.event_camera.solvers import run_solver
x = run_solver('event_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
