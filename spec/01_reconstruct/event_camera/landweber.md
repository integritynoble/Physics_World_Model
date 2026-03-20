# Event Camera / Dynamic Vision Sensor (DVS) — Landweber Iteration

**CPU**  *Landweber, Am J Math 1951*
**Input**: event stream (N × 4: t,x,y,p)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/public/`

```python
from algorithm_base.event_camera.solvers import run_solver
cfg = {'iters': 50, 'step': 0.5}
x = run_solver('landweber', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
