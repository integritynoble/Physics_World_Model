# Event Camera / Dynamic Vision Sensor (DVS) — CS-Transformer

**GPU**  *Transformer for CS, 2023*
**Input**: event stream (N × 4: t,x,y,p)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/public/`

```python
from algorithm_base.event_camera.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
