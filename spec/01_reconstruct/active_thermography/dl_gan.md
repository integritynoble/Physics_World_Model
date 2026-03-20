# Active Thermography (IR) — Probe-GAN

**GPU**  *GAN super-resolution, 2020*
**Input**: thermal sequence (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

```python
from algorithm_base.active_thermography.solvers import run_solver
x = run_solver('dl_gan', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
