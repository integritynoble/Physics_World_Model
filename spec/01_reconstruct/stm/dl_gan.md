# Scanning Tunneling Microscopy (STM) — Probe-GAN

**GPU**  *GAN super-resolution, 2020*
**Input**: tunneling map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver
x = run_solver('dl_gan', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
