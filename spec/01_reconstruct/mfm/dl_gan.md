# Magnetic Force Microscopy (MFM) — Probe-GAN

**GPU**  *GAN super-resolution, 2020*
**Input**: magnetic force map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mfm/public/`

```python
from algorithm_base.mfm.solvers import run_solver
x = run_solver('dl_gan', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
