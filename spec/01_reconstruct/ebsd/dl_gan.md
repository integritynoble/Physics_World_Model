# Electron Backscatter Diffraction (EBSD) — Probe-GAN

**GPU**  *GAN super-resolution, 2020*
**Input**: Kikuchi pattern (H × W × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ebsd/public/`

```python
from algorithm_base.ebsd.solvers import run_solver
x = run_solver('dl_gan', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
