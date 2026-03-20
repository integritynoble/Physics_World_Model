# Fourier Ptychographic Microscopy (FPM) — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: LED images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
