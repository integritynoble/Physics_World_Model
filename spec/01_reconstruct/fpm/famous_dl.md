# Fourier Ptychographic Microscopy (FPM) — Fourier Ptychnet

**CPU**  *Jiang et al. 2018, Biomed. Optics Express*
**Input**: LED images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
