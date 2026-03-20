# Fourier Ptychographic Microscopy (FPM) — Sequential Phase Retrieval

**CPU**
**Input**: LED images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
