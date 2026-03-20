# Integral Photography — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: integral image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/public/`

```python
from algorithm_base.integral.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
