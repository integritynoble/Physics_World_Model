# Focused Ion Beam SEM (FIB-SEM) — NeRF-DL

**GPU**  *Neural rendering, 2020*
**Input**: cross-sections (Z × H × W, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

```python
from algorithm_base.fib_sem.solvers import run_solver
x = run_solver('dl_nerf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
