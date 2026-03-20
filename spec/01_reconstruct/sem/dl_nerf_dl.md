# Scanning Electron Microscopy (SEM) — NeRF-DL

**GPU**  *Neural rendering, 2020*
**Input**: SEM image (H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver
x = run_solver('dl_nerf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
