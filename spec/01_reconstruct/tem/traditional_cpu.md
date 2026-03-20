# Transmission Electron Microscopy (TEM) — FISTA-L2 (CTF correction)

**CPU**
**Input**: TEM image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

```python
from algorithm_base.tem.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
