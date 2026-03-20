# Scanning Electron Microscopy (SEM) — Richardson-Lucy (SEM)

**CPU**
**Input**: SEM image (H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
