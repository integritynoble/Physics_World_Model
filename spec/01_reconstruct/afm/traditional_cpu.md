# Atomic Force Microscopy (AFM) — Richardson-Lucy

**CPU**
**Input**: force-distance map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/public/`

```python
from algorithm_base.afm.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
