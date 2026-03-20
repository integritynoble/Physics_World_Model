# Second Harmonic Generation (SHG) Microscopy — Richardson-Lucy

**CPU**
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/public/`

```python
from algorithm_base.shg.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
