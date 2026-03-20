# Widefield Fluorescence Microscopy — Landweber Iteration

**CPU**  *Landweber 1951, Amer. J. Math.*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('landweber', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
