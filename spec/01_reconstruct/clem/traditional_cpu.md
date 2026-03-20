# Correlative Light-Electron Microscopy (CLEM) — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: EM + fluorescence (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/public/`

```python
from algorithm_base.clem.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
