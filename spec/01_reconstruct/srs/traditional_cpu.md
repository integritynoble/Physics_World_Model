# Stimulated Raman Scattering (SRS) Microscopy — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: SRS image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/srs/public/`

```python
from algorithm_base.srs.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
