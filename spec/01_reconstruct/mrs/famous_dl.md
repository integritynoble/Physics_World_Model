# MR Spectroscopy (MRS) — HLSVD-MRS [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: FID (T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
