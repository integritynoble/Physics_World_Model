# MR Spectroscopy (MRS) — MRS-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: FID (T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
