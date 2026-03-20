# MR Elastography (MRE) — FBP [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: wave images (slices × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/public/`

```python
from algorithm_base.mr_elastography.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
