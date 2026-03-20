# MR Elastography (MRE) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: wave images (slices × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/public/`

```python
from algorithm_base.mr_elastography.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
