# Cone-Beam Computed Tomography (CBCT) — Landweber Iteration

**CPU**  *Landweber, L. (1951) An iteration formula for Fredholm integral equations, American Journal of Mathematics*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('landweber', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
