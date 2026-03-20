# Cone-Beam Computed Tomography (CBCT) — Conjugate Gradient Least Squares (CGLS)

**CPU**  *Hestenes, M.R. & Stiefel, E. (1952) Methods of conjugate gradients for solving linear systems, J. Res. NBS*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('cgls', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
