# Cone-Beam Computed Tomography (CBCT) — Chambolle-Pock Primal-Dual

**CPU**  *Chambolle, A. & Pock, T. (2011) A first-order primal-dual algorithm for convex problems, J. Math. Imaging Vis.*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('chambolle_pock', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
