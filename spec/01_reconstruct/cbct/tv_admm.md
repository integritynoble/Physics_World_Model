# Cone-Beam Computed Tomography (CBCT) — TV-ADMM

**CPU**  *Sidky, E.Y., Kao, C.-M. & Pan, X. (2008) Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT, JXST*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('tv_admm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
