# Cone-Beam Computed Tomography (CBCT) — Ordered Subsets EM (OS-EM)

**CPU**  *Hudson, H.M. & Larkin, R.S. (1994) Accelerated image reconstruction using ordered subsets of projection data, IEEE TMI*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('osem', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
