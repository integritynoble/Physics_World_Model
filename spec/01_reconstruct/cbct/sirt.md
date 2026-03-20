# Cone-Beam Computed Tomography (CBCT) — Simultaneous Iterative Reconstruction (SIRT)

**CPU**  *Gilbert, P. (1972) Iterative methods for the three-dimensional reconstruction of an object, Journal of Theoretical Biology*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('sirt', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
