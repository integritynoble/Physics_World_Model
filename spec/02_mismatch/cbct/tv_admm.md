# Cone-Beam Computed Tomography (CBCT) — TV-ADMM + Gradient

**CPU**  **Mismatch**: source-detector geometry `SAD/SDD ±5 mm`
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
from pwm_core.mismatch.operators import cbct_calibrate_offset

x_wrong = run_solver('tv_admm', y)           # no correction
geo = cbct_calibrate_offset(y)
calib_cfg = {"geometry_error": geo}
x = run_solver('tv_admm', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
