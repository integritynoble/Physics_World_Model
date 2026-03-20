# X-ray Computed Tomography (CT) — SIRT + Gradient

**CPU**  **PSNR**: ~29.5 dB  **Mismatch**: center-of-rotation offset `[-5, +5] px`
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('sirt', y)           # no correction
cor_offset = ct_calibrate_cor(y, shift_range=5)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('sirt', y, cfg={**calib_cfg, **{'iters': 20}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
