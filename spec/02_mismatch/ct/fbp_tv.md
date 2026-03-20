# X-ray Computed Tomography (CT) — FBP + TV + Gradient

**CPU**  **Mismatch**: center-of-rotation offset `[-5, +5] px`
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('fbp_tv', y)           # no correction
cor_offset = ct_calibrate_cor(y, shift_range=5)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('fbp_tv', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
