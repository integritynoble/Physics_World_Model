# Industrial X-ray CT — Richardson-Lucy + Gradient

**CPU**  **Mismatch**: center-of-rotation offset `[-10, +10] px`
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/public/`

```python
from algorithm_base.industrial_ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('richardson_lucy', y)           # no correction
cor_offset = ct_calibrate_cor(y, shift_range=10)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('richardson_lucy', y, cfg={**calib_cfg, **{'iters': 50}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
