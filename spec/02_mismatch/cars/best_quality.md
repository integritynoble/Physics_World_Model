# Coherent Anti-Stokes Raman (CARS) Microscopy — PnP-ADMM [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: CARS spectrum cube (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

```python
from algorithm_base.cars.solvers import run_solver


x_wrong = run_solver('best_quality', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('best_quality', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
