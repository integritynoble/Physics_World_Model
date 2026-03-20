# Ultrasonic Phased Array (TFM/FMC) — Tikhonov Regularization + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: FMC data (elem × elem × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/public/`

```python
from algorithm_base.ultrasonic_phased_array.solvers import run_solver


x_wrong = run_solver('tikhonov', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('tikhonov', y, cfg={**calib_cfg, **{'iters': 50, 'lam': 0.01, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
