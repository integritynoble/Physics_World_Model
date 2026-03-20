# Radio Aperture Synthesis — Tikhonov Regularization + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: visibilities (baselines × freq × T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/public/`

```python
from algorithm_base.radio_astronomy.solvers import run_solver


x_wrong = run_solver('tikhonov', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('tikhonov', y, cfg={**calib_cfg, **{'iters': 50, 'lam': 0.01, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
