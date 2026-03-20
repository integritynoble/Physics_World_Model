# Single Photon Emission CT (SPECT) — Tikhonov Regularization + Gradient

**CPU**  **Mismatch**: scatter fraction `[0.1, 0.4]`
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
from pwm_core.mismatch.operators import spect_calibrate_scatter

x_wrong = run_solver('tikhonov', y)           # no correction
scatter_frac = spect_calibrate_scatter(y)
calib_cfg = {"scatter_frac": float(scatter_frac)}
x = run_solver('tikhonov', y, cfg={**calib_cfg, **{'iters': 50, 'lam': 0.01, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
