# Photoacoustic Imaging — Chambolle-Pock + Gradient

**CPU**  **Mismatch**: speed of sound `[1480, 1560] m/s`
**Input**: time-series (elements × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/public/`

```python
from algorithm_base.photoacoustic.solvers import run_solver
from pwm_core.mismatch.operators import pa_calibrate_sos

x_wrong = run_solver('chambolle_pock', y)           # no correction
c0 = pa_calibrate_sos(y)
calib_cfg = {"c0": float(c0)}
x = run_solver('chambolle_pock', y, cfg={**calib_cfg, **{'iters': 30, 'lam': 0.005}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
