# Single Photon Emission CT (SPECT) — PnP-FISTA (NLM) + Gradient

**CPU**  **Mismatch**: scatter fraction `[0.1, 0.4]`
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
from pwm_core.mismatch.operators import spect_calibrate_scatter

x_wrong = run_solver('pnp_fista_nlm', y)           # no correction
scatter_frac = spect_calibrate_scatter(y)
calib_cfg = {"scatter_frac": float(scatter_frac)}
x = run_solver('pnp_fista_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'mu': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
