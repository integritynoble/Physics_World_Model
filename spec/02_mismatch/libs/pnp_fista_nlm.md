# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — PnP-FISTA (NLM) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: emission spectrum (wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver


x_wrong = run_solver('pnp_fista_nlm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pnp_fista_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'mu': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
