# Radio Interferometry (VLBI) — PnP-ADMM (NLM) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: UV-plane data (N_baselines, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/public/`

```python
from algorithm_base.radio_interferometry.solvers import run_solver


x_wrong = run_solver('pnp_admm_nlm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pnp_admm_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'rho': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
