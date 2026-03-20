# Pump-Probe Microscopy — PnP-ADMM (NLM) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver


x_wrong = run_solver('pnp_admm_nlm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pnp_admm_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'rho': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
