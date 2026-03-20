# SPECT/CT Fusion — PnP-ADMM (NLM) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: SPECT proj + CT sino (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/public/`

```python
from algorithm_base.spect_ct.solvers import run_solver


x_wrong = run_solver('pnp_admm_nlm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pnp_admm_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'rho': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
