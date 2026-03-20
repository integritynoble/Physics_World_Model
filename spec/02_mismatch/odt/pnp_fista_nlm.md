# Optical Diffraction Tomography (ODT) — PnP-FISTA (NLM) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: holograms (angles × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/public/`

```python
from algorithm_base.odt.solvers import run_solver


x_wrong = run_solver('pnp_fista_nlm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pnp_fista_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'mu': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
