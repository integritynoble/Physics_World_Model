# Optical Coherence Tomography (OCT) — PnP-FISTA (NLM) + Gradient

**CPU**  **Mismatch**: dispersion coefficients `β₂ ∈ [-1e-27, 1e-27] s²/m`
**Input**: spectrum (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

```python
from algorithm_base.oct.solvers import run_solver
from pwm_core.mismatch.operators import oct_calibrate_dispersion

x_wrong = run_solver('pnp_fista_nlm', y)           # no correction
disp = oct_calibrate_dispersion(y)
calib_cfg = {"disp_coeff": disp}
x = run_solver('pnp_fista_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'mu': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
