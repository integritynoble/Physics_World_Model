# Fourier Ptychographic Microscopy (FPM) — PnP-ADMM (NLM) + Gradient

**CPU**  **Mismatch**: LED position error `[-2, +2] mm`
**Input**: LED images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
from pwm_core.mismatch.operators import fpm_calibrate_led_pos

x_wrong = run_solver('pnp_admm_nlm', y)           # no correction
led_err = fpm_calibrate_led_pos(y)
calib_cfg = {"led_pos_error": led_err}
x = run_solver('pnp_admm_nlm', y, cfg={**calib_cfg, **{'iters': 20, 'sigma': 0.05, 'rho': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
