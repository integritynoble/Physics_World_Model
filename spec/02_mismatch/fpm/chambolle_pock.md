# Fourier Ptychographic Microscopy (FPM) — Chambolle-Pock + Gradient

**CPU**  **Mismatch**: LED position error `[-2, +2] mm`
**Input**: LED images (N_leds × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
from pwm_core.mismatch.operators import fpm_calibrate_led_pos

x_wrong = run_solver('chambolle_pock', y)           # no correction
led_err = fpm_calibrate_led_pos(y)
calib_cfg = {"led_pos_error": led_err}
x = run_solver('chambolle_pock', y, cfg={**calib_cfg, **{'iters': 30, 'lam': 0.005}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
