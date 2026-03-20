# XFEL Serial Femtosecond Crystallography (SFX) — Landweber Iteration + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: diffraction patterns (N_shots × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/public/`

```python
from algorithm_base.xfel_sfx.solvers import run_solver


x_wrong = run_solver('landweber', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('landweber', y, cfg={**calib_cfg, **{'iters': 50, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
