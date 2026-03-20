# Intravascular Ultrasound (IVUS) — TV-ADMM + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: RF pullback (frames × elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/public/`

```python
from algorithm_base.ivus.solvers import run_solver


x_wrong = run_solver('tv_admm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('tv_admm', y, cfg={**calib_cfg, **{'iters': 20, 'lam': 0.005, 'rho': 1.0}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
