# Image Scanning Microscopy (ISM) — TV-ADMM + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: raw stack (H_scan × W_scan × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ism/public/`

```python
from algorithm_base.ism.solvers import run_solver


x_wrong = run_solver('tv_admm', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('tv_admm', y, cfg={**calib_cfg, **{'iters': 20, 'lam': 0.005, 'rho': 1.0}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
