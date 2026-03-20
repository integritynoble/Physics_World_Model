# Scanning Tunneling Microscopy (STM) — Richardson-Lucy + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: tunneling map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver


x_wrong = run_solver('richardson_lucy', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('richardson_lucy', y, cfg={**calib_cfg, **{'iters': 50}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
