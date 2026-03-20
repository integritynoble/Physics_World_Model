# Focused Ion Beam SEM (FIB-SEM) — Chambolle-Pock + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: cross-sections (Z × H × W, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

```python
from algorithm_base.fib_sem.solvers import run_solver


x_wrong = run_solver('chambolle_pock', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('chambolle_pock', y, cfg={**calib_cfg, **{'iters': 30, 'lam': 0.005}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
