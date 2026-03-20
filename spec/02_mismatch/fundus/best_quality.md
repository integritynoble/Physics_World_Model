# Fundus Camera — RETFound + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: photograph (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/public/`

```python
from algorithm_base.fundus.solvers import run_solver


x_wrong = run_solver('best_quality', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('best_quality', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
