# TIRF Microscopy — Restormer + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: TIRF frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tirf/public/`

```python
from algorithm_base.tirf.solvers import run_solver


x_wrong = run_solver('dl_restormer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_restormer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
