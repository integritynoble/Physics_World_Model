# STED Microscopy — Restormer + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: STED + confocal (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/public/`

```python
from algorithm_base.sted.solvers import run_solver


x_wrong = run_solver('dl_restormer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_restormer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
