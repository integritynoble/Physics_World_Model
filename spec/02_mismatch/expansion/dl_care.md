# Expansion Microscopy (ExM) — CARE + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: confocal + expansion (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver


x_wrong = run_solver('dl_care', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_care', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
