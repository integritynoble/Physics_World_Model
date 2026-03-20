# Transmission Electron Microscopy (TEM) — 3D-CNN + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: TEM image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

```python
from algorithm_base.tem.solvers import run_solver


x_wrong = run_solver('dl_3dcnn', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_3dcnn', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
