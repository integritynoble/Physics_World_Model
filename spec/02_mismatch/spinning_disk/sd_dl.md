# Spinning Disk Confocal Microscopy — SD-CARE + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/public/`

```python
from algorithm_base.spinning_disk.solvers import run_solver


x_wrong = run_solver('sd_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('sd_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
