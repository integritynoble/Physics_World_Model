# Fluoroscopy — Med-UNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: X-ray frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/public/`

```python
from algorithm_base.fluoroscopy.solvers import run_solver


x_wrong = run_solver('dl_unet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_unet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
