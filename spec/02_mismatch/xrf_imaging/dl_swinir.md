# X-ray Fluorescence (XRF) Imaging — XR-SwinIR + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: fluorescence map (H × W × elements, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/public/`

```python
from algorithm_base.xrf_imaging.solvers import run_solver


x_wrong = run_solver('dl_swinir', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_swinir', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
