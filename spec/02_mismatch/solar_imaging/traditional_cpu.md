# Solar EUV/X-ray Imaging — Adjoint [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: EUV image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/public/`

```python
from algorithm_base.solar_imaging.solvers import run_solver


x_wrong = run_solver('traditional_cpu', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
