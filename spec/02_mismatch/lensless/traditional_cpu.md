# Lensless (Diffuser Camera) Imaging — Richardson-Lucy Deconvolution + Gradient

**CPU**  **Mismatch**: PSF shift `[-5, +5] px`
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
from pwm_core.mismatch.operators import lensless_calibrate_shift

x_wrong = run_solver('traditional_cpu', y)           # no correction
shift = lensless_calibrate_shift(y)
calib_cfg = {"psf_shift": shift}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
