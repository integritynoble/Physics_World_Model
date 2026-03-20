# Coded Aperture Compressive Temporal Imaging (CACTI) — EfficientSCI-T + Gradient

**CPU**  **Mismatch**: frame timing offset `[-1, +1] frames`
**Input**: coded frames (B × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/public/`

```python
from algorithm_base.cacti.solvers import run_solver
from pwm_core.mismatch.operators import cacti_calibrate_timing

x_wrong = run_solver('small_gpu', y)           # no correction
timing = cacti_calibrate_timing(y)
calib_cfg = {"timing_offset": timing}
x = run_solver('small_gpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
