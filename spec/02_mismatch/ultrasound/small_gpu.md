# Ultrasound B-mode Imaging — US-CNN (DnCNN denoise) + Gradient

**GPU**  **Mismatch**: speed of sound `[1400, 1600] m/s`
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
from pwm_core.mismatch.operators import ultrasound_calibrate_sos

x_wrong = run_solver('small_gpu', y)           # no correction
c0 = ultrasound_calibrate_sos(y)
calib_cfg = {"c0": float(c0)}
x = run_solver('small_gpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
