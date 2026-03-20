# Diffuse Optical Tomography (DOT) — Richardson-Lucy + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: boundary flux (sources × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dot/public/`

```python
from algorithm_base.dot.solvers import run_solver


x_wrong = run_solver('richardson_lucy', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('richardson_lucy', y, cfg={**calib_cfg, **{'iters': 50}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
