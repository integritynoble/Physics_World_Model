# Coded Exposure / Flutter Shutter — Adjoint [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: coded frames (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/public/`

```python
from algorithm_base.coded_exposure.solvers import run_solver


x_wrong = run_solver('traditional_cpu', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
