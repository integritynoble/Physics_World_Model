# Sonar Imaging — Wiener Deconvolution + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: echo data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sonar/public/`

```python
from algorithm_base.sonar.solvers import run_solver


x_wrong = run_solver('wiener', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('wiener', y, cfg={**calib_cfg, **{'reg': 0.01}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
