# Shear-Wave Elastography — Richardson-Lucy + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: displacement (H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/elastography/public/`

```python
from algorithm_base.elastography.solvers import run_solver


x_wrong = run_solver('richardson_lucy', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('richardson_lucy', y, cfg={**calib_cfg, **{'iters': 50}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
