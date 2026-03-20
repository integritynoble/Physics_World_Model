# Low-Dose Widefield Microscopy — Landweber Iteration + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: photon-limited image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/public/`

```python
from algorithm_base.widefield_lowdose.solvers import run_solver


x_wrong = run_solver('landweber', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('landweber', y, cfg={**calib_cfg, **{'iters': 50, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
