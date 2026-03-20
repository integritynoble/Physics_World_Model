# Digital Holographic Microscopy — Gerchberg-Saxton + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver


x_wrong = run_solver('gerchberg_saxton', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('gerchberg_saxton', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
