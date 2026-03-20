# Electron Energy Loss Spectroscopy (EELS) — MLLS-EELS [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: energy-loss spectrum (H × W × E, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

```python
from algorithm_base.eels.solvers import run_solver


x_wrong = run_solver('famous_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('famous_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
