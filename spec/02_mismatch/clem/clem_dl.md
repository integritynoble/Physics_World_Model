# Correlative Light-Electron Microscopy (CLEM) — CLEM-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: EM + fluorescence (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/public/`

```python
from algorithm_base.clem.solvers import run_solver


x_wrong = run_solver('clem_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('clem_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
