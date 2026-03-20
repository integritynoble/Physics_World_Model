# Interferometric SAR (InSAR) — InSAR-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: interferometric phase (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver


x_wrong = run_solver('insar_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('insar_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
