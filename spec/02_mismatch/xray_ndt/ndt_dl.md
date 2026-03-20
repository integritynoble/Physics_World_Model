# X-ray NDT (Radiography) — NDT-DefectNet [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/public/`

```python
from algorithm_base.xray_ndt.solvers import run_solver


x_wrong = run_solver('ndt_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('ndt_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
