# PET/MR Fusion — PnP-ADMM [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: PET sino + MRI k-space (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_mr/public/`

```python
from algorithm_base.pet_mr.solvers import run_solver


x_wrong = run_solver('best_quality', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('best_quality', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
