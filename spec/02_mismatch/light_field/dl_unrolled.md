# Light Field Imaging — Unrolled-Net + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: light field (u × v × s × t, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/light_field/public/`

```python
from algorithm_base.light_field.solvers import run_solver


x_wrong = run_solver('dl_unrolled', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_unrolled', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
