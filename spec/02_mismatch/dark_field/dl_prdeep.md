# Dark-Field Microscopy — prDeep + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: grating images (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/public/`

```python
from algorithm_base.dark_field.solvers import run_solver


x_wrong = run_solver('dl_prdeep', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_prdeep', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
