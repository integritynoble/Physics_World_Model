# STEM-EDX Elemental Mapping — Richardson-Lucy (DL baseline) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: X-ray counts (H × W × channels, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/public/`

```python
from algorithm_base.edx_mapping.solvers import run_solver


x_wrong = run_solver('edx_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('edx_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
