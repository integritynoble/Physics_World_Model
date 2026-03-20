# SPECT/CT Fusion — TransCT + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: SPECT proj + CT sino (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/public/`

```python
from algorithm_base.spect_ct.solvers import run_solver


x_wrong = run_solver('dl_transformer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_transformer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
