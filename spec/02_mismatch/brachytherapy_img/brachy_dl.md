# Brachytherapy Imaging — BrachyNet [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: dose map (H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/public/`

```python
from algorithm_base.brachytherapy_img.solvers import run_solver


x_wrong = run_solver('brachy_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('brachy_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
