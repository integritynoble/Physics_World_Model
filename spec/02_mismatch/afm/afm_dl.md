# Atomic Force Microscopy (AFM) — AFM-UNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: force-distance map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/public/`

```python
from algorithm_base.afm.solvers import run_solver


x_wrong = run_solver('afm_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('afm_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
