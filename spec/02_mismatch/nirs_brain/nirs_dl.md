# Functional Near-Infrared Spectroscopy (fNIRS) — fNIRS-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: optical signal (channels × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/public/`

```python
from algorithm_base.nirs_brain.solvers import run_solver


x_wrong = run_solver('nirs_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('nirs_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
