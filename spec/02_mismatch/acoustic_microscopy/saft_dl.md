# Scanning Acoustic Microscopy (SAM) — SAFT-DL [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: RF data (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/public/`

```python
from algorithm_base.acoustic_microscopy.solvers import run_solver


x_wrong = run_solver('saft_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('saft_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
