# Raman Imaging / Microscopy — RamanNet [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: Raman spectra (H × W × wn, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/`

```python
from algorithm_base.raman_imaging.solvers import run_solver


x_wrong = run_solver('raman_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('raman_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
