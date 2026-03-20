# Brillouin Microscopy — Spec-AE + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: spectral shift map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brillouin/public/`

```python
from algorithm_base.brillouin.solvers import run_solver


x_wrong = run_solver('dl_autoencoder', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_autoencoder', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
