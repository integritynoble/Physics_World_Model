# Cathodoluminescence (CL) Imaging — Spec-AE + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: spectrum image (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/public/`

```python
from algorithm_base.cathodoluminescence.solvers import run_solver


x_wrong = run_solver('dl_autoencoder', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_autoencoder', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
