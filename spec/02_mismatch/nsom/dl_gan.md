# Near-field Scanning Optical Microscopy (NSOM) — Probe-GAN + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: near-field signal (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

```python
from algorithm_base.nsom.solvers import run_solver


x_wrong = run_solver('dl_gan', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_gan', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
