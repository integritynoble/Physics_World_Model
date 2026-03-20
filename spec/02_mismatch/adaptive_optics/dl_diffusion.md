# Adaptive Optics (AO) Imaging — DL-Diffusion + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: wavefront sensor (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/public/`

```python
from algorithm_base.adaptive_optics.solvers import run_solver


x_wrong = run_solver('dl_diffusion', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_diffusion', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
