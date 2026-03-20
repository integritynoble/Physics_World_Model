# Ghost Imaging — Unrolled-Net + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: bucket signal (N_patterns, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ghost_imaging/public/`

```python
from algorithm_base.ghost_imaging.solvers import run_solver


x_wrong = run_solver('dl_unrolled', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_unrolled', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
