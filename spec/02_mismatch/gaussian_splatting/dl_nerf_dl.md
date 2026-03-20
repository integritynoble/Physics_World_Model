# 3D Gaussian Splatting (3DGS) — NeRF-DL + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/public/`

```python
from algorithm_base.gaussian_splatting.solvers import run_solver


x_wrong = run_solver('dl_nerf_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_nerf_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
