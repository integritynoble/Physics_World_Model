# Neural Radiance Fields (NeRF) — Richardson-Lucy (proxy baseline) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/public/`

```python
from algorithm_base.nerf.solvers import run_solver


x_wrong = run_solver('rl_proxy', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('rl_proxy', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
