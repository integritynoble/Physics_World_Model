# Neural Radiance Fields (NeRF) — Richardson-Lucy (proxy baseline)

**CPU**  *Richardson 1972, JOSA*
**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/public/`

```python
from algorithm_base.nerf.solvers import run_solver
x = run_solver('rl_proxy', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
