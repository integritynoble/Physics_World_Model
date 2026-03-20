# Adaptive Optics (AO) Imaging — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: wavefront sensor (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/public/`

```python
from algorithm_base.adaptive_optics.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
