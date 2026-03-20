# Compressed Ultrafast Photography (CUP) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: streak image (H × W_streak, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/public/`

```python
from algorithm_base.cup.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
