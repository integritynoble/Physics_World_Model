# Panorama Multi-Focus Fusion — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: images (N × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/panorama/public/`

```python
from algorithm_base.panorama.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
