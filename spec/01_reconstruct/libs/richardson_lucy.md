# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: emission spectrum (wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
