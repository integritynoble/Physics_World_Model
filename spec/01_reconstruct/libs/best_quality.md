# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: emission spectrum (wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
