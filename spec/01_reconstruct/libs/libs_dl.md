# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — LIBS-CNN [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: emission spectrum (wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver
x = run_solver('libs_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
