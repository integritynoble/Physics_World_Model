# Solar EUV/X-ray Imaging — SolarNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: EUV image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/public/`

```python
from algorithm_base.solar_imaging.solvers import run_solver
x = run_solver('solar_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
