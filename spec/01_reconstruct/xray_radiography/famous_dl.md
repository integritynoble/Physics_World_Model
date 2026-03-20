# X-ray Radiography — X-ray UNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: attenuation image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/public/`

```python
from algorithm_base.xray_radiography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
