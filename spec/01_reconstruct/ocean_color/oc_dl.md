# Ocean Color Remote Sensing — OC-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: radiance (H × W × bands, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_color/public/`

```python
from algorithm_base.ocean_color.solvers import run_solver
x = run_solver('oc_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
