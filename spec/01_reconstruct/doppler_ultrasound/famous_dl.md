# Doppler Ultrasound — Doppler CFAR [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: IQ data (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/public/`

```python
from algorithm_base.doppler_ultrasound.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
