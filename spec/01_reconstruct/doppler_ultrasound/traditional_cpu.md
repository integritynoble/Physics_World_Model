# Doppler Ultrasound — Back-Projection (Doppler)

**CPU**
**Input**: IQ data (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/public/`

```python
from algorithm_base.doppler_ultrasound.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
