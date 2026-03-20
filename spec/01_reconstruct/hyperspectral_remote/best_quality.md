# Hyperspectral Remote Sensing — SAR-DL [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: radiance cube (H × W × bands, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/public/`

```python
from algorithm_base.hyperspectral_remote.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
