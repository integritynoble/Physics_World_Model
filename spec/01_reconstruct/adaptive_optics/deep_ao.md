# Adaptive Optics (AO) Imaging — Deep-AO [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: wavefront sensor (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/public/`

```python
from algorithm_base.adaptive_optics.solvers import run_solver
x = run_solver('deep_ao', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
