# Multispectral Satellite Imaging — RDA [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: radiance (H × W × bands, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/public/`

```python
from algorithm_base.multispectral_sat.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
