# Shearography — ShearNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: shearograms (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shearography/public/`

```python
from algorithm_base.shearography.solvers import run_solver
x = run_solver('shear_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
