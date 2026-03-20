# Shear-Wave Elastography — NLSI-Solver [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: displacement (H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/elastography/public/`

```python
from algorithm_base.elastography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
