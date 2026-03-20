# Eddy Current Imaging — ECT-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: induced voltage (coils × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eddy_current/public/`

```python
from algorithm_base.eddy_current.solvers import run_solver
x = run_solver('ec_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
