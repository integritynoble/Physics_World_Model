# Active Thermography (IR) — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: thermal sequence (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

```python
from algorithm_base.active_thermography.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
