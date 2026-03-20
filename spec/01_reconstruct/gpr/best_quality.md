# Ground-Penetrating Radar (GPR) — SAR-DL [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: B-scan (traces × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gpr/public/`

```python
from algorithm_base.gpr.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
