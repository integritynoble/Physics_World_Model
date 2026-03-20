# Ground-Penetrating Radar (GPR) — GPR-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: B-scan (traces × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gpr/public/`

```python
from algorithm_base.gpr.solvers import run_solver
x = run_solver('gpr_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
