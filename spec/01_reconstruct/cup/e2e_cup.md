# Compressed Ultrafast Photography (CUP) — E2E-CUP [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: streak image (H × W_streak, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/public/`

```python
from algorithm_base.cup.solvers import run_solver
x = run_solver('e2e_cup', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
