# TIRF Microscopy — TIRF-SRRF [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: TIRF frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tirf/public/`

```python
from algorithm_base.tirf.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
