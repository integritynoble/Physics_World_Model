# Contrast-Enhanced Ultrasound (CEUS) — US-DeepSight [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: contrast frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ceus/public/`

```python
from algorithm_base.ceus.solvers import run_solver
x = run_solver('us_dl_enhance', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
