# Terahertz Imaging (THz) — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: THz waveform (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/terahertz/public/`

```python
from algorithm_base.terahertz.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
