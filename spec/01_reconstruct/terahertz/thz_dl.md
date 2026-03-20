# Terahertz Imaging (THz) — THz-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: THz waveform (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/terahertz/public/`

```python
from algorithm_base.terahertz.solvers import run_solver
x = run_solver('thz_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
