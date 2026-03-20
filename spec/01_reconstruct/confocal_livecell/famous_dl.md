# Confocal Live-Cell Microscopy — CARE

**CPU**
**Input**: time-lapse (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/public/`

```python
from algorithm_base.confocal_livecell.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
