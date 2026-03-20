# Differential Interference Contrast (DIC) — Restormer

**GPU**  *Zamir et al., CVPR 2022*
**Input**: DIC image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dic/public/`

```python
from algorithm_base.dic.solvers import run_solver
x = run_solver('dl_restormer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
