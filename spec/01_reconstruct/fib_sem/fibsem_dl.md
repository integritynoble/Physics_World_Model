# Focused Ion Beam SEM (FIB-SEM) — FIB-SEM-Net

**GPU**  *Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141*
**Input**: cross-sections (Z × H × W, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

```python
from algorithm_base.fib_sem.solvers import run_solver
x = run_solver('fibsem_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
