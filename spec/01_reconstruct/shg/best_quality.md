# Second Harmonic Generation (SHG) Microscopy — CARE

**GPU**  *Weigert et al. 2018*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/public/`

```python
from algorithm_base.shg.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
